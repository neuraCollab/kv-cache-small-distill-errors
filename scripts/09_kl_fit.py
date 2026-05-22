"""Phase 9: подгонка функционального закона KL(t) для AR-режима.

Берём AR-trajectory KL из outputs/kv_capture/.../analysis/logit_kl_trajectory_*.npz,
вычисляем median KL по шагам декода, подгоняем три модели:

  1. Linear:        KL(t) = a*t + b
  2. Power-law:     KL(t) = c * t^α
  3. Saturating exp: KL(t) = K∞ * (1 - exp(-t/τ))

Сравниваем R² и AIC. Сохраняем коэффициенты и residuals в JSON, рисуем
overlay-plot с тремя fit'ами поверх данных.

В отчёте печатается лучшая модель по AIC + интерпретация τ (время до plateau).

Usage:
    python scripts/09_kl_fit.py --quant fp8_e4m3
    python scripts/09_kl_fit.py --quant fp8_e4m3 --no-plots
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("phase9")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 9: fit KL trajectory")
    p.add_argument(
        "--analysis-dir", type=Path,
        default=Path("outputs/kv_capture/qwen3-1.7b/analysis"),
    )
    p.add_argument("--quant", default="fp8_e4m3", choices=["fp8_e4m3", "fp8_e5m2"])
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


# --- модели ---


def linear(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * t + b


def power_law(t: np.ndarray, c: float, alpha: float) -> np.ndarray:
    return c * np.power(np.maximum(t, 1e-9), alpha)


def saturating_exp(t: np.ndarray, K_inf: float, tau: float) -> np.ndarray:
    return K_inf * (1 - np.exp(-t / tau))


def fit_and_score(t: np.ndarray, y: np.ndarray, model_fn, p0: tuple, name: str) -> dict:
    """Fit `y = model_fn(t, *params)`, return params + R² + AIC."""
    from scipy.optimize import curve_fit
    try:
        params, _ = curve_fit(model_fn, t, y, p0=p0, maxfev=20000)
    except Exception as e:
        return {"name": name, "error": str(e)}
    y_pred = model_fn(t, *params)
    residuals = y - y_pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    n = len(t)
    k = len(params)
    # AIC for least-squares with Gaussian residuals:
    # AIC = n*ln(RSS/n) + 2k
    aic = n * np.log(ss_res / n + 1e-12) + 2 * k
    return {
        "name": name,
        "params": [float(p) for p in params],
        "r2": float(r2),
        "aic": float(aic),
        "rss": ss_res,
        "n_params": k,
    }


def main() -> int:
    args = parse_args()

    kl_path = args.analysis_dir / f"logit_kl_trajectory_{args.quant}.npz"
    if not kl_path.exists():
        log.error("Missing %s — запустите 08_divergence_mechanism.py --mode ar", kl_path)
        return 2

    kl = np.load(kl_path)["kl"]  # [n_problems, 301]
    log.info("Loaded %s, shape %s", kl_path, kl.shape)

    # Median across problems per step, full 301 positions (-150..+150 relative to FDP)
    median = np.nanmedian(kl, axis=0)
    mean = np.nanmean(kl, axis=0)
    n_valid = (~np.isnan(kl)).sum(axis=0)

    # Use steps where we have decent coverage (n >= 30 problems)
    mask = n_valid >= 30
    # t = decode step number (0-based from start of generation = position 0 in window)
    t_all = np.arange(301)
    # We want only "growing" portion: from step 0 (window start) up to FDP-region.
    # Median KL crosses 1.0 around step 39, plateaus around step 100.
    # Fit on the rising portion: steps 0..150 (covers from start to FDP).
    fit_mask = mask & (t_all <= 150)
    t = t_all[fit_mask].astype(np.float64)
    y_median = median[fit_mask]
    y_mean = mean[fit_mask]

    log.info("Fitting %d steps (from step %d to %d)", len(t), int(t.min()), int(t.max()))

    # Try fits on median (more robust to early-EOS outliers)
    fits = []
    # Initial guesses based on observed: KL plateau ≈ 18, τ ≈ 30 steps
    fits.append(fit_and_score(t, y_median, linear, p0=(0.1, 0.0), name="linear"))
    fits.append(fit_and_score(t, y_median, power_law, p0=(0.1, 1.0), name="power_law"))
    fits.append(fit_and_score(t, y_median, saturating_exp, p0=(18.0, 30.0),
                              name="saturating_exp"))

    summary = {
        "quant": args.quant,
        "n_steps": len(t),
        "t_range": [int(t.min()), int(t.max())],
        "fits_on_median": fits,
    }

    # Pick best by AIC
    valid = [f for f in fits if "error" not in f]
    if valid:
        best = min(valid, key=lambda f: f["aic"])
        summary["best_model"] = best["name"]
        summary["best_params"] = best["params"]
        summary["best_r2"] = best["r2"]
        summary["best_aic"] = best["aic"]
        log.info("=== Results ===")
        for f in fits:
            if "error" in f:
                log.warning("  %s: FAILED (%s)", f["name"], f["error"])
                continue
            log.info("  %s: R²=%.4f, AIC=%.1f, params=%s",
                     f["name"], f["r2"], f["aic"],
                     [f"{p:.4f}" for p in f["params"]])
        log.info("→ Best fit: %s (AIC=%.1f, R²=%.4f)",
                 best["name"], best["aic"], best["r2"])

        # Interpretive output
        if best["name"] == "saturating_exp":
            K_inf, tau = best["params"]
            log.info("  K∞ = %.3f (asymptotic divergence)", K_inf)
            log.info("  τ  = %.2f decode steps (time to reach 1-1/e ≈ 63%% of K∞)",
                     tau)
            log.info("  → trajectories diverge with characteristic time of %.1f tokens", tau)
        elif best["name"] == "power_law":
            c, alpha = best["params"]
            log.info("  KL(t) ≈ %.4f · t^%.2f", c, alpha)
            if alpha < 1.2:
                log.info("  → roughly linear/sub-linear growth")
            elif alpha < 2.2:
                log.info("  → quadratic-ish growth (consistent with compounding error)")
            else:
                log.info("  → super-quadratic growth")

    # Save JSON
    args.analysis_dir.mkdir(exist_ok=True)
    out_path = args.analysis_dir / f"kl_fit_{args.quant}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("Saved %s", out_path)

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not installed, skipping plot")
            return 0
        plot_dir = args.analysis_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(11, 6))
        # Raw data: median and shaded IQR
        p25 = np.nanpercentile(kl, 25, axis=0)
        p75 = np.nanpercentile(kl, 75, axis=0)
        x_full = t_all - 150  # offset from FDP
        ax.fill_between(x_full[mask], p25[mask], p75[mask],
                        color="lightgray", alpha=0.5, label="IQR (data)")
        ax.plot(x_full[mask], median[mask], "k.", label="median KL", markersize=3)
        # Overlays
        x_fit = t - 150
        for f in fits:
            if "error" in f:
                continue
            if f["name"] == "linear":
                y_fit = linear(t, *f["params"])
            elif f["name"] == "power_law":
                y_fit = power_law(t, *f["params"])
            else:
                y_fit = saturating_exp(t, *f["params"])
            ax.plot(x_fit, y_fit, label=f"{f['name']} (R²={f['r2']:.3f})", linewidth=2)
        ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="vLLM FDP")
        ax.set_xlabel("Step offset from vLLM FDP (decode step − 150)")
        ax.set_ylabel("KL(bf16 || fp8) [nats]")
        ax.set_title(f"KL trajectory fit — {args.quant} (median across 80 problems)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = plot_dir / f"kl_fit_{args.quant}.png"
        plt.savefig(out_png, dpi=100)
        plt.close()
        log.info("Saved %s", out_png)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
