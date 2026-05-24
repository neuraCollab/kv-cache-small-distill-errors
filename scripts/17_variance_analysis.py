"""Phase 17: variance analysis across multi-seed AR captures (Phase 16).

Загружает outputs/kv_capture/<model>_multiseed/seed{S}/<quant>_ar/<pid>.safetensors
и считает variance основных метрик статьи через несколько seeds.

Метрики:
  1. KL trajectory shape variance: per-position KL(bf16_seed || fp8_seed),
     align по relative position, агрегировать ± CI через seeds.
  2. KL fit τ variance: fit saturating exp per seed, среднее ± std для τ.
  3. Outlier-channel stability: top-10 channel set Jaccard overlap
     между seeds (per problem, per layer).
  4. Per-layer K-noise variance: для каждого слоя CI mean K-noise.
  5. Trajectory FDP variance: position первого argmax flip per (problem, seed),
     std across seeds → robustness FDP location.

Outputs:
  variance_summary_<quant>.json — все CI's
  plots/variance_trajectory_<quant>.png — KL trajectory with CI bands

Usage:
    python scripts/17_variance_analysis.py --quant fp8_e4m3
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from kvtrace.capture.analysis import (
    logit_kl_trajectory,
    captures_share_window,
    outlier_channel_impact,
    per_position_kv_quant_noise,
)
from kvtrace.capture.storage import load_capture

log = logging.getLogger("phase17")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 17: variance analysis")
    p.add_argument("--multiseed-root", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b_multiseed"))
    p.add_argument("--quant", default="fp8_e4m3", choices=["fp8_e4m3", "fp8_e5m2"])
    p.add_argument("--ref-quant", default="bf16",
                   help="Reference quant for KL comparison (default bf16)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _discover_seeds(root: Path) -> list[int]:
    return sorted(int(p.name[4:]) for p in root.glob("seed*") if p.is_dir())


def _load_pair(root: Path, seed: int, ref_quant: str, quant: str, pid: int):
    """Load (bf16, fp8) pair for given seed/problem. Returns (None, None) if missing."""
    a = root / f"seed{seed}" / f"{ref_quant}_ar" / f"{pid}.safetensors"
    b = root / f"seed{seed}" / f"{quant}_ar" / f"{pid}.safetensors"
    if not a.exists() or not b.exists():
        return None, None
    return load_capture(a), load_capture(b)


def _saturating_exp_fit(t, y):
    """Fit y = K_inf * (1 - exp(-t/tau)). Returns (K_inf, tau, r2) or None."""
    try:
        from scipy.optimize import curve_fit
        def fn(t, K_inf, tau):
            return K_inf * (1 - np.exp(-t / tau))
        params, _ = curve_fit(fn, t, y, p0=(20.0, 50.0), maxfev=10000)
        y_pred = fn(t, *params)
        ss_res = float(np.sum((y - y_pred)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        return float(params[0]), float(params[1]), float(r2)
    except Exception:
        return None


def analyze_kl_trajectory_variance(root, seeds, ref_quant, quant) -> dict:
    """Per-seed KL trajectory aligned by relative position, then CI across seeds."""
    log.info("=== KL trajectory variance ===")
    # For each (seed, problem): trajectory KL[position] from start of generation
    per_seed_trajectories: dict[int, list[np.ndarray]] = {s: [] for s in seeds}
    common_problems = None

    for seed in seeds:
        seed_dir = root / f"seed{seed}" / f"{quant}_ar"
        if not seed_dir.exists():
            log.warning("Missing %s", seed_dir)
            continue
        pids = sorted(int(p.stem) for p in seed_dir.glob("*.safetensors"))
        if common_problems is None:
            common_problems = set(pids)
        else:
            common_problems &= set(pids)

    if not common_problems:
        log.error("No problems with all seeds available")
        return {}
    common_problems = sorted(common_problems)
    log.info("Common problems across seeds: %d", len(common_problems))

    for seed in seeds:
        for pid in common_problems:
            cap_a, cap_b = _load_pair(root, seed, ref_quant, quant, pid)
            if cap_a is None:
                continue
            try:
                kl = logit_kl_trajectory(cap_a, cap_b).numpy()
                per_seed_trajectories[seed].append(kl)
            except Exception as e:
                log.warning("Skip seed=%d pid=%d: %s", seed, pid, e)
    # Stack per seed: align by relative position 0
    # Each trajectory starts at position 0 (start of generation). Length = min(W, W_other).
    # Pad to max length with NaN.
    all_arrays = []
    for seed in seeds:
        trajs = per_seed_trajectories[seed]
        if not trajs:
            continue
        max_len = max(t.shape[0] for t in trajs)
        padded = np.full((len(trajs), max_len), np.nan, dtype=np.float32)
        for i, t in enumerate(trajs):
            padded[i, :t.shape[0]] = t
        all_arrays.append(padded)

    if not all_arrays:
        return {}
    max_len = max(a.shape[1] for a in all_arrays)
    # Pad each seed's array to common max_len
    def pad_to(arr, L):
        if arr.shape[1] == L:
            return arr
        out = np.full((arr.shape[0], L), np.nan, dtype=np.float32)
        out[:, :arr.shape[1]] = arr
        return out
    padded_arrays = [pad_to(a, max_len) for a in all_arrays]
    # Per-seed mean per position: [n_seeds, max_len]
    per_seed_mean = np.stack(
        [np.nanmean(a, axis=0) for a in padded_arrays], axis=0
    )
    # Variance across seeds, per position
    mean_across_seeds = np.nanmean(per_seed_mean, axis=0)
    std_across_seeds = np.nanstd(per_seed_mean, axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean_across_seeds)
    sem = std_across_seeds / np.sqrt(len(seeds))

    # 95% CI (normal approx)
    ci_low = mean_across_seeds - 1.96 * sem
    ci_high = mean_across_seeds + 1.96 * sem

    # Fit saturating exp per seed (using fit window 0..min(150, max_len))
    fit_end = min(150, max_len)
    fits = []
    for seed, sm in zip(seeds, per_seed_mean):
        t = np.arange(fit_end, dtype=np.float64)
        y = sm[:fit_end]
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            fits.append({"seed": seed, "K_inf": None, "tau": None, "r2": None})
            continue
        res = _saturating_exp_fit(t[mask], y[mask])
        if res:
            K_inf, tau, r2 = res
            fits.append({"seed": seed, "K_inf": K_inf, "tau": tau, "r2": r2})
        else:
            fits.append({"seed": seed, "K_inf": None, "tau": None, "r2": None})

    valid = [f for f in fits if f["tau"] is not None]
    summary = {
        "n_seeds_with_data": int(len(all_arrays)),
        "n_common_problems": len(common_problems),
        "trajectory_mean": mean_across_seeds.tolist(),
        "trajectory_std_across_seeds": std_across_seeds.tolist(),
        "trajectory_ci_low": ci_low.tolist(),
        "trajectory_ci_high": ci_high.tolist(),
        "per_seed_fits": fits,
    }
    if valid:
        taus = [f["tau"] for f in valid]
        kinfs = [f["K_inf"] for f in valid]
        r2s = [f["r2"] for f in valid]
        summary["tau_mean"] = float(np.mean(taus))
        summary["tau_std"] = float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0
        summary["tau_ci_low"] = float(np.mean(taus) - 1.96 * np.std(taus, ddof=1) / np.sqrt(len(taus))) if len(taus) > 1 else float(taus[0])
        summary["tau_ci_high"] = float(np.mean(taus) + 1.96 * np.std(taus, ddof=1) / np.sqrt(len(taus))) if len(taus) > 1 else float(taus[0])
        summary["Kinf_mean"] = float(np.mean(kinfs))
        summary["Kinf_std"] = float(np.std(kinfs, ddof=1)) if len(kinfs) > 1 else 0.0
        summary["r2_mean"] = float(np.mean(r2s))
        log.info("  τ = %.1f ± %.1f (n=%d seeds), K∞ = %.2f ± %.2f, R² = %.3f",
                 summary["tau_mean"], summary["tau_std"], len(valid),
                 summary["Kinf_mean"], summary["Kinf_std"], summary["r2_mean"])
    return summary


def analyze_outlier_stability(root, seeds, quant) -> dict:
    """Top-10 outlier channel stability: Jaccard overlap across seeds."""
    log.info("=== Outlier channel stability ===")
    # For each problem, for each seed, compute top-10 outlier channels per layer
    per_problem_overlap: dict[int, list[float]] = {}
    fractions: list[float] = []

    common_problems = None
    for seed in seeds:
        seed_dir = root / f"seed{seed}" / f"{quant}_ar"
        if not seed_dir.exists():
            continue
        pids = sorted(int(p.stem) for p in seed_dir.glob("*.safetensors"))
        if common_problems is None:
            common_problems = set(pids)
        else:
            common_problems &= set(pids)
    if not common_problems:
        return {}
    common_problems = sorted(common_problems)

    for pid in common_problems:
        seed_caps = {}
        for seed in seeds:
            p = root / f"seed{seed}" / f"{quant}_ar" / f"{pid}.safetensors"
            if p.exists():
                seed_caps[seed] = load_capture(p)
        if len(seed_caps) < 2:
            continue
        # Per-layer top-10 channels per seed
        n_layers = len(next(iter(seed_caps.values())).k_pre)
        per_layer_overlap = []
        per_layer_frac = []
        for layer in range(n_layers):
            sets_by_seed = {}
            fracs_by_seed = {}
            for seed, cap in seed_caps.items():
                impact = outlier_channel_impact(cap, top_n_channels=10)
                top10 = {(c["head"], c["channel"]) for c in impact[layer]["top_channels"]}
                sets_by_seed[seed] = top10
                fracs_by_seed[seed] = impact[layer]["top_n_fraction"]
            # Jaccard across all pairs of seeds
            seeds_list = list(sets_by_seed)
            jacs = []
            for i in range(len(seeds_list)):
                for j in range(i+1, len(seeds_list)):
                    a, b = sets_by_seed[seeds_list[i]], sets_by_seed[seeds_list[j]]
                    inter = len(a & b)
                    union = len(a | b)
                    jacs.append(inter / union if union > 0 else 1.0)
            per_layer_overlap.append(np.mean(jacs))
            per_layer_frac.append(np.mean(list(fracs_by_seed.values())))
        per_problem_overlap[pid] = list(per_layer_overlap)
        fractions.extend(per_layer_frac)

    if not per_problem_overlap:
        return {}
    all_overlaps = np.array([o for lst in per_problem_overlap.values() for o in lst])
    summary = {
        "n_problems": len(per_problem_overlap),
        "top10_jaccard_mean": float(np.mean(all_overlaps)),
        "top10_jaccard_std": float(np.std(all_overlaps, ddof=1)),
        "top10_jaccard_median": float(np.median(all_overlaps)),
        "top10_jaccard_q25": float(np.percentile(all_overlaps, 25)),
        "top10_jaccard_q75": float(np.percentile(all_overlaps, 75)),
        "top10_fraction_mean": float(np.mean(fractions)),
        "top10_fraction_std": float(np.std(fractions, ddof=1)),
    }
    log.info("  top-10 channel Jaccard across seeds: median=%.3f, IQR=[%.3f, %.3f]",
             summary["top10_jaccard_median"], summary["top10_jaccard_q25"],
             summary["top10_jaccard_q75"])
    log.info("  top-10 fraction: %.1f%% ± %.1f%% (seed-wise)",
             100*summary["top10_fraction_mean"], 100*summary["top10_fraction_std"])
    return summary


def analyze_layer_noise_variance(root, seeds, quant) -> dict:
    """Per-layer K-noise variance across seeds."""
    log.info("=== Per-layer K-noise variance ===")
    common_problems = None
    for seed in seeds:
        seed_dir = root / f"seed{seed}" / f"{quant}_ar"
        if not seed_dir.exists():
            continue
        pids = sorted(int(p.stem) for p in seed_dir.glob("*.safetensors"))
        if common_problems is None:
            common_problems = set(pids)
        else:
            common_problems &= set(pids)
    if not common_problems:
        return {}
    common_problems = sorted(common_problems)
    # Per (seed, layer): mean K-noise across problems
    n_layers = None
    per_seed_layer_noise: dict[int, np.ndarray] = {}
    for seed in seeds:
        noises = []
        for pid in common_problems:
            p = root / f"seed{seed}" / f"{quant}_ar" / f"{pid}.safetensors"
            if not p.exists():
                continue
            cap = load_capture(p)
            n = per_position_kv_quant_noise(cap)
            k_noise = n["k_noise"].numpy()  # [n_layers, W]
            mean_per_layer = np.nanmean(k_noise, axis=1)
            noises.append(mean_per_layer)
            if n_layers is None:
                n_layers = k_noise.shape[0]
        if noises:
            per_seed_layer_noise[seed] = np.stack(noises).mean(axis=0)

    if not per_seed_layer_noise:
        return {}
    arr = np.stack(list(per_seed_layer_noise.values()))  # [n_seeds, n_layers]
    summary = {
        "n_seeds": int(arr.shape[0]),
        "n_layers": int(arr.shape[1]),
        "per_layer_mean": arr.mean(axis=0).tolist(),
        "per_layer_std": arr.std(axis=0, ddof=1).tolist() if arr.shape[0] > 1 else [0.0]*arr.shape[1],
        "global_mean": float(arr.mean()),
        "global_std": float(arr.std(ddof=1)) if arr.shape[0] > 1 else 0.0,
    }
    log.info("  global K-noise: %.4f ± %.4f (across seeds)",
             summary["global_mean"], summary["global_std"])
    log.info("  Per-layer std max: %.5f (most variable layer)",
             max(summary["per_layer_std"]))
    return summary


def main() -> int:
    args = parse_args()
    seeds = _discover_seeds(args.multiseed_root)
    log.info("Discovered seeds: %s", seeds)
    if not seeds:
        log.error("No seeds in %s", args.multiseed_root)
        return 2
    output_dir = args.output_dir or args.multiseed_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "quant": args.quant,
        "ref_quant": args.ref_quant,
        "seeds": seeds,
        "kl_trajectory": analyze_kl_trajectory_variance(
            args.multiseed_root, seeds, args.ref_quant, args.quant
        ),
        "outlier_stability": analyze_outlier_stability(
            args.multiseed_root, seeds, args.quant
        ),
        "layer_noise": analyze_layer_noise_variance(
            args.multiseed_root, seeds, args.quant
        ),
    }
    out_json = output_dir / f"variance_summary_{args.quant}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log.info("Saved %s", out_json)

    if not args.no_plots and "trajectory_mean" in summary["kl_trajectory"]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return 0
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        traj_mean = np.array(summary["kl_trajectory"]["trajectory_mean"])
        ci_low = np.array(summary["kl_trajectory"]["trajectory_ci_low"])
        ci_high = np.array(summary["kl_trajectory"]["trajectory_ci_high"])
        x = np.arange(len(traj_mean))
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(x, traj_mean, "k-", linewidth=2, label="mean across seeds")
        ax.fill_between(x, ci_low, ci_high, alpha=0.3, color="C0",
                        label=f"95% CI ({len(seeds)} seeds)")
        ax.set_xlabel("Decode step (from start of generation)")
        ax.set_ylabel(f"KL({args.ref_quant} || {args.quant})")
        ax.set_title(f"KL trajectory variance — T=0.6 multi-seed\n"
                     f"n={summary['kl_trajectory']['n_common_problems']} problems × "
                     f"{len(seeds)} seeds")
        if "tau_mean" in summary["kl_trajectory"]:
            ax.text(0.7, 0.1,
                    f"τ = {summary['kl_trajectory']['tau_mean']:.1f} ± "
                    f"{summary['kl_trajectory']['tau_std']:.1f}\n"
                    f"K∞ = {summary['kl_trajectory']['Kinf_mean']:.2f} ± "
                    f"{summary['kl_trajectory']['Kinf_std']:.2f}",
                    transform=ax.transAxes,
                    bbox={"facecolor": "white", "alpha": 0.8})
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = plot_dir / f"variance_trajectory_{args.quant}.png"
        plt.savefig(out_png, dpi=100)
        plt.close()
        log.info("Saved %s", out_png)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
