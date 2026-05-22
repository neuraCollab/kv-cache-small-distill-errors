"""Phase 8: визуализация механизма расхождения.

Производит три анализа, прямо отображающих математическую цепочку из
docs/divergence_mechanism.md:

  1. **Logit KL trajectory** через окно [FDP-150, FDP+100], усреднённая
     по 80 задачам. Ожидание: монотонный рост, пик у FDP.
  2. **bf16 margin trajectory** через окно. Ожидание: малый margin в
     позиции FDP — там модель не уверена → шум перекидывает argmax.
  3. **Per-layer K-noise**: ||ΔK||/||K|| по слоям и позициям. Ожидание:
     приблизительно константа по слоям, могут быть outlier-слои.

Outputs:
  outputs/kv_capture/qwen3-1.7b/analysis/
    logit_kl_trajectory_<quant>.npz       — [n_problems, W] KL per position
    margin_trajectory_bf16.npz            — [n_problems, W] margin per position
    layer_noise_<quant>.npz               — [n_problems, n_layers, W] K/V noise
    plots/
      kl_vs_position_<quant>.png          — mean KL ± std, FDP marked
      margin_vs_position.png              — bf16 margin distribution
      layer_noise_heatmap_<quant>.png     — layer × position K-noise heatmap

Usage:
    python scripts/08_divergence_mechanism.py
    python scripts/08_divergence_mechanism.py --no-plots --mode tf
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from kvtrace.capture.analysis import (
    bf16_margin_trajectory,
    captures_share_window,
    load_captures_for_quant,
    logit_kl_trajectory,
    per_position_kv_quant_noise,
)

log = logging.getLogger("phase8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 8: divergence mechanism analysis")
    p.add_argument(
        "--captures-dir", type=Path,
        default=Path("outputs/kv_capture/qwen3-1.7b"),
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--mode", default="tf", choices=["tf", "ar"])
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _align_to_fdp(trajectory: torch.Tensor, fdp_in_window: int, half_width: int = 150
                  ) -> torch.Tensor:
    """Re-center a per-position trajectory so position 0 = FDP.

    Returns Tensor[2*half_width+1] padded with NaN if window truncated.
    """
    aligned = torch.full((2 * half_width + 1,), float("nan"))
    src_start = max(0, fdp_in_window - half_width)
    src_end = min(len(trajectory), fdp_in_window + half_width + 1)
    dst_start = half_width - (fdp_in_window - src_start)
    dst_end = dst_start + (src_end - src_start)
    aligned[dst_start:dst_end] = trajectory[src_start:src_end]
    return aligned


def analyze_logit_kl_trajectory(
    captures_dir: Path, output_dir: Path, mode: str
) -> dict:
    """Per-problem logit KL through window, aligned to FDP."""
    log.info("=== Analysis: logit KL trajectory ===")
    bf16_caps = load_captures_for_quant(captures_dir, "bf16", mode)
    bf16_by_pid = {c.meta["problem_id"]: c for c in bf16_caps}
    summary: dict = {}

    for quant in ["fp8_e4m3", "fp8_e5m2"]:
        quant_caps = load_captures_for_quant(captures_dir, quant, mode)
        aligned_kls: list[np.ndarray] = []
        skipped = 0
        for cap_q in quant_caps:
            pid = cap_q.meta["problem_id"]
            cap_b = bf16_by_pid.get(pid)
            if cap_b is None or not captures_share_window(cap_b, cap_q):
                skipped += 1
                continue
            kl = logit_kl_trajectory(cap_b, cap_q)  # [W]
            fdp_in_window = cap_b.meta["fdp_token_idx"] - cap_b.meta["window_start"]
            aligned = _align_to_fdp(kl, fdp_in_window).numpy()
            aligned_kls.append(aligned)

        if not aligned_kls:
            log.warning("  %s: no captures with shared window", quant)
            continue
        stack = np.stack(aligned_kls, axis=0)  # [n, 301]

        out_path = output_dir / f"logit_kl_trajectory_{quant}.npz"
        np.savez(out_path, kl=stack)
        summary[quant] = {
            "n_problems": int(stack.shape[0]),
            "skipped": skipped,
            "kl_at_fdp_mean": float(np.nanmean(stack[:, 150])),
            "kl_at_fdp_minus_1_mean": float(np.nanmean(stack[:, 149])),
            "kl_at_fdp_plus_1_mean": float(np.nanmean(stack[:, 151])),
            "kl_pre_fdp_mean": float(np.nanmean(stack[:, 100:150])),
            "kl_post_fdp_mean": float(np.nanmean(stack[:, 151:201])),
        }
        log.info(
            "  %s: n=%d (skipped %d). KL@FDP-50:FDP=%.4f, KL@FDP=%.4f, KL@FDP+1:FDP+50=%.4f",
            quant, stack.shape[0], skipped,
            summary[quant]["kl_pre_fdp_mean"],
            summary[quant]["kl_at_fdp_mean"],
            summary[quant]["kl_post_fdp_mean"],
        )
    return summary


def analyze_margin_trajectory(
    captures_dir: Path, output_dir: Path, mode: str
) -> dict:
    """bf16 margin through window, aligned to FDP."""
    log.info("=== Analysis: bf16 margin trajectory ===")
    bf16_caps = load_captures_for_quant(captures_dir, "bf16", mode)
    aligned_margins: list[np.ndarray] = []
    for cap_b in bf16_caps:
        margin = bf16_margin_trajectory(cap_b)
        fdp_in_window = cap_b.meta["fdp_token_idx"] - cap_b.meta["window_start"]
        aligned = _align_to_fdp(margin, fdp_in_window).numpy()
        aligned_margins.append(aligned)

    if not aligned_margins:
        log.warning("  no bf16 captures")
        return {}

    stack = np.stack(aligned_margins, axis=0)
    out_path = output_dir / "margin_trajectory_bf16.npz"
    np.savez(out_path, margin=stack)
    summary = {
        "n_problems": int(stack.shape[0]),
        "margin_at_fdp_mean": float(np.nanmean(stack[:, 150])),
        "margin_at_fdp_median": float(np.nanmedian(stack[:, 150])),
        "margin_pre_fdp_mean": float(np.nanmean(stack[:, 100:150])),
        "margin_post_fdp_mean": float(np.nanmean(stack[:, 151:201])),
    }
    log.info(
        "  margin@FDP-50:FDP=%.3f, margin@FDP=%.3f (median %.3f), margin@FDP+1:FDP+50=%.3f",
        summary["margin_pre_fdp_mean"], summary["margin_at_fdp_mean"],
        summary["margin_at_fdp_median"], summary["margin_post_fdp_mean"],
    )
    return summary


def analyze_layer_noise(captures_dir: Path, output_dir: Path, mode: str) -> dict:
    """Per-layer K/V noise across positions, aggregated."""
    log.info("=== Analysis: per-layer K/V noise ===")
    summary: dict = {}
    for quant in ["fp8_e4m3", "fp8_e5m2"]:
        captures = load_captures_for_quant(captures_dir, quant, mode)
        if not captures:
            continue
        all_k: list[np.ndarray] = []
        all_v: list[np.ndarray] = []
        for cap in captures:
            noise = per_position_kv_quant_noise(cap)
            all_k.append(noise["k_noise"].numpy())
            all_v.append(noise["v_noise"].numpy())
        # Each has [n_layers, W_i] — W may differ if some truncated
        # Pad to max W for stacking
        max_w = max(arr.shape[1] for arr in all_k)
        def pad(arr: np.ndarray) -> np.ndarray:
            if arr.shape[1] == max_w:
                return arr
            out = np.full((arr.shape[0], max_w), np.nan)
            out[:, :arr.shape[1]] = arr
            return out
        k_stack = np.stack([pad(a) for a in all_k], axis=0)  # [n, L, W]
        v_stack = np.stack([pad(a) for a in all_v], axis=0)

        out_path = output_dir / f"layer_noise_{quant}.npz"
        np.savez(out_path, k_noise=k_stack, v_noise=v_stack)

        per_layer_k_mean = np.nanmean(k_stack, axis=(0, 2))  # [L]
        per_layer_v_mean = np.nanmean(v_stack, axis=(0, 2))
        worst_k = int(np.argmax(per_layer_k_mean))
        worst_v = int(np.argmax(per_layer_v_mean))
        summary[quant] = {
            "n_problems": int(k_stack.shape[0]),
            "n_layers": int(k_stack.shape[1]),
            "global_k_noise_mean": float(np.nanmean(k_stack)),
            "global_v_noise_mean": float(np.nanmean(v_stack)),
            "worst_layer_k": worst_k,
            "worst_layer_k_noise": float(per_layer_k_mean[worst_k]),
            "worst_layer_v": worst_v,
            "worst_layer_v_noise": float(per_layer_v_mean[worst_v]),
        }
        log.info(
            "  %s: K-noise mean=%.4f (worst layer %d: %.4f), V-noise mean=%.4f (worst layer %d: %.4f)",
            quant,
            summary[quant]["global_k_noise_mean"], worst_k,
            summary[quant]["worst_layer_k_noise"],
            summary[quant]["global_v_noise_mean"], worst_v,
            summary[quant]["worst_layer_v_noise"],
        )
    return summary


def plot_all(output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed, skipping plots")
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # 1. KL trajectory
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(-150, 151)
    for quant, color in [("fp8_e4m3", "C0"), ("fp8_e5m2", "C1")]:
        path = output_dir / f"logit_kl_trajectory_{quant}.npz"
        if not path.exists():
            continue
        kl = np.load(path)["kl"]
        mean = np.nanmean(kl, axis=0)
        sem = np.nanstd(kl, axis=0) / np.sqrt(np.sum(~np.isnan(kl), axis=0))
        ax.plot(x, mean, color=color, label=quant, linewidth=1.5)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="FDP")
    ax.set_xlabel("Position relative to FDP")
    ax.set_ylabel("KL(bf16 || fp8)")
    ax.set_title("Logit KL divergence through window (mean ± SEM over problems)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "kl_vs_position.png", dpi=100)
    plt.close()
    log.info("Saved kl_vs_position.png")

    # 2. Margin trajectory
    path = output_dir / "margin_trajectory_bf16.npz"
    if path.exists():
        margins = np.load(path)["margin"]
        fig, ax = plt.subplots(figsize=(12, 5))
        mean = np.nanmean(margins, axis=0)
        median = np.nanmedian(margins, axis=0)
        p25 = np.nanpercentile(margins, 25, axis=0)
        p75 = np.nanpercentile(margins, 75, axis=0)
        ax.plot(x, mean, color="C2", label="mean", linewidth=1.5)
        ax.plot(x, median, color="C3", label="median", linewidth=1.5)
        ax.fill_between(x, p25, p75, color="C2", alpha=0.2, label="IQR")
        ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="FDP")
        ax.set_xlabel("Position relative to FDP")
        ax.set_ylabel("bf16 margin (top1 − top2 logit)")
        ax.set_title("Model confidence (margin) through window")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "margin_vs_position.png", dpi=100)
        plt.close()
        log.info("Saved margin_vs_position.png")

    # 3. Layer noise heatmap
    for quant in ["fp8_e4m3", "fp8_e5m2"]:
        path = output_dir / f"layer_noise_{quant}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        k_mean = np.nanmean(data["k_noise"], axis=0)  # [L, W]
        v_mean = np.nanmean(data["v_noise"], axis=0)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        im1 = ax1.imshow(k_mean, aspect="auto", cmap="hot", origin="lower")
        ax1.set_title(f"K-noise per (layer, position) — {quant}")
        ax1.set_xlabel("Position in window")
        ax1.set_ylabel("Layer")
        plt.colorbar(im1, ax=ax1, label="||ΔK|| / ||K||")
        im2 = ax2.imshow(v_mean, aspect="auto", cmap="hot", origin="lower")
        ax2.set_title(f"V-noise per (layer, position) — {quant}")
        ax2.set_xlabel("Position in window")
        ax2.set_ylabel("Layer")
        plt.colorbar(im2, ax=ax2, label="||ΔV|| / ||V||")
        plt.tight_layout()
        plt.savefig(plot_dir / f"layer_noise_heatmap_{quant}.png", dpi=100)
        plt.close()
        log.info("Saved layer_noise_heatmap_%s.png", quant)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.captures_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "mode": args.mode,
        "logit_kl_trajectory": analyze_logit_kl_trajectory(
            args.captures_dir, output_dir, args.mode
        ),
        "margin_trajectory": analyze_margin_trajectory(
            args.captures_dir, output_dir, args.mode
        ),
        "layer_noise": analyze_layer_noise(args.captures_dir, output_dir, args.mode),
    }
    (output_dir / "divergence_mechanism_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    if not args.no_plots:
        plot_all(output_dir)

    log.info("Done. Output in %s", output_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
