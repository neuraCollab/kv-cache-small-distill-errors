"""Phase 11: outlier-channel ablation analysis.

Гипотеза: ~1% каналов K (выбранных как outliers) объясняют 50%+ общего
квант-шума. Это даёт practical recommendation: per-channel scaling для
этих каналов восстанавливает большую часть качества.

Анализирует существующие captures без перезахвата:
  1. Для каждой задачи и слоя: вычисляет per-(head, channel) вклад в
     ||K_post - K_pre||² (см. analysis.outlier_channel_impact)
  2. Top-N каналов на слое → их fraction of total noise
  3. Aggregates: сколько каналов нужно чтобы покрыть 50%, 80%, 95% шума

Outputs:
  - outlier_channel_impact_<quant>.json — per-layer breakdown
  - plots/outlier_concentration_<quant>.png — Lorenz curve

Usage:
    python scripts/11_outlier_analysis.py --quant fp8_e4m3
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from kvtrace.capture.analysis import (
    load_captures_for_quant,
    outlier_channel_impact,
)

log = logging.getLogger("phase11")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 11: outlier-channel analysis")
    p.add_argument("--captures-dir", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b"))
    p.add_argument("--quant", default="fp8_e4m3", choices=["fp8_e4m3", "fp8_e5m2"])
    p.add_argument("--mode", default="tf", choices=["tf", "ar"])
    p.add_argument("--top-n", type=int, default=20,
                   help="Top-N каналов на слой")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.captures_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    captures = load_captures_for_quant(args.captures_dir, args.quant, args.mode)
    if not captures:
        log.error("No captures for %s %s", args.quant, args.mode)
        return 2
    log.info("Loaded %d captures", len(captures))

    # Per-layer Lorenz curve: sort channels by noise contribution, cumulative
    # fraction of total noise vs fraction of channels
    n_layers = len(captures[0].k_pre)
    num_kv_heads = captures[0].k_pre[0].shape[1]
    head_dim = captures[0].k_pre[0].shape[2]
    total_channels = num_kv_heads * head_dim
    log.info("Layers=%d, channels per layer=%d (%d kv_heads × %d head_dim)",
             n_layers, total_channels, num_kv_heads, head_dim)

    # Accumulate noise per (problem, layer, channel)
    all_per_channel = np.zeros((len(captures), n_layers, total_channels), dtype=np.float32)
    for p_idx, cap in enumerate(captures):
        for layer in range(n_layers):
            delta = (cap.k_post[layer].float() - cap.k_pre[layer].float())
            # [W, num_kv_heads, head_dim] → sum over W: [num_kv_heads, head_dim]
            per_ch = (delta ** 2).sum(dim=0).flatten().numpy()
            all_per_channel[p_idx, layer] = per_ch

    # Per-layer mean across problems
    mean_per_channel = all_per_channel.mean(axis=0)  # [n_layers, total_channels]

    # Per-layer cumulative fractions (Lorenz)
    # For each layer: sort descending, cumulative sum, divide by total
    lorenz: dict[int, dict] = {}
    for layer in range(n_layers):
        contribs = mean_per_channel[layer]
        total = float(contribs.sum())
        if total < 1e-12:
            lorenz[layer] = {"total": 0.0, "fraction_at": {}}
            continue
        sorted_desc = np.sort(contribs)[::-1]
        cumsum = np.cumsum(sorted_desc) / total  # [total_channels]
        # Сколько каналов до 50/80/95%?
        n_for_50 = int((cumsum >= 0.50).argmax()) + 1
        n_for_80 = int((cumsum >= 0.80).argmax()) + 1
        n_for_95 = int((cumsum >= 0.95).argmax()) + 1
        lorenz[layer] = {
            "total": total,
            "n_channels_for_50pct": n_for_50,
            "n_channels_for_80pct": n_for_80,
            "n_channels_for_95pct": n_for_95,
            "frac_top1_channel": float(cumsum[0]),
            "frac_top5_channels": float(cumsum[4]) if len(cumsum) >= 5 else float(cumsum[-1]),
            "frac_top10_channels": float(cumsum[9]) if len(cumsum) >= 10 else float(cumsum[-1]),
        }

    # Per-problem detailed (top-N)
    per_problem: list[list[dict]] = []
    for cap in captures:
        per_problem.append(outlier_channel_impact(cap, top_n_channels=args.top_n))

    summary = {
        "quant": args.quant,
        "mode": args.mode,
        "n_problems": len(captures),
        "n_layers": n_layers,
        "total_channels_per_layer": total_channels,
        "lorenz_per_layer": lorenz,
        "global_stats": {
            "median_n_channels_for_50pct": float(np.median(
                [v["n_channels_for_50pct"] for v in lorenz.values()
                 if "n_channels_for_50pct" in v]
            )),
            "median_n_channels_for_80pct": float(np.median(
                [v["n_channels_for_80pct"] for v in lorenz.values()
                 if "n_channels_for_80pct" in v]
            )),
            "median_n_channels_for_95pct": float(np.median(
                [v["n_channels_for_95pct"] for v in lorenz.values()
                 if "n_channels_for_95pct" in v]
            )),
            "median_top1_frac": float(np.median(
                [v["frac_top1_channel"] for v in lorenz.values()
                 if "frac_top1_channel" in v]
            )),
            "median_top10_frac": float(np.median(
                [v["frac_top10_channels"] for v in lorenz.values()
                 if "frac_top10_channels" in v]
            )),
        },
    }
    out_path = output_dir / f"outlier_channel_impact_{args.quant}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    log.info("=== Concentration of K-quant noise (median across %d layers) ===", n_layers)
    log.info("  top-1 channel: %.1f%% of layer total",
             100 * summary["global_stats"]["median_top1_frac"])
    log.info("  top-10 channels: %.1f%% of layer total",
             100 * summary["global_stats"]["median_top10_frac"])
    log.info("  median #channels for 50%% of noise: %.0f / %d (%.1f%%)",
             summary["global_stats"]["median_n_channels_for_50pct"], total_channels,
             100 * summary["global_stats"]["median_n_channels_for_50pct"] / total_channels)
    log.info("  median #channels for 80%% of noise: %.0f / %d (%.1f%%)",
             summary["global_stats"]["median_n_channels_for_80pct"], total_channels,
             100 * summary["global_stats"]["median_n_channels_for_80pct"] / total_channels)
    log.info("  median #channels for 95%% of noise: %.0f / %d (%.1f%%)",
             summary["global_stats"]["median_n_channels_for_95pct"], total_channels,
             100 * summary["global_stats"]["median_n_channels_for_95pct"] / total_channels)

    log.info("Saved %s", out_path)

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return 0
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        # Lorenz curve: per layer, fraction of channels (x) vs fraction of noise (y)
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(1, total_channels + 1) / total_channels  # fraction of channels
        # Per-layer Lorenz curves (light)
        for layer in range(n_layers):
            contribs = mean_per_channel[layer]
            total = float(contribs.sum())
            if total < 1e-12:
                continue
            cumsum = np.cumsum(np.sort(contribs)[::-1]) / total
            ax.plot(x, cumsum, color="lightblue", alpha=0.3, linewidth=0.5)
        # Mean across layers
        global_mean = np.zeros(total_channels)
        for layer in range(n_layers):
            contribs = mean_per_channel[layer]
            total = float(contribs.sum())
            if total < 1e-12:
                continue
            global_mean += np.cumsum(np.sort(contribs)[::-1]) / total
        global_mean /= n_layers
        ax.plot(x, global_mean, "k-", linewidth=2, label="mean across layers")
        # Reference: uniform distribution
        ax.plot(x, x, "r--", alpha=0.5, label="uniform (no concentration)")
        # Reference lines for 50/80/95%
        for pct, label in [(0.5, "50%"), (0.8, "80%"), (0.95, "95%")]:
            ax.axhline(pct, color="gray", linestyle=":", alpha=0.4)
            n = int(np.argmax(global_mean >= pct)) + 1
            ax.axvline(n / total_channels, color="gray", linestyle=":", alpha=0.4)
            ax.text(n / total_channels + 0.005, pct - 0.04,
                    f"{label}: {n}/{total_channels}", fontsize=8)
        ax.set_xlabel(f"Fraction of channels (sorted by noise contribution)\n"
                      f"({total_channels} channels per layer = {num_kv_heads}h × {head_dim}d)")
        ax.set_ylabel("Cumulative fraction of K-quant noise")
        ax.set_title(f"Outlier-channel concentration — {args.quant}\n"
                     f"({len(captures)} problems × {n_layers} layers)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        out_png = plot_dir / f"outlier_concentration_{args.quant}.png"
        plt.savefig(out_png, dpi=100)
        plt.close()
        log.info("Saved %s", out_png)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
