"""Phase 14: attention map shift due to K-quantization (central mechanism).

Использует attention_shift_kl(cap) на новых captures с q_post_rope.
Для каждого слоя ℓ и позиции query t:
  KL[ℓ, t] = KL(softmax(Q_t · K_pre^T/√d) || softmax(Q_t · K_post^T/√d))

Это ПРЯМОЕ измерение attention shift, без приближений Тейлора.
Math story Chapter 2 из docs/divergence_mechanism.md проверяется
конкретными числами.

Усреднение по 80 problems → per-(layer, position) heatmap. Также:
  - Mean attention KL per layer (vs K-noise magnitude → correlation?)
  - Per-position trajectory near FDP

Outputs:
  - attention_shift_kl_<quant>.npz: [n_problems, n_layers, W] KL
  - attention_shift_summary_<quant>.json: per-layer aggregates
  - plots/attention_shift_heatmap_<quant>.png

Usage:
    python scripts/14_attention_shift.py --quant fp8_e4m3 --mode tf
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from kvtrace.capture.analysis import (
    attention_shift_kl,
    load_captures_for_quant,
)

log = logging.getLogger("phase14")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 14: attention shift analysis")
    p.add_argument("--captures-dir", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b"))
    p.add_argument("--quant", default="fp8_e4m3", choices=["fp8_e4m3", "fp8_e5m2"])
    p.add_argument("--mode", default="tf", choices=["tf", "ar", "tf_prompt"])
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.captures_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    captures = load_captures_for_quant(args.captures_dir, args.quant, args.mode)
    if not captures:
        log.error("No captures for %s %s in %s", args.quant, args.mode, args.captures_dir)
        return 2
    # Filter captures with q_post_rope (skip older files без него)
    captures = [c for c in captures if c.q_post_rope is not None]
    if not captures:
        log.error("No captures have q_post_rope (re-capture with new code)")
        return 2
    log.info("Loaded %d captures with q_post_rope", len(captures))

    n_layers = len(captures[0].q_post_rope)
    log.info("n_layers=%d", n_layers)

    # Compute per-(problem, layer, position) KL
    all_kl: list[np.ndarray] = []
    fdp_in_window: list[int] = []
    Ws: list[int] = []
    for i, cap in enumerate(captures):
        kl = attention_shift_kl(cap).numpy()  # [n_layers, W]
        all_kl.append(kl)
        fdp = cap.meta["fdp_token_idx"] - cap.meta["window_start"]
        fdp_in_window.append(fdp)
        Ws.append(cap.meta["W"])
        if i % 10 == 0:
            log.info("  problem %d: W=%d, FDP@%d, mean attention KL=%.5f, max=%.5f",
                     cap.meta["problem_id"], kl.shape[1], fdp, kl.mean(), kl.max())

    # Aligned-to-FDP stack: [n_problems, n_layers, 301]
    aligned = np.full((len(all_kl), n_layers, 301), np.nan, dtype=np.float32)
    for i, (kl, fdp) in enumerate(zip(all_kl, fdp_in_window)):
        src_start = max(0, fdp - 150)
        src_end = min(kl.shape[1], fdp + 151)
        dst_start = 150 - (fdp - src_start)
        dst_end = dst_start + (src_end - src_start)
        aligned[i, :, dst_start:dst_end] = kl[:, src_start:src_end]

    out_path = output_dir / f"attention_shift_kl_{args.quant}.npz"
    np.savez(out_path, kl=aligned, fdp_in_window=fdp_in_window, Ws=Ws)
    log.info("Saved %s", out_path)

    # Per-layer aggregation
    per_layer_mean = np.nanmean(aligned, axis=(0, 2))  # [n_layers]
    per_layer_at_fdp = np.nanmean(aligned[:, :, 150], axis=0)  # mean at FDP
    summary = {
        "quant": args.quant,
        "mode": args.mode,
        "n_problems": len(captures),
        "n_layers": int(n_layers),
        "per_layer": [
            {
                "layer": int(L),
                "mean_kl": float(per_layer_mean[L]),
                "kl_at_fdp_mean": float(per_layer_at_fdp[L]),
            } for L in range(n_layers)
        ],
        "worst_layer_mean": int(np.argmax(per_layer_mean)),
        "worst_layer_at_fdp": int(np.argmax(per_layer_at_fdp)),
        "global_mean_kl": float(np.nanmean(aligned)),
        "kl_at_fdp_overall": float(np.nanmean(aligned[:, :, 150])),
        "kl_at_fdp_minus_1_overall": float(np.nanmean(aligned[:, :, 149])),
        "kl_at_fdp_plus_1_overall": float(np.nanmean(aligned[:, :, 151])),
    }
    out_json = output_dir / f"attention_shift_summary_{args.quant}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    log.info("=== Attention shift KL summary ===")
    log.info("  global mean attention KL: %.5f", summary["global_mean_kl"])
    log.info("  attention KL @FDP: %.5f", summary["kl_at_fdp_overall"])
    log.info("  attention KL @FDP-1: %.5f", summary["kl_at_fdp_minus_1_overall"])
    log.info("  worst layer (mean): %d (KL=%.5f)",
             summary["worst_layer_mean"], per_layer_mean[summary["worst_layer_mean"]])
    log.info("  worst layer (@FDP): %d (KL=%.5f)",
             summary["worst_layer_at_fdp"], per_layer_at_fdp[summary["worst_layer_at_fdp"]])

    ranked = sorted(range(n_layers), key=lambda L: -per_layer_mean[L])
    log.info("  TOP-5 layers by mean attention KL:")
    for L in ranked[:5]:
        log.info("    L%2d: %.5f", L, per_layer_mean[L])

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return 0
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        # Heatmap layer × position relative to FDP
        mean_per_layer_pos = np.nanmean(aligned, axis=0)  # [n_layers, 301]
        fig, ax = plt.subplots(figsize=(13, 7))
        im = ax.imshow(mean_per_layer_pos, aspect="auto", cmap="hot",
                       extent=[-150, 150, n_layers - 0.5, -0.5])
        ax.axvline(0, color="cyan", linestyle="--", alpha=0.7, label="FDP")
        ax.set_xlabel("Position relative to FDP")
        ax.set_ylabel("Layer")
        ax.set_title(f"Attention map shift KL — {args.quant} ({args.mode})\n"
                     f"per (layer, query position), mean across {len(captures)} problems")
        plt.colorbar(im, ax=ax, label="KL(attn_bf16 || attn_quant)")
        ax.legend()
        plt.tight_layout()
        out_png = plot_dir / f"attention_shift_heatmap_{args.quant}.png"
        plt.savefig(out_png, dpi=100)
        plt.close()
        log.info("Saved %s", out_png)

        # Per-layer bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(n_layers)
        ax1.bar(x, per_layer_mean, color="C1", alpha=0.7)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Mean attention KL (across window)")
        ax1.set_title(f"Per-layer attention shift — {args.quant}")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(x, per_layer_at_fdp, color="C3", alpha=0.7)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Attention KL @FDP position")
        ax2.set_title(f"Per-layer attention KL @ FDP — {args.quant}")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        out_png = plot_dir / f"attention_shift_per_layer_{args.quant}.png"
        plt.savefig(out_png, dpi=100)
        plt.close()
        log.info("Saved %s", out_png)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
