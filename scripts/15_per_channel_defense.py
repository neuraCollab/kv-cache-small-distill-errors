"""Phase 15: per-channel defense — keep top-N outlier channels in bf16.

Эксперимент: для каждой задачи и каждого слоя выбираем top-N outlier-каналов
по max|K_pre[:, head, channel]|. Симулируем "defense": эти каналы НЕ квантуются
(остаются bf16), остальные 1024-N каналов квантуются FP8 e4m3.

Измеряем:
  1. ||K_post_defended - K_pre|| / ||K_pre|| — quant error reduction
  2. Attention shift KL (через attention_shift_kl с заменённым K_post)
  3. Logit-level effect через reconstruct (ОПЦИОНАЛЬНО: дорого, skip)

Configs: N ∈ {0 (= baseline FP8), 1, 5, 10, 25, 50, 100}.

Outputs:
  outputs/.../per_channel_defense_<quant>.json
  outputs/.../plots/per_channel_defense_<quant>.png

Stoимость: чисто offline на existing captures, no model forward. <2 min.

Usage:
    python scripts/15_per_channel_defense.py --quant fp8_e4m3
"""
from __future__ import annotations

import argparse
import json
import logging
from copy import copy
from pathlib import Path

import numpy as np
import torch

from kvtrace.capture.analysis import (
    attention_shift_kl,
    load_captures_for_quant,
)
from kvtrace.capture.fp8_sim import (
    fp8_skip_outliers,
    identify_top_outlier_channels,
    QUANT_FNS,
)

log = logging.getLogger("phase15")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFENSE_LEVELS = [0, 1, 5, 10, 25, 50, 100]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 15: per-channel defense")
    p.add_argument("--captures-dir", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b"))
    p.add_argument("--quant", default="fp8_e4m3",
                   choices=["fp8_e4m3", "fp8_e5m2", "hqq_int4", "hqq_int2"])
    p.add_argument("--mode", default="tf", choices=["tf", "ar", "tf_prompt"])
    p.add_argument("--n-problems", type=int, default=None,
                   help="Limit к первым N задачам (default: все)")
    p.add_argument("--skip-attention", action="store_true",
                   help="Skip attention shift (only K error metric)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _frobenius_rel_error(K_pre: torch.Tensor, K_post: torch.Tensor) -> float:
    diff = (K_post.float() - K_pre.float()).norm()
    base = K_pre.float().norm() + 1e-12
    return float(diff / base)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.captures_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    captures = load_captures_for_quant(args.captures_dir, args.quant, args.mode)
    if args.n_problems:
        captures = captures[:args.n_problems]
    if not captures:
        log.error("No captures for %s %s", args.quant, args.mode)
        return 2
    if any(c.q_post_rope is None for c in captures) and not args.skip_attention:
        log.warning("Some captures lack q_post_rope — attention shift will be skipped")
        args.skip_attention = True
    log.info("Loaded %d captures", len(captures))

    base_fn = QUANT_FNS[args.quant]
    n_layers = len(captures[0].k_pre)

    # Per-problem results: { N: [ {layer, k_err_baseline, k_err_defended, ...} ] }
    # Average across problems → per-N metrics.
    per_N: dict[int, dict] = {}

    for N in DEFENSE_LEVELS:
        log.info("=== Defense level: top-%d outlier channels per layer ===", N)
        k_errors_baseline: list[float] = []
        k_errors_defended: list[float] = []
        att_shifts_baseline: list[float] = []
        att_shifts_defended: list[float] = []

        for cap in captures:
            for layer in range(n_layers):
                K_pre = cap.k_pre[layer].float()
                K_post_baseline = cap.k_post[layer].float()
                # Identify top-N outliers
                outliers = identify_top_outlier_channels(K_pre, N)
                # Defended K_post: same FP8 but skip those channels
                K_post_defended = fp8_skip_outliers(K_pre, outliers, base_fn=base_fn).float()

                k_errors_baseline.append(_frobenius_rel_error(K_pre, K_post_baseline))
                k_errors_defended.append(_frobenius_rel_error(K_pre, K_post_defended))

            # Attention shift (всем слоям сразу через подменённый cap)
            if not args.skip_attention:
                # Baseline attention shift: cap as is
                baseline_kl = attention_shift_kl(cap).mean().item()
                att_shifts_baseline.append(baseline_kl)

                # Defended: build modified cap with new k_post per layer
                new_k_post = []
                new_v_post = list(cap.v_post)  # V unchanged
                for layer in range(n_layers):
                    K_pre = cap.k_pre[layer].float()
                    outliers = identify_top_outlier_channels(K_pre, N)
                    K_post_d = fp8_skip_outliers(K_pre, outliers, base_fn=base_fn)
                    new_k_post.append(K_post_d.to(cap.k_post[layer].dtype))

                cap_d = copy(cap)
                cap_d.k_post = new_k_post
                cap_d.v_post = new_v_post
                defended_kl = attention_shift_kl(cap_d).mean().item()
                att_shifts_defended.append(defended_kl)

        mean_k_err_base = float(np.mean(k_errors_baseline))
        mean_k_err_def = float(np.mean(k_errors_defended))
        k_err_reduction = (mean_k_err_base - mean_k_err_def) / mean_k_err_base * 100

        result = {
            "N": N,
            "n_problems": len(captures),
            "n_layers": n_layers,
            "mean_k_error_baseline": mean_k_err_base,
            "mean_k_error_defended": mean_k_err_def,
            "k_error_reduction_pct": k_err_reduction,
            "channels_protected_pct": 100 * N / (n_layers * 1024) * n_layers,
        }
        if not args.skip_attention and att_shifts_defended:
            mean_att_base = float(np.mean(att_shifts_baseline))
            mean_att_def = float(np.mean(att_shifts_defended))
            att_reduction = (mean_att_base - mean_att_def) / mean_att_base * 100
            result["mean_attention_kl_baseline"] = mean_att_base
            result["mean_attention_kl_defended"] = mean_att_def
            result["attention_kl_reduction_pct"] = att_reduction

        per_N[N] = result
        log.info("  N=%d: K-err %.4f → %.4f (%.1f%% reduction)",
                 N, mean_k_err_base, mean_k_err_def, k_err_reduction)
        if not args.skip_attention and att_shifts_defended:
            log.info("  N=%d: AttKL %.5f → %.5f (%.1f%% reduction)",
                     N, mean_att_base, mean_att_def, att_reduction)

    summary = {
        "quant": args.quant,
        "mode": args.mode,
        "defense_levels": DEFENSE_LEVELS,
        "per_N": per_N,
    }
    out_json = output_dir / f"per_channel_defense_{args.quant}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    log.info("=== Summary ===")
    log.info(f"{'N':>5} {'%chan':>7} {'K-err':>10} {'red%':>7} {'AttKL':>10} {'red%':>7}")
    for N in DEFENSE_LEVELS:
        r = per_N[N]
        pct = 100 * N / 1024
        att_kl = r.get("mean_attention_kl_defended", float("nan"))
        att_red = r.get("attention_kl_reduction_pct", float("nan"))
        log.info(f"  {N:>3} {pct:>6.2f}% {r['mean_k_error_defended']:>10.5f} "
                 f"{r['k_error_reduction_pct']:>6.1f}% {att_kl:>10.5f} {att_red:>6.1f}%")

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return 0
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        Ns = DEFENSE_LEVELS
        k_red = [per_N[N]["k_error_reduction_pct"] for N in Ns]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(Ns, k_red, "o-", color="C3", linewidth=2, markersize=10,
                label="K-quant error reduction")
        if not args.skip_attention:
            att_red = [per_N[N].get("attention_kl_reduction_pct", 0) for N in Ns]
            ax.plot(Ns, att_red, "s-", color="C0", linewidth=2, markersize=10,
                    label="Attention KL reduction")
        ax.set_xlabel("N = top outlier channels per layer kept in bf16")
        ax.set_ylabel("Reduction (%) vs baseline FP8")
        ax.set_title(f"Per-channel defense effectiveness — {args.quant} ({args.mode})\n"
                     f"n={len(captures)} problems × {n_layers} layers")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("symlog", linthresh=1)
        # Annotate %
        for N, kr in zip(Ns, k_red):
            ax.annotate(f"{kr:.0f}%", (N, kr), textcoords="offset points",
                        xytext=(8, 8), fontsize=8, color="darkred")
        plt.tight_layout()
        out_png = plot_dir / f"per_channel_defense_{args.quant}.png"
        plt.savefig(out_png, dpi=100)
        plt.close()
        log.info("Saved %s", out_png)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
