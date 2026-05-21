"""Phase 7: aggregate analysis of KV-matrix captures.

Reads outputs/kv_capture/qwen3-1.7b/{bf16,fp8_e4m3,fp8_e5m2}_{tf,ar}/*.safetensors
и пишет в outputs/kv_capture/qwen3-1.7b/analysis/:
  - layer_head_error_<quant>.npz       — [n_problems, n_layers, num_kv_heads]
  - logits_kl_bf16_vs_<quant>.json     — per-problem JS/KL + top1 match
  - kv_stats_per_layer_<quant>.json    — aggregated K/V distribution stats
  - summary.json                       — top-level numeric findings

С --plots дополнительно рисует PNG-heatmaps в analysis/plots/.

Usage:
    python scripts/07_analyze_captures.py
    python scripts/07_analyze_captures.py --plots --mode tf
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from kvtrace.capture.analysis import (
    align_captures_by_absolute_position,
    captures_share_window,
    compute_kv_value_stats_per_layer,
    compute_logits_kl_at_fdp,
    compute_relative_quant_error,
    load_captures_for_quant,
)

log = logging.getLogger("analyze")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

QUANTS = ["bf16", "fp8_e4m3", "fp8_e5m2"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7 KV-capture analysis")
    p.add_argument(
        "--captures-dir", type=Path,
        default=Path("outputs/kv_capture/qwen3-1.7b"),
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="default = <captures-dir>/analysis",
    )
    p.add_argument("--mode", default="tf", choices=["tf", "ar"])
    p.add_argument(
        "--plots", action="store_true",
        help="Generate PNG heatmaps (requires matplotlib)",
    )
    return p.parse_args()


def analyze_layer_head_error(captures_dir: Path, output_dir: Path, mode: str) -> dict:
    """Analysis 1: per-(layer, head) quant error, aggregated."""
    log.info("=== Analysis 1: layer × head quant error ===")
    summary: dict = {}

    for quant in ["fp8_e4m3", "fp8_e5m2"]:
        captures = load_captures_for_quant(captures_dir, quant, mode)
        if not captures:
            log.warning("No captures for %s_%s", quant, mode)
            continue

        k_errors: list[torch.Tensor] = []
        v_errors: list[torch.Tensor] = []
        for cap in captures:
            err = compute_relative_quant_error(cap)
            k_errors.append(err.k_relative_error)
            v_errors.append(err.v_relative_error)

        k_stack = torch.stack(k_errors, dim=0).numpy()  # [n_problems, L, H]
        v_stack = torch.stack(v_errors, dim=0).numpy()
        problem_ids = np.array([c.meta["problem_id"] for c in captures])

        out_path = output_dir / f"layer_head_error_{quant}.npz"
        np.savez(out_path, k_error=k_stack, v_error=v_stack, problem_ids=problem_ids)
        log.info(
            "  %s: n=%d, k mean=%.4f (max=%.4f), v mean=%.4f (max=%.4f)",
            quant, len(captures),
            k_stack.mean(), k_stack.max(),
            v_stack.mean(), v_stack.max(),
        )
        summary[quant] = {
            "n_problems": int(len(captures)),
            "k_error_mean": float(k_stack.mean()),
            "k_error_max": float(k_stack.max()),
            "v_error_mean": float(v_stack.mean()),
            "v_error_max": float(v_stack.max()),
            "k_worst_layer": int(np.argmax(k_stack.mean(axis=(0, 2)))),
            "k_worst_head": int(np.argmax(k_stack.mean(axis=(0, 1)))),
        }

    return summary


def analyze_logits_kl(captures_dir: Path, output_dir: Path, mode: str) -> dict:
    """Analysis 2: logits divergence at FDP, bf16 vs each fp8 quant.

    Only works for quants sharing window with bf16. bf16 was captured with
    fp8_e4m3's FDP coordinate, so bf16 vs fp8_e4m3 is direct. bf16 vs
    fp8_e5m2 use different windows → align via overlapping abs positions.
    """
    log.info("=== Analysis 2: logits KL at FDP ===")
    bf16_caps = load_captures_for_quant(captures_dir, "bf16", mode)
    bf16_by_pid = {c.meta["problem_id"]: c for c in bf16_caps}
    summary: dict = {}

    for quant in ["fp8_e4m3", "fp8_e5m2"]:
        quant_caps = load_captures_for_quant(captures_dir, quant, mode)
        results = []

        for cap_q in quant_caps:
            pid = cap_q.meta["problem_id"]
            cap_b = bf16_by_pid.get(pid)
            if cap_b is None:
                continue
            if not captures_share_window(cap_b, cap_q):
                results.append({
                    "problem_id": pid,
                    "quant": quant,
                    "skipped": "different_windows",
                    "bf16_window": (cap_b.meta["window_start"], cap_b.meta["window_end"]),
                    "quant_window": (cap_q.meta["window_start"], cap_q.meta["window_end"]),
                })
                continue
            try:
                r = compute_logits_kl_at_fdp(cap_b, cap_q)
                r["problem_id"] = pid
                r["quant"] = quant
                results.append(r)
            except ValueError as e:
                results.append({"problem_id": pid, "quant": quant, "error": str(e)})

        out_path = output_dir / f"logits_kl_bf16_vs_{quant}.json"
        out_path.write_text(json.dumps(results, indent=2))

        valid = [r for r in results if "kl_a_to_b" in r]
        n_skipped = sum(1 for r in results if "skipped" in r)
        if valid:
            mean_kl = sum(r["kl_a_to_b"] for r in valid) / len(valid)
            mean_js = sum(r["js_divergence"] for r in valid) / len(valid)
            n_match = sum(r["top1_match"] for r in valid)
            mean_top5 = sum(r["top5_overlap"] for r in valid) / len(valid)
            log.info(
                "  %s: n=%d (skipped %d), mean KL=%.4f, mean JS=%.4f, "
                "top1 match=%d/%d (%.1f%%), mean top5 overlap=%.2f/5",
                quant, len(valid), n_skipped, mean_kl, mean_js,
                n_match, len(valid), 100 * n_match / len(valid), mean_top5,
            )
            summary[quant] = {
                "n_compared": len(valid),
                "n_skipped_different_windows": n_skipped,
                "mean_kl_bf16_to_quant": mean_kl,
                "mean_js": mean_js,
                "top1_match_count": int(n_match),
                "top1_match_pct": 100 * n_match / len(valid),
                "mean_top5_overlap": mean_top5,
            }

    return summary


def analyze_kv_stats(captures_dir: Path, output_dir: Path, mode: str) -> dict:
    """Analysis 3: per-layer K/V value distributions, outlier rates."""
    log.info("=== Analysis 3: K/V value distributions ===")
    summary: dict = {}

    for quant in QUANTS:
        captures = load_captures_for_quant(captures_dir, quant, mode)
        if not captures:
            continue

        n_layers = len(captures[0].k_pre)
        # Per-layer aggregator
        agg: dict[int, dict[str, list[float]]] = {
            layer: {
                "k_max_abs": [], "k_mean_abs": [], "k_std": [], "k_out_448": [], "k_out_57344": [],
                "v_max_abs": [], "v_mean_abs": [], "v_std": [], "v_out_448": [], "v_out_57344": [],
            } for layer in range(n_layers)
        }

        for cap in captures:
            per_layer = compute_kv_value_stats_per_layer(cap)
            for s in per_layer:
                layer = s["layer"]
                agg[layer]["k_max_abs"].append(s["k_max_abs"])
                agg[layer]["k_mean_abs"].append(s["k_mean_abs"])
                agg[layer]["k_std"].append(s["k_std"])
                agg[layer]["k_out_448"].append(s["k_outliers_pct_448"])
                agg[layer]["k_out_57344"].append(s["k_outliers_pct_57344"])
                agg[layer]["v_max_abs"].append(s["v_max_abs"])
                agg[layer]["v_mean_abs"].append(s["v_mean_abs"])
                agg[layer]["v_std"].append(s["v_std"])
                agg[layer]["v_out_448"].append(s["v_outliers_pct_448"])
                agg[layer]["v_out_57344"].append(s["v_outliers_pct_57344"])

        per_layer_summary = []
        for layer in range(n_layers):
            d = agg[layer]
            per_layer_summary.append({
                "layer": layer,
                "k_max_abs_mean": float(np.mean(d["k_max_abs"])),
                "k_max_abs_p99": float(np.percentile(d["k_max_abs"], 99)),
                "k_mean_abs_mean": float(np.mean(d["k_mean_abs"])),
                "k_std_mean": float(np.mean(d["k_std"])),
                "k_outliers_pct_448_mean": float(np.mean(d["k_out_448"])),
                "v_max_abs_mean": float(np.mean(d["v_max_abs"])),
                "v_max_abs_p99": float(np.percentile(d["v_max_abs"], 99)),
                "v_mean_abs_mean": float(np.mean(d["v_mean_abs"])),
                "v_std_mean": float(np.mean(d["v_std"])),
                "v_outliers_pct_448_mean": float(np.mean(d["v_out_448"])),
            })

        out_path = output_dir / f"kv_stats_per_layer_{quant}.json"
        out_path.write_text(json.dumps(per_layer_summary, indent=2))
        log.info(
            "  %s: saved per-layer stats for %d layers, %d problems",
            quant, n_layers, len(captures),
        )
        summary[quant] = {
            "n_layers": n_layers,
            "k_max_abs_global": float(max(s["k_max_abs_p99"] for s in per_layer_summary)),
            "v_max_abs_global": float(max(s["v_max_abs_p99"] for s in per_layer_summary)),
            "k_outliers_448_max_layer": int(
                np.argmax([s["k_outliers_pct_448_mean"] for s in per_layer_summary])
            ),
        }

    return summary


def plot_layer_head_heatmap(output_dir: Path) -> None:
    """Heatmaps from layer_head_error_*.npz."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plots")
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    for quant in ["fp8_e4m3", "fp8_e5m2"]:
        path = output_dir / f"layer_head_error_{quant}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        k_mean = data["k_error"].mean(axis=0)  # [L, H]
        v_mean = data["v_error"].mean(axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        im1 = ax1.imshow(k_mean, aspect="auto", cmap="viridis")
        ax1.set_title(f"K relative quant error (mean over 80) — {quant}")
        ax1.set_xlabel("KV head")
        ax1.set_ylabel("Layer")
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(v_mean, aspect="auto", cmap="viridis")
        ax2.set_title(f"V relative quant error (mean over 80) — {quant}")
        ax2.set_xlabel("KV head")
        ax2.set_ylabel("Layer")
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        out_png = plot_dir / f"layer_head_error_{quant}.png"
        plt.savefig(out_png, dpi=100)
        plt.close()
        log.info("Saved plot %s", out_png)


def plot_kv_max_abs_per_layer(output_dir: Path) -> None:
    """Line plot: K_max_abs per layer across quants."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    fig, (ax_k, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
    for quant in QUANTS:
        path = output_dir / f"kv_stats_per_layer_{quant}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        layers = [d["layer"] for d in data]
        k_max = [d["k_max_abs_p99"] for d in data]
        v_max = [d["v_max_abs_p99"] for d in data]
        ax_k.plot(layers, k_max, marker="o", label=quant)
        ax_v.plot(layers, v_max, marker="o", label=quant)

    ax_k.axhline(448, color="red", linestyle="--", alpha=0.5, label="e4m3 max (±448)")
    ax_k.set_title("K_max_abs p99 per layer")
    ax_k.set_xlabel("Layer")
    ax_k.set_ylabel("|K|")
    ax_k.set_yscale("log")
    ax_k.legend()
    ax_k.grid(True, alpha=0.3)

    ax_v.axhline(448, color="red", linestyle="--", alpha=0.5, label="e4m3 max (±448)")
    ax_v.set_title("V_max_abs p99 per layer")
    ax_v.set_xlabel("Layer")
    ax_v.set_ylabel("|V|")
    ax_v.set_yscale("log")
    ax_v.legend()
    ax_v.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = plot_dir / "kv_max_abs_per_layer.png"
    plt.savefig(out_png, dpi=100)
    plt.close()
    log.info("Saved plot %s", out_png)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.captures_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "captures_dir": str(args.captures_dir),
        "mode": args.mode,
        "layer_head_error": analyze_layer_head_error(args.captures_dir, output_dir, args.mode),
        "logits_kl": analyze_logits_kl(args.captures_dir, output_dir, args.mode),
        "kv_stats": analyze_kv_stats(args.captures_dir, output_dir, args.mode),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Saved summary %s", output_dir / "summary.json")

    if args.plots:
        plot_layer_head_heatmap(output_dir)
        plot_kv_max_abs_per_layer(output_dir)

    log.info("Done. Output in %s", output_dir)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
