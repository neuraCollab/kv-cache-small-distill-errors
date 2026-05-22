"""Phase 13: counterfactual — что если ЗАЩИТИТЬ top-K самых impactful слоёв?

Гипотеза: per-layer ablation (Phase 10, 80 проблем) показал что слои
3, 15, 0, 5, 9 вносят 30-40% больше KL чем остальные. Если их НЕ
квантовать (оставить bf16), а остальные 23 квантовать, recover ли мы
большую часть accuracy?

Эксперимент:
  Для K ∈ {0 (no skip = full quant baseline), 1, 3, 5, 10}:
    Для каждой из N задач:
      Run TF forward с quant_only_layers = ALL \\ TOP_K_LAYERS
      Capture logits
      Compute KL(bf16 || protected_run) per position
    Aggregate: mean KL across problems, top1 match rate

Skip configurations (top-K в порядке убывания mean KL impact из Phase 10):
  K=0:  quant ALL 28 layers (baseline = current production)
  K=1:  protect L3
  K=3:  protect L3, L15, L0
  K=5:  protect L3, L15, L0, L5, L9
  K=10: protect L3, L15, L0, L5, L9, L7, L1, L8, L16, L11 (next 5 from rank list)

Outputs:
  outputs/kv_capture/qwen3-1.7b/analysis/counterfactual_skipK.json
  outputs/kv_capture/qwen3-1.7b/analysis/plots/counterfactual_skipK.png

Stoимость: K_configs × N_problems × 1 forward each.
  Для N=20 проблем × 5 конфигов = 100 forwards ≈ 8-12 мин.

Usage:
    python scripts/13_counterfactual.py --n-problems 20 --quant fp8_e4m3
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from kvtrace.capture.attention_hooks import install_capture_hooks
from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.fp8_sim import QUANT_FNS
from kvtrace.capture.storage import load_capture

log = logging.getLogger("phase13")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Top-K rankings from Phase 10 layer ablation on 80 problems (fp8_e4m3).
# Order: descending mean KL impact.
TOP_LAYERS_BY_IMPACT = [3, 15, 0, 5, 9, 7, 1, 8, 16, 11]

SKIP_CONFIGS = {
    0: [],
    1: TOP_LAYERS_BY_IMPACT[:1],
    3: TOP_LAYERS_BY_IMPACT[:3],
    5: TOP_LAYERS_BY_IMPACT[:5],
    10: TOP_LAYERS_BY_IMPACT[:10],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 13: skip-top-K counterfactual")
    p.add_argument("--captures-dir", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b"))
    p.add_argument("--traces-dir", type=Path, default=Path("outputs/traces"))
    p.add_argument("--quant", default="fp8_e4m3", choices=["fp8_e4m3", "fp8_e5m2"])
    p.add_argument("--n-problems", type=int, default=20)
    p.add_argument("--problems", default=None,
                   help="'0,3,5' for explicit list; overrides --n-problems")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _load_bf16_trace_tokens(traces_dir: Path) -> dict[int, list[int]]:
    path = traces_dir / "qwen3-1.7b_bf16.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out[rec["idx"]] = rec["token_ids"]
    return out


def _kl_per_position(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    p_a = torch.softmax(log_a.float(), dim=-1)
    p_b = torch.softmax(log_b.float(), dim=-1)
    eps = 1e-12
    return (p_a * (torch.log(p_a + eps) - torch.log(p_b + eps))).sum(dim=-1)


def _argmax_match_rate(log_a: torch.Tensor, log_b: torch.Tensor) -> float:
    a = log_a.argmax(dim=-1)
    b = log_b.argmax(dim=-1)
    return float((a == b).float().mean())


def run_skip_forward(runner: CaptureRunner, input_token_ids: list[int],
                     window_we: int, quant_fn, skip_layers: list[int],
                     n_layers: int) -> torch.Tensor:
    """Run TF forward с quant_only_layers = ALL \\ skip_layers."""
    quant_layers = set(range(n_layers)) - set(skip_layers)
    handle = install_capture_hooks(
        runner._model,
        attention_modules=runner._attention_modules,
        quant_fn=quant_fn,
        quant_only_layers=quant_layers,
    )
    try:
        feed_ids = torch.tensor([input_token_ids[:window_we]], dtype=torch.long)
        with torch.no_grad():
            out = runner._model(input_ids=feed_ids, use_cache=True)
        return out.logits[0].detach().clone()
    finally:
        handle.remove()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.captures_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.problems:
        problem_ids = [int(p) for p in args.problems.split(",")]
    else:
        problem_ids = list(range(args.n_problems))

    quant_fn = QUANT_FNS[args.quant]

    log.info("Loading Qwen3-1.7B on CPU...")
    runner = CaptureRunner()
    runner.load_model("Qwen/Qwen3-1.7B")
    n_layers = runner._n_layers
    log.info("n_layers=%d, configs=%s", n_layers, list(SKIP_CONFIGS.keys()))

    bf16_tokens = _load_bf16_trace_tokens(args.traces_dir)

    # Per-(K, problem): {'kl_mean', 'kl_at_fdp', 'argmax_match_rate', 'argmax_at_fdp_match'}
    results: dict[int, list[dict]] = {K: [] for K in SKIP_CONFIGS}

    for pid in problem_ids:
        bf16_path = args.captures_dir / "bf16_tf" / f"{pid}.safetensors"
        if not bf16_path.exists():
            log.warning("Missing bf16 capture for problem %d", pid)
            continue
        bf16_cap = load_capture(bf16_path)
        ws, we = bf16_cap.meta["window_start"], bf16_cap.meta["window_end"]
        fdp = bf16_cap.meta["fdp_token_idx"]
        ref_logits = bf16_cap.logits.float()  # [W, vocab]
        fdp_in_window = fdp - ws

        token_ids = bf16_tokens[pid]
        log.info("  problem %d: FDP=%d, W=%d", pid, fdp, ref_logits.shape[0])

        for K, skip_list in SKIP_CONFIGS.items():
            logits = run_skip_forward(runner, token_ids, we, quant_fn,
                                       skip_layers=skip_list, n_layers=n_layers)
            sliced = logits[ws:we].cpu()
            kl = _kl_per_position(ref_logits, sliced)
            mean_kl = float(kl.mean())
            kl_at_fdp = float(kl[fdp_in_window]) if 0 <= fdp_in_window < len(kl) else float("nan")
            am_match = _argmax_match_rate(ref_logits, sliced)
            am_at_fdp = (
                int(ref_logits[fdp_in_window].argmax() == sliced[fdp_in_window].argmax())
                if 0 <= fdp_in_window < len(kl) else None
            )
            results[K].append({
                "problem_id": pid,
                "kl_mean": mean_kl,
                "kl_at_fdp": kl_at_fdp,
                "argmax_match_rate": am_match,
                "argmax_at_fdp_match": am_at_fdp,
            })
            log.info("    K=%2d (skip %s): mean_kl=%.5f, kl@FDP=%.5f, argmax_match=%.3f, argmax@FDP=%s",
                     K, skip_list, mean_kl, kl_at_fdp, am_match, am_at_fdp)

    runner.unload()

    # Aggregate
    summary = {
        "quant": args.quant,
        "n_problems": len(problem_ids),
        "skip_configs": {str(K): list(v) for K, v in SKIP_CONFIGS.items()},
        "per_K": {},
    }
    for K, per_problem in results.items():
        if not per_problem:
            continue
        kls = [r["kl_mean"] for r in per_problem]
        kls_fdp = [r["kl_at_fdp"] for r in per_problem if not np.isnan(r["kl_at_fdp"])]
        matches = [r["argmax_match_rate"] for r in per_problem]
        am_fdp = [r["argmax_at_fdp_match"] for r in per_problem if r["argmax_at_fdp_match"] is not None]
        summary["per_K"][str(K)] = {
            "K": K,
            "skip_layers": list(SKIP_CONFIGS[K]),
            "n": len(per_problem),
            "mean_kl_mean": float(np.mean(kls)),
            "mean_kl_at_fdp": float(np.mean(kls_fdp)) if kls_fdp else float("nan"),
            "mean_argmax_match": float(np.mean(matches)),
            "argmax_at_fdp_match_rate": float(np.mean(am_fdp)) if am_fdp else float("nan"),
            "per_problem": per_problem,
        }

    out_path = output_dir / f"counterfactual_skipK_{args.quant}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("Saved %s", out_path)

    # Report
    log.info("=== Counterfactual: skip-top-K layers ===")
    log.info(f"{'K':>3} {'skip':<35} {'mean_KL':>10} {'KL@FDP':>10} {'argmax':>8} {'@FDP':>6}")
    baseline_kl = summary["per_K"]["0"]["mean_kl_mean"] if "0" in summary["per_K"] else None
    for K in sorted(SKIP_CONFIGS.keys()):
        s = summary["per_K"].get(str(K))
        if not s:
            continue
        skip_str = str(s["skip_layers"])[:32]
        recovery_pct = ""
        if baseline_kl and K > 0:
            recovery = (baseline_kl - s["mean_kl_mean"]) / baseline_kl * 100
            recovery_pct = f" ({recovery:+.0f}%)"
        log.info(f"  {K:>2} {skip_str:<35} {s['mean_kl_mean']:>10.5f}{recovery_pct} "
                 f"{s['mean_kl_at_fdp']:>10.5f} {s['mean_argmax_match']:>8.3f} "
                 f"{s['argmax_at_fdp_match_rate']:>6.3f}")

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return 0
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        Ks = sorted(SKIP_CONFIGS.keys())
        kl_vals = [summary["per_K"][str(K)]["mean_kl_mean"] for K in Ks]
        am_vals = [summary["per_K"][str(K)]["mean_argmax_match"] for K in Ks]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(Ks, kl_vals, "o-", color="C3", linewidth=2, markersize=10)
        ax1.set_xlabel("K = number of top-impact layers protected (kept bf16)")
        ax1.set_ylabel("Mean KL(bf16 || skip-top-K)")
        ax1.set_title(f"KL recovery as we protect top-K layers — {args.quant}")
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(Ks)
        # Annotate recovery %
        if baseline_kl:
            for K, v in zip(Ks, kl_vals):
                if K == 0: continue
                rec = (baseline_kl - v) / baseline_kl * 100
                ax1.annotate(f"{rec:+.0f}%", (K, v), textcoords="offset points",
                             xytext=(8, 8), fontsize=9, color="darkgreen")

        ax2.plot(Ks, am_vals, "s-", color="C2", linewidth=2, markersize=10)
        ax2.set_xlabel("K = number of top-impact layers protected")
        ax2.set_ylabel("Argmax match rate vs bf16 (over window)")
        ax2.set_title(f"Token-level agreement vs bf16 — {args.quant}")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(Ks)
        plt.tight_layout()
        out_png = plot_dir / f"counterfactual_skipK_{args.quant}.png"
        plt.savefig(out_png, dpi=100)
        plt.close()
        log.info("Saved %s", out_png)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
