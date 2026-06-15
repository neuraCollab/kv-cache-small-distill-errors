"""Phase 20: Practical re-validation of paper's headline claims on live model.

Re-computes на свежих проблемах через live model.forward:
  - K-noise concentration (top-1, top-10 fractions vs 51.7%/10.3% paper)
  - Layer-wise relative Frobenius K-error (vs 2.65% paper для fp8_e4m3)
  - Per-channel defense (N=10) lab-metric K-error reduction (vs -34% paper)
  - HQQ INT4 — те же measurements (vs 10.3% baseline, 13.1% top-10, +6% defense)

Сравниваем с lab JSONs (которые тоже were measured on real model, но в TF mode
80 problems). Эта валидация — sanity check что pipeline даёт consistent числа
on independent fresh prompts.

Usage:
    python scripts/20_paper_validation.py --n-problems 5
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
from kvtrace.capture.fp8_sim import (
    QUANT_FNS,
    fp8_e4m3,
    fp8_e5m2,
    hqq_int4,
    hqq_int2,
    fp8_skip_outliers,
    identify_top_outlier_channels,
)
from kvtrace.dataset_loader import load_math_dataset

log = logging.getLogger("phase20")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_USER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 20: paper claims live validation")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--n-problems", type=int, default=5)
    p.add_argument("--output", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b/analysis/paper_validation_live.json"))
    return p.parse_args()


def _build_prompt_tokens(tokenizer, problem_text: str) -> list[int]:
    messages = [{"role": "user",
                 "content": f"{problem_text}\n\n{DEFAULT_USER_INSTRUCTION}"}]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=True, return_tensors="pt",
    )[0].tolist()
    return ids


def _collect_K_per_layer(runner: CaptureRunner, prompt_tokens: list[int]) -> list[torch.Tensor]:
    """bf16 prefill, return list of K_pre[layer] of shape [B=1, num_kv_heads, seq, head_dim]."""
    handle = install_capture_hooks(
        runner._model,
        attention_modules=runner._attention_modules,
        quant_fn=QUANT_FNS["bf16"],
    )
    try:
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        with torch.no_grad():
            runner._model(input_ids)
        n_layers = len(runner._attention_modules)
        return [handle.k_pre[i].clone() for i in range(n_layers)]
    finally:
        handle.remove()


def _layer_metrics(K_pre: torch.Tensor, quant_fn) -> dict:
    """For one layer's K [1, h, seq, d], compute concentration + Frobenius error
    + defense-N=10 effect."""
    K_post = quant_fn(K_pre)
    # Layer Frobenius error (overall, not per-head, to match paper §3.1 mean)
    diff = K_pre - K_post
    layer_rel_err = float(torch.norm(diff) / torch.norm(K_pre))
    # Per-channel noise concentration
    # [1, h, seq, d] → per-channel squared L2 over (B, seq) → [h, d]
    ch_noise_sq = (diff ** 2).sum(dim=(0, 2))  # [h, d]
    flat = ch_noise_sq.flatten()
    total = float(flat.sum())
    sorted_desc, _ = flat.sort(descending=True)
    top1_frac = float(sorted_desc[0]) / total if total > 0 else 0.0
    top10_frac = float(sorted_desc[:10].sum()) / total if total > 0 else 0.0
    # Per-channel defense (N=10)
    outliers = identify_top_outlier_channels(K_pre, top_n=10)
    K_defended = fp8_skip_outliers(K_pre, outliers, base_fn=quant_fn)
    defended_rel_err = float(torch.norm(K_pre - K_defended) / torch.norm(K_pre))
    return {
        "layer_rel_err": layer_rel_err,
        "top1_ch_noise_frac": top1_frac,
        "top10_ch_noise_frac": top10_frac,
        "defended_rel_err": defended_rel_err,
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset (math-500), fresh problems 70-%d", 70 + args.n_problems - 1)
    problems = load_math_dataset("math-500", num_samples=70 + args.n_problems, shuffle=False)
    problems = problems[70:70 + args.n_problems]

    log.info("Loading %s...", args.model)
    runner = CaptureRunner()
    runner.load_model(args.model)
    n_layers = len(runner._attention_modules)
    log.info("n_layers=%d", n_layers)

    quants = {
        "fp8_e4m3": fp8_e4m3,
        "fp8_e5m2": fp8_e5m2,
        "hqq_int4": hqq_int4,
    }
    # Аккумуляторы: для каждого quant — list of per-layer metrics
    per_problem_per_quant: dict[str, list[list[dict]]] = {q: [] for q in quants}

    for p_idx, problem in enumerate(problems):
        prompt_tokens = _build_prompt_tokens(runner._tokenizer, problem.problem)
        log.info("problem %d (orig idx %d): prompt_len=%d",
                 p_idx, problem.idx, len(prompt_tokens))

        K_per_layer = _collect_K_per_layer(runner, prompt_tokens)
        log.info("  collected K_pre for %d layers", len(K_per_layer))

        for q_name, q_fn in quants.items():
            log.info("  computing metrics for %s...", q_name)
            layer_metrics = [_layer_metrics(K, q_fn) for K in K_per_layer]
            per_problem_per_quant[q_name].append(layer_metrics)

    runner.unload()

    # Aggregate: mean across (problems, layers) for each metric
    summary: dict = {
        "n_problems": args.n_problems,
        "n_layers": n_layers,
        "model": args.model,
        "fresh_indices": [int(p.idx) for p in problems],
        "per_quant": {},
    }
    for q_name in quants:
        arr = per_problem_per_quant[q_name]  # [n_problems][n_layers]{...}
        metric_names = ["layer_rel_err", "top1_ch_noise_frac",
                        "top10_ch_noise_frac", "defended_rel_err"]
        agg = {}
        for m in metric_names:
            vals = [layer[m] for prob in arr for layer in prob]
            agg[m + "_mean"] = float(np.mean(vals))
            agg[m + "_median"] = float(np.median(vals))
            agg[m + "_max"] = float(np.max(vals))
        # Defense effect = (defended - baseline) / baseline
        baseline = agg["layer_rel_err_mean"]
        defended = agg["defended_rel_err_mean"]
        agg["defense_change_pct"] = 100 * (defended - baseline) / baseline
        summary["per_quant"][q_name] = agg
        log.info("=== %s (n=%d problems × %d layers) ===", q_name, args.n_problems, n_layers)
        log.info("  layer_rel_err: mean=%.4f, max=%.4f", agg["layer_rel_err_mean"], agg["layer_rel_err_max"])
        log.info("  top1_ch_frac:  mean=%.4f, median=%.4f", agg["top1_ch_noise_frac_mean"], agg["top1_ch_noise_frac_median"])
        log.info("  top10_ch_frac: mean=%.4f, median=%.4f", agg["top10_ch_noise_frac_mean"], agg["top10_ch_noise_frac_median"])
        log.info("  defense(N=10): baseline=%.4f, defended=%.4f, change=%+.1f%%",
                 baseline, defended, agg["defense_change_pct"])

    args.output.write_text(json.dumps(summary, indent=2))
    log.info("Saved %s", args.output)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
