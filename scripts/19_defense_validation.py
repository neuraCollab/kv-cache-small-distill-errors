"""Phase 19: End-to-end validation of per-channel defense in REAL AR generation.

Сравниваем три config'а на N задачах через actual model.generate():
  1. bf16 (ground truth — что модель РЕАЛЬНО говорит без quant)
  2. fp8_e4m3 (baseline — что quant ломает)
  3. fp8_e4m3_top10 (defense — что recovers per paper's recipe)

Metrics:
  - first_divergence_step: позиция первого расхождения с bf16 (НЕ FDP_token_idx,
    а relative step number от старта генерации). Defense должен ОТЛОЖИТ её.
  - token_agreement_rate@N: какая доля первых N сгенерированных токенов
    совпадает с bf16. Defense должен ПОВЫСИТЬ её.
  - Mean logit KL vs bf16 per step.

Если defense reduces first_divergence_step → результат в lab прокидывается
в production.

Usage:
    python scripts/19_defense_validation.py --n-problems 10 --max-tokens 100
"""
from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from kvtrace.capture.attention_hooks import install_capture_hooks
from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.fp8_sim import QUANT_FNS, make_calibrated_defense
from kvtrace.dataset_loader import load_math_dataset

log = logging.getLogger("phase19")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_USER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 19: defense end-to-end validation")
    p.add_argument("--model", default="qwen3-1.7b")
    p.add_argument("--n-problems", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b/analysis"))
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _build_prompt_tokens(tokenizer, problem_text: str) -> list[int]:
    messages = [{"role": "user",
                 "content": f"{problem_text}\n\n{DEFAULT_USER_INSTRUCTION}"}]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=True, return_tensors="pt",
    )[0].tolist()
    return ids


def _generate_with_quant(runner: CaptureRunner, prompt_tokens: list[int],
                         max_new_tokens: int, quant_name_or_fn) -> dict:
    """Greedy AR generation with given KV-quant. Returns dict with tokens + logits.

    quant_name_or_fn: либо str (lookup в QUANT_FNS), либо callable
        (используется напрямую — для calibrated defense_fn).
    """
    if isinstance(quant_name_or_fn, str):
        quant_fn = QUANT_FNS[quant_name_or_fn]
    else:
        quant_fn = quant_name_or_fn
    handle = install_capture_hooks(
        runner._model,
        attention_modules=runner._attention_modules,
        quant_fn=quant_fn,
    )
    try:
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        with torch.no_grad():
            out = runner._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=runner._tokenizer.eos_token_id,
            )
        sequences = out.sequences[0].tolist()
        gen_tokens = sequences[len(prompt_tokens):]
        # logits: list of [1, vocab] tensors per step
        logits_per_step = torch.stack([s[0] for s in out.scores], dim=0)  # [n_gen, vocab]
        return {
            "gen_tokens": gen_tokens,
            "logits": logits_per_step.float(),
        }
    finally:
        handle.remove()


def _collect_calibration_K(runner: CaptureRunner, prompt_tokens: list[int]) -> dict[int, torch.Tensor]:
    """Один bf16 prefill forward → собрать K_pre per layer.

    K layout: [B=1, num_kv_heads, prompt_len, head_dim]. Берём первые N_layers
    элементов из handle.k_pre (по одному на слой за один forward).
    """
    handle = install_capture_hooks(
        runner._model,
        attention_modules=runner._attention_modules,
        quant_fn=QUANT_FNS["bf16"],  # identity, чтобы не искажать calibration
    )
    try:
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        with torch.no_grad():
            runner._model(input_ids)
        n_layers = len(runner._attention_modules)
        # k_pre — один tensor per layer per forward call. Single forward → ровно n_layers.
        calibration = {i: handle.k_pre[i].clone() for i in range(n_layers)}
        return calibration
    finally:
        handle.remove()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset (math-500)...")
    problems = load_math_dataset("math-500", num_samples=args.n_problems + 60, shuffle=False)
    # Skip first 50 problems (those were in training set) — take fresh ones
    problems = problems[60:60 + args.n_problems]
    log.info("Got %d fresh problems (indices 60-%d)", len(problems), 60 + args.n_problems - 1)

    log.info("Loading Qwen3-1.7B...")
    runner = CaptureRunner()
    runner.load_model("Qwen/Qwen3-1.7B")

    # Три конфига для прямого сравнения:
    #   bf16            — ground truth
    #   fp8_e4m3        — baseline (что quant ломает)
    #   fp8_e4m3_top10  — per-step defense (paper's stateless recipe, AR-нестабилен)
    #   calibrated      — SmoothQuant-style: outliers зафиксированы из prefill K
    configs_static = ["bf16", "fp8_e4m3", "fp8_e4m3_top10"]
    analysis_configs = ["fp8_e4m3", "fp8_e4m3_top10", "fp8_e4m3_top10_calibrated"]
    results: list[dict] = []

    for p_idx, problem in enumerate(problems):
        prompt_tokens = _build_prompt_tokens(runner._tokenizer, problem.problem)
        log.info("  problem %d (orig idx %d): prompt_len=%d",
                 p_idx, problem.idx, len(prompt_tokens))

        # Calibration: один bf16 prefill, snapshot K per layer
        log.info("    calibration: bf16 prefill to extract per-layer K...")
        calib_K = _collect_calibration_K(runner, prompt_tokens)
        calibrated_defense_fn = make_calibrated_defense(calib_K, top_n=10)
        log.info("      done, n_layers=%d", len(calib_K))

        per_config = {}
        for cfg in configs_static:
            log.info("    %s: generating %d tokens...", cfg, args.max_tokens)
            r = _generate_with_quant(runner, prompt_tokens, args.max_tokens, cfg)
            per_config[cfg] = r
            log.info("      done, n_gen=%d", len(r["gen_tokens"]))

        # Calibrated defense — отдельный путь, callable передаётся напрямую
        log.info("    fp8_e4m3_top10_calibrated: generating %d tokens...", args.max_tokens)
        r = _generate_with_quant(runner, prompt_tokens, args.max_tokens, calibrated_defense_fn)
        per_config["fp8_e4m3_top10_calibrated"] = r
        log.info("      done, n_gen=%d", len(r["gen_tokens"]))

        # First divergence step vs bf16 + token agreement
        bf16_tokens = per_config["bf16"]["gen_tokens"]
        bf16_logits = per_config["bf16"]["logits"]
        analysis_per_config = {}
        for cfg in analysis_configs:
            cfg_tokens = per_config[cfg]["gen_tokens"]
            cfg_logits = per_config[cfg]["logits"]
            # First divergence position
            n = min(len(bf16_tokens), len(cfg_tokens))
            first_div = n  # default: no divergence within window
            for i in range(n):
                if bf16_tokens[i] != cfg_tokens[i]:
                    first_div = i
                    break
            # Token agreement
            agreement = sum(int(a == b) for a, b in zip(bf16_tokens[:n], cfg_tokens[:n])) / max(1, n)
            # Logit KL per step
            n_logit = min(bf16_logits.shape[0], cfg_logits.shape[0])
            p_b = torch.softmax(bf16_logits[:n_logit], dim=-1)
            p_c = torch.softmax(cfg_logits[:n_logit], dim=-1)
            eps = 1e-12
            kl_per_step = (p_b * (torch.log(p_b + eps) - torch.log(p_c + eps))).sum(dim=-1).numpy()
            analysis_per_config[cfg] = {
                "first_divergence_step": first_div,
                "token_agreement_rate": agreement,
                "kl_mean": float(np.mean(kl_per_step)),
                "kl_max": float(np.max(kl_per_step)),
                "kl_at_step_10": float(kl_per_step[10]) if n_logit > 10 else None,
                "kl_at_step_50": float(kl_per_step[50]) if n_logit > 50 else None,
                "kl_trajectory": kl_per_step.tolist(),
            }
            log.info("    %s: first_div=%d, agreement=%.2f, mean_kl=%.4f",
                     cfg, first_div, agreement, np.mean(kl_per_step))
        results.append({
            "problem_id": p_idx,
            "orig_idx": problem.idx,
            "prompt_len": len(prompt_tokens),
            "configs": analysis_per_config,
        })

    runner.unload()

    # Aggregate
    summary = {
        "n_problems": len(results),
        "max_tokens": args.max_tokens,
    }
    for cfg in analysis_configs:
        first_divs = [r["configs"][cfg]["first_divergence_step"] for r in results]
        agreements = [r["configs"][cfg]["token_agreement_rate"] for r in results]
        kls = [r["configs"][cfg]["kl_mean"] for r in results]
        summary[cfg] = {
            "first_divergence_step_mean": float(np.mean(first_divs)),
            "first_divergence_step_median": float(np.median(first_divs)),
            "token_agreement_mean": float(np.mean(agreements)),
            "kl_mean": float(np.mean(kls)),
        }

    # Defense effectiveness vs baseline (fp8_e4m3): two defenses сравниваем
    baseline = summary["fp8_e4m3"]
    summary["defense_effect"] = {}
    for defense_name in ["fp8_e4m3_top10", "fp8_e4m3_top10_calibrated"]:
        defended = summary[defense_name]
        effect = {
            "first_div_step_delay_mean": defended["first_divergence_step_mean"] - baseline["first_divergence_step_mean"],
            "first_div_step_delay_median": defended["first_divergence_step_median"] - baseline["first_divergence_step_median"],
            "agreement_improvement": defended["token_agreement_mean"] - baseline["token_agreement_mean"],
            "kl_reduction_pct": 100 * (baseline["kl_mean"] - defended["kl_mean"]) / baseline["kl_mean"],
        }
        summary["defense_effect"][defense_name] = effect
        log.info("=== Defense effectiveness: %s vs baseline FP8 ===", defense_name)
        log.info("  Mean first divergence step: baseline=%.1f, defended=%.1f (delay=%+.1f steps)",
                 baseline["first_divergence_step_mean"], defended["first_divergence_step_mean"],
                 effect["first_div_step_delay_mean"])
        log.info("  Token agreement: baseline=%.3f, defended=%.3f (improvement=%+.3f)",
                 baseline["token_agreement_mean"], defended["token_agreement_mean"],
                 effect["agreement_improvement"])
        log.info("  Mean KL: baseline=%.4f, defended=%.4f (reduction=%+.1f%%)",
                 baseline["kl_mean"], defended["kl_mean"],
                 effect["kl_reduction_pct"])

    # Save (without per-step KL trajectories — too verbose)
    out_path = args.output_dir / "defense_validation_e2e.json"
    summary["per_problem"] = [
        {
            "problem_id": r["problem_id"],
            "orig_idx": r["orig_idx"],
            "prompt_len": r["prompt_len"],
            "configs": {
                cfg: {k: v for k, v in r["configs"][cfg].items() if k != "kl_trajectory"}
                for cfg in r["configs"]
            },
        } for r in results
    ]
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("Saved %s", out_path)

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return 0
        plot_dir = args.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        # Mean KL trajectory across problems for each config
        max_len = max(len(r["configs"]["fp8_e4m3"]["kl_trajectory"]) for r in results)
        for cfg, color in [
            ("fp8_e4m3", "C3"),
            ("fp8_e4m3_top10", "C1"),
            ("fp8_e4m3_top10_calibrated", "C2"),
        ]:
            mat = np.full((len(results), max_len), np.nan)
            for i, r in enumerate(results):
                traj = r["configs"][cfg]["kl_trajectory"]
                mat[i, :len(traj)] = traj
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            x = np.arange(len(mean))
            ax.plot(x, mean, color=color, label=cfg, linewidth=2)
            ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
        ax.set_xlabel("Decode step (from start of generation)")
        ax.set_ylabel("Mean KL(bf16 || config)")
        ax.set_title(f"End-to-end defense validation — n={len(results)} fresh problems")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "defense_validation_kl_trajectory.png", dpi=100)
        plt.close()
        log.info("Saved KL trajectory plot")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
