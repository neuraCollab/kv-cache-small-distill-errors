"""Phase 16: multi-seed AR capture at T=0.6 для variance estimates.

Запускает AR generation с do_sample=True T=0.6 для нескольких seeds.
Каждая комбинация (seed, problem, quant) даёт независимую trajectory.

Цель: измерить sampling variance для метрик paper'а:
  - KL trajectory shape variance (τ, K∞ CIs)
  - FDP position variance per problem
  - Outlier-channel pattern stability across seeds

Структура output:
  outputs/kv_capture/qwen3-1.7b_multiseed/seed{S}/<quant>_ar/<pid>.safetensors

Каждый файл = full AR capture: prompt prefill + 250 generated tokens
с Q/K/V/q_post_rope/logits. Окно = весь сгенерированный участок
(нет FDP-windowing потому что с sampling FDP теряет смысл).

Stoимость: N_seeds × N_problems × N_quants × ~1.5 мин AR.
  Для 3×20×3 = 180 generations ≈ 4-5 часов.

Usage:
    python scripts/16_multiseed_ar.py --seeds 1 2 3 --n-problems 20 \\
        --quants bf16 fp8_e4m3 fp8_e5m2 --max-new-tokens 250
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from kvtrace.capture.attention_hooks import install_capture_hooks
from kvtrace.capture.cpu_runner import _slice_kv, _slice_q
from kvtrace.capture.fp8_sim import QUANT_FNS
from kvtrace.capture.storage import CaptureData, save_capture
from kvtrace.capture.window import Window
from kvtrace.dataset_loader import load_math_dataset

log = logging.getLogger("phase16")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_USER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 16: multi-seed AR capture")
    p.add_argument("--model", default="qwen3-1.7b")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--n-problems", type=int, default=20)
    p.add_argument("--source", default="aime", choices=["aime", "math"])
    p.add_argument("--quants", nargs="+", default=["bf16", "fp8_e4m3", "fp8_e5m2"])
    p.add_argument("--max-new-tokens", type=int, default=250)
    p.add_argument("--output-root", type=Path,
                   default=Path("outputs/kv_capture"))
    return p.parse_args()


MODEL_HF_IDS = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_prompt_tokens(tokenizer, problem_text: str) -> list[int]:
    messages = [{"role": "user",
                 "content": f"{problem_text}\n\n{DEFAULT_USER_INSTRUCTION}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=True, return_tensors="pt",
    )[0].tolist()


def _capture_one(
    model, attention_modules, tokenizer, prompt_tokens: list[int],
    quant: str, temperature: float, seed: int,
    max_new_tokens: int, problem_id: int, model_name: str, model_hash: str,
) -> CaptureData:
    """Multi-seed AR: prefill prompt → generate max_new_tokens at T, capture."""
    _set_all_seeds(seed)
    quant_fn = QUANT_FNS[quant]
    handle = install_capture_hooks(
        model, attention_modules=attention_modules, quant_fn=quant_fn,
    )
    try:
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        prefix_len = len(prompt_tokens)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.05,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        full_sequence = out.sequences[0].tolist()
        gen_logits = torch.stack(list(out.scores), dim=0)
        if gen_logits.dim() == 3:
            gen_logits = gen_logits[:, 0, :]
        n_generated = gen_logits.shape[0]
        early_eos = n_generated < max_new_tokens
        # Window = всё сгенерированное (no FDP)
        # Indices: [prefix_len, prefix_len + n_generated)
        ws = prefix_len
        we = prefix_len + n_generated
        W = we - ws

        # Stitch per-layer Q/K/V across prefill+decode calls.
        n_layers = len(attention_modules)
        q_all = [_concat_along_seq(handle.q[i::n_layers]) for i in range(n_layers)]
        q_post_all = (
            [_concat_along_seq(handle.q_post_rope[i::n_layers]) for i in range(n_layers)]
            if handle.q_post_rope else None
        )
        k_pre_all = [_concat_along_seq(handle.k_pre[i::n_layers]) for i in range(n_layers)]
        v_pre_all = [_concat_along_seq(handle.v_pre[i::n_layers]) for i in range(n_layers)]
        k_post_all = [_concat_along_seq(handle.k_post[i::n_layers]) for i in range(n_layers)]
        v_post_all = [_concat_along_seq(handle.v_post[i::n_layers]) for i in range(n_layers)]

        # Slice the WS:WE region (generation only)
        q_sliced = [_slice_q(t, ws, we) for t in q_all]
        q_post_sliced = (
            [_slice_kv(t, ws, we) for t in q_post_all]
            if q_post_all else None
        )
        k_pre = [_slice_kv(t, ws, we) for t in k_pre_all]
        v_pre = [_slice_kv(t, ws, we) for t in v_pre_all]
        k_post = [_slice_kv(t, ws, we) for t in k_post_all]
        v_post = [_slice_kv(t, ws, we) for t in v_post_all]
        logits = gen_logits.to(torch.float16).contiguous()  # [n_generated, vocab]

        window = Window(ws=ws, we=we, truncated_left=False, truncated_right=False)
        import transformers
        meta = {
            "model": model_name,
            "quant": quant,
            "mode": "ar_multiseed",
            "seed": seed,
            "temperature": temperature,
            "problem_id": problem_id,
            "fdp_token_idx": -1,  # no FDP for sampled traces
            "window_start": ws,
            "window_end": we,
            "W": W,
            "prefix_len": prefix_len,
            "input_token_ids": list(prompt_tokens),
            "gen_token_ids": full_sequence[prefix_len:],
            "truncated_left": False,
            "truncated_right": False,
            "early_eos": early_eos,
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "model_revision_hash": model_hash,
            "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        return CaptureData(
            meta=meta, q=q_sliced, k_pre=k_pre, v_pre=v_pre,
            k_post=k_post, v_post=v_post, logits=logits,
            q_post_rope=q_post_sliced,
        )
    finally:
        handle.remove()


def _concat_along_seq(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Concat along seq dim=-2. Handles non-overlapping per-call captures."""
    if not tensors:
        return torch.empty(0)
    try:
        return torch.cat(tensors, dim=-2)
    except RuntimeError:
        return max(tensors, key=lambda t: t.numel())


def main() -> int:
    args = parse_args()
    hf_id = MODEL_HF_IDS[args.model]

    log.info("Loading dataset: %s", args.source)
    ds_name = "aime-24" if args.source == "aime" else "math-500"
    problems = load_math_dataset(ds_name, num_samples=args.n_problems, shuffle=False)

    log.info("Loading %s on CPU...", hf_id)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16,
        device_map={"": "cpu"}, trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    attention_modules = [layer.self_attn for layer in model.model.layers]
    model_hash = getattr(model.config, "_commit_hash", "unknown")

    plan = []
    for seed in args.seeds:
        for problem in problems:
            for quant in args.quants:
                plan.append((seed, problem, quant))
    log.info("Plan: %d × %d × %d = %d AR generations (T=%.2f)",
             len(args.seeds), len(problems), len(args.quants),
             len(plan), args.temperature)

    output_root = args.output_root / f"{args.model}_multiseed"
    for i, (seed, problem, quant) in enumerate(plan):
        out_dir = output_root / f"seed{seed}" / f"{quant}_ar"
        out_path = out_dir / f"{problem.idx}.safetensors"
        if out_path.exists():
            log.info("  [%d/%d] exists, skip: %s", i+1, len(plan), out_path)
            continue
        prompt_tokens = _build_prompt_tokens(tokenizer, problem.problem)
        log.info("  [%d/%d] seed=%d, problem=%d, quant=%s, prompt_len=%d",
                 i+1, len(plan), seed, problem.idx, quant, len(prompt_tokens))
        try:
            cap = _capture_one(
                model, attention_modules, tokenizer, prompt_tokens,
                quant=quant, temperature=args.temperature, seed=seed,
                max_new_tokens=args.max_new_tokens,
                problem_id=problem.idx, model_name=args.model, model_hash=model_hash,
            )
            save_capture(cap, out_path)
            log.info("    saved %s (W=%d, n_gen=%d)", out_path, cap.meta["W"], len(cap.meta["gen_token_ids"]))
        except Exception as e:
            log.exception("FAILED: seed=%d problem=%d quant=%s: %s",
                          seed, problem.idx, quant, e)
            continue

    log.info("Done. Output: %s", output_root)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
