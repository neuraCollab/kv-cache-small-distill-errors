"""Phase 12: cross-architecture validation на Qwen3-4B / Qwen3-8B.

Цель: проверить что механизм расхождения (outlier-channel concentration,
per-layer ablation ranking, attention shift), найденный на Qwen3-1.7B,
ОБОБЩАЕТСЯ на другие размеры/архитектуры.

Поскольку у новой модели НЕТ existing FDPs (vLLM run был только на 1.7B),
работаем по упрощённой схеме:

  1. Берём N задач из AIME-24 / MATH-500 (тех же)
  2. Применяем chat_template + tokenize → prompt_tokens
  3. Делаем TF forward на prompt (без авторегрессии) — capture K/V для
     prompt-позиций. FDP-окно не применяется (берём full prompt).
  4. Для bf16, fp8_e4m3, fp8_e5m2 → 3 forward'а на model
  5. Сохраняем в outputs/kv_capture/<model>/<quant>_tf_prompt/<pid>.safetensors
  6. Запускаем outlier_channel_impact на новых файлах
  7. Сравниваем concentration metrics с Qwen3-1.7B

Это даёт material для paper's section "Generalization across model sizes".
Если ту же ~10% top-1 channel concentration видим на 4B/8B → mechanism
не Qwen3-1.7B-specific.

Note: capture WITHOUT FDP-окно потому что для новой модели мы не знаем
где квант реально разводит модель. Анализ static (на prompt) ≠ dynamic
(на generated tokens), но outlier-distribution в K-pre — структурное
свойство model weights × prompt tokens, **не** trajectory-зависимое.

Usage:
    python scripts/12_cross_arch_validation.py --model qwen3-4b --n-problems 10
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path

import torch

from kvtrace.capture.attention_hooks import install_capture_hooks
from kvtrace.capture.cpu_runner import _slice_kv, _slice_q
from kvtrace.capture.fp8_sim import QUANT_FNS
from kvtrace.capture.storage import CaptureData, save_capture
from kvtrace.capture.window import Window
from kvtrace.dataset_loader import load_math_dataset

log = logging.getLogger("phase12")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_HF_IDS = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}

DEFAULT_USER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 12: cross-arch validation")
    p.add_argument("--model", required=True, choices=list(MODEL_HF_IDS))
    p.add_argument("--n-problems", type=int, default=10)
    p.add_argument("--quants", nargs="+", default=["bf16", "fp8_e4m3", "fp8_e5m2"])
    p.add_argument("--source", default="aime", choices=["aime", "math"])
    p.add_argument("--output-root", type=Path,
                   default=Path("outputs/kv_capture"))
    return p.parse_args()


def _build_prompt_tokens(tokenizer, problem_text: str) -> list[int]:
    """Тот же chat_template что в vllm_gen.py для consistency."""
    messages = [{"role": "user",
                 "content": f"{problem_text}\n\n{DEFAULT_USER_INSTRUCTION}"}]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=True, return_tensors="pt",
    )[0].tolist()
    return ids


def _capture_one(model, attention_modules, tokenizer, prompt_tokens, quant: str,
                 problem_id: int, model_name: str, model_hash: str) -> CaptureData:
    """One TF forward → CaptureData покрывающая ВЕСЬ prompt."""
    handle = install_capture_hooks(
        model, attention_modules=attention_modules,
        quant_fn=QUANT_FNS[quant],
    )
    try:
        T = len(prompt_tokens)
        feed_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=feed_ids, use_cache=True)
        logits_full = out.logits[0]  # [T, vocab]
        # Window = full prompt (нет FDP)
        window = Window(ws=0, we=T, truncated_left=False, truncated_right=False)
        q_sliced = [_slice_q(t, 0, T) for t in handle.q]
        q_post_sliced = (
            [_slice_kv(t, 0, T) for t in handle.q_post_rope]
            if handle.q_post_rope else None
        )
        k_pre = [_slice_kv(t, 0, T) for t in handle.k_pre]
        v_pre = [_slice_kv(t, 0, T) for t in handle.v_pre]
        k_post = [_slice_kv(t, 0, T) for t in handle.k_post]
        v_post = [_slice_kv(t, 0, T) for t in handle.v_post]
        logits = logits_full.to(torch.float16).contiguous()

        import transformers
        meta = {
            "model": model_name,
            "quant": quant,
            "mode": "tf_prompt",  # marker: this is prompt-only TF (no FDP)
            "problem_id": problem_id,
            "fdp_token_idx": -1,  # n/a for cross-arch (no FDP determined)
            "window_start": 0,
            "window_end": T,
            "W": T,
            "input_token_ids": list(prompt_tokens),
            "gen_token_ids": list(prompt_tokens),
            "truncated_left": False,
            "truncated_right": False,
            "early_eos": False,
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


def main() -> int:
    args = parse_args()
    hf_id = MODEL_HF_IDS[args.model]

    log.info("Loading dataset: %s", args.source)
    ds_name = "aime-24" if args.source == "aime" else "math-500"
    problems = load_math_dataset(ds_name, num_samples=args.n_problems, shuffle=False)
    log.info("Got %d problems", len(problems))

    log.info("Loading %s on CPU (bf16)...", hf_id)
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
    log.info("Model loaded: %d layers, vocab=%d, %d attention modules",
             model.config.num_hidden_layers, model.config.vocab_size,
             len(attention_modules))

    output_root = args.output_root / args.model
    log.info("Output dir: %s", output_root)

    for p_idx, problem in enumerate(problems):
        prompt_tokens = _build_prompt_tokens(tokenizer, problem.problem)
        log.info("  problem %d: prompt_len=%d", p_idx, len(prompt_tokens))
        for quant in args.quants:
            out_path = output_root / f"{quant}_tf_prompt" / f"{p_idx}.safetensors"
            if out_path.exists():
                log.info("    %s: exists, skip", quant)
                continue
            cap = _capture_one(
                model, attention_modules, tokenizer, prompt_tokens,
                quant=quant, problem_id=p_idx,
                model_name=args.model, model_hash=model_hash,
            )
            save_capture(cap, out_path)
            log.info("    %s: saved %s (W=%d)", quant, out_path, cap.meta["W"])

    log.info("Done. Now run:")
    log.info("  python scripts/11_outlier_analysis.py --captures-dir %s --quant fp8_e4m3 --mode tf_prompt", output_root)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
