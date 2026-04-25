"""
run_vllm_pipeline.py
--------------------
Generate reasoning traces with DeepSeek-R1-Distill-Qwen using vLLM offline
batched inference, with configurable KV cache quantization.

Usage (baseline, unquantized):

    python run_vllm_pipeline.py \\
        --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
        --dataset aime-24 \\
        --dataset_size 30 \\
        --kv_dtype auto \\
        --output_file outputs/traces_fp16.jsonl

Usage (Ampere-safe quantized KV cache):

    python run_vllm_pipeline.py \\
        --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
        --dataset aime-24 \\
        --dataset_size 30 \\
        --kv_dtype fp8_e5m2 \\
        --output_file outputs/traces_fp8e5m2.jsonl

--------------------------------------------------------------------------
HARDWARE NOTE: RTX 3090 (Ampere, CC 8.6)
--------------------------------------------------------------------------
Ampere GPUs have no native FP8 tensor cores. vLLM's FP8 KV cache on Ampere
works as *storage-only* quantization: values are kept in FP8 in the KV cache
and dequantized to a wider dtype (fp16/bf16) at attention-compute time. That
still gives a ~2x reduction in KV memory footprint and induces numerical
drift in generation - which is exactly the perturbation a trace-level
diagnostic study wants to capture.

  * fp8_e5m2 is the recommended choice on Ampere (wider dynamic range,
    matches the numerics most vLLM tests cover on this hardware).
  * fp8_e4m3 is tuned for Hopper and may be less well-behaved on Ampere.
  * int8 is NOT a valid value for vLLM's --kv-cache-dtype flag. If you
    pass --kv_dtype int8 this script raises a clear error pointing to
    fp8_e5m2 as the substitute.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Defer heavy imports until after arg parsing so --help is fast.


SUPPORTED_KV_DTYPES_CLI = ["auto", "fp8", "fp8_e5m2", "fp8_e4m3", "int8"]

# vLLM EngineArgs accepts: auto | fp8 | fp8_e5m2 | fp8_e4m3
VLLM_SUPPORTED_KV_DTYPES = {"auto", "fp8", "fp8_e5m2", "fp8_e4m3"}

# DeepSeek-R1's recommended inference prompt appends this instruction to the
# user turn. It nudges the model to emit its final answer inside \boxed{}.
DEFAULT_USER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def resolve_kv_cache_dtype(cli_arg: str) -> str:
    """Map the user-facing CLI value to vLLM's kv_cache_dtype string."""
    if cli_arg == "int8":
        raise ValueError(
            "--kv_dtype int8 is not supported by vLLM's --kv-cache-dtype flag. "
            "vLLM accepts: auto | fp8 | fp8_e5m2 | fp8_e4m3. "
            "On RTX 3090 (Ampere, CC 8.6) the closest equivalent with "
            "storage-only quantization is fp8_e5m2 - pass --kv_dtype fp8_e5m2 "
            "instead. (True INT8 KV cache requires a model-specific "
            "quantization recipe that goes through vLLM's model quantization "
            "path, not the runtime KV cache flag.)"
        )
    if cli_arg not in VLLM_SUPPORTED_KV_DTYPES:
        raise ValueError(
            f"Unsupported --kv_dtype {cli_arg!r}. Choose from: "
            f"{sorted(VLLM_SUPPORTED_KV_DTYPES | {'int8'})}"
        )
    return cli_arg


def build_prompts(tokenizer, problems) -> list[str]:
    """Format each problem through the model's chat template.

    DeepSeek-R1 distills rely on apply_chat_template + add_generation_prompt
    to pre-seed the assistant turn with ``<think>\\n``, so the model starts
    reasoning immediately.
    """
    prompts: list[str] = []
    for p in problems:
        messages = [
            {"role": "user", "content": f"{p.problem}\n\n{DEFAULT_USER_INSTRUCTION}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate DeepSeek-R1 reasoning traces via vLLM, with "
                    "optional KV cache quantization for diagnostic study.",
    )
    # --- Model / data ---
    p.add_argument("--model_path", type=str, required=True,
                   help="HF repo id or local path to a DeepSeek-R1-Distill model.")
    p.add_argument("--dataset", type=str, default="aime-24",
                   choices=["aime-24", "aime-24-aimo", "math-500"])
    p.add_argument("--dataset_size", type=int, default=30,
                   help="Number of problems to run. -1 means 'all'.")
    p.add_argument("--output_file", type=str, required=True,
                   help="Path to the output .jsonl file.")

    # --- Quantization ---
    p.add_argument("--kv_dtype", type=str, default="auto",
                   choices=SUPPORTED_KV_DTYPES_CLI,
                   help="KV cache dtype. 'auto' = unquantized baseline. "
                        "For RTX 3090, 'fp8_e5m2' is the recommended quantized option. "
                        "'int8' is exposed for discoverability and raises a helpful error.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "auto"],
                   help="Model weight dtype. BF16 is the reference baseline.")

    # --- Sampling (deterministic) ---
    p.add_argument("--max_tokens", type=int, default=32768,
                   help="Max generated tokens. AIME traces can exceed 16k tokens.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Greedy decoding by default. DeepSeek recommends 0.6 for "
                        "general use; 0.0 gives run-to-run reproducibility which "
                        "is what a diagnostic trace study wants.")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--num_samples_per_problem", type=int, default=1,
                   help="n for SamplingParams; >1 only meaningful if temperature>0.")
    p.add_argument("--seed", type=int, default=42)

    # --- Memory / engine tuning (RTX 3090 24GB) ---
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                   help="Fraction of GPU memory vLLM may claim. 0.90 is a good "
                        "default for a 24GB RTX 3090; drop to 0.85 if you OOM.")
    p.add_argument("--max_model_len", type=int, default=None,
                   help="Override model context length. Lowering this saves KV "
                        "cache memory if you don't need the full window.")
    p.add_argument("--swap_space", type=int, default=4,
                   help="GB of CPU swap for KV cache overflow.")
    p.add_argument("--enforce_eager", action="store_true",
                   help="Disable CUDA graphs. Slower but simpler for OOM debugging.")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="Leave at 1 for a single RTX 3090.")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve KV dtype BEFORE loading vLLM so we fail fast on bad args.
    try:
        kv_cache_dtype = resolve_kv_cache_dtype(args.kv_dtype)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    # Heavy imports
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from dataset_loader import load_math_dataset
    from trace_utils import parse_trace

    # --- Hardware report --------------------------------------------------
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        print(f"[hw ] GPU: {dev_name}  compute capability {cc[0]}.{cc[1]}")
        if cc == (8, 6) and kv_cache_dtype in ("fp8", "fp8_e4m3"):
            print(
                "[hw ] WARNING: You selected an E4M3-flavor FP8 KV cache on an "
                "Ampere GPU. Ampere has no native FP8 tensor cores; vLLM will "
                "use storage-only quantization. This works, but fp8_e5m2 is "
                "the more commonly validated path on RTX 3090."
            )
    else:
        print("[hw ] WARNING: No CUDA device visible. vLLM will fail to start.")

    # --- Config echo ------------------------------------------------------
    print(f"[cfg] model                    = {args.model_path}")
    print(f"[cfg] dtype                    = {args.dtype}")
    print(f"[cfg] kv_dtype (cli)           = {args.kv_dtype}")
    print(f"[cfg] kv_cache_dtype (vllm)    = {kv_cache_dtype}")
    print(f"[cfg] gpu_memory_utilization   = {args.gpu_memory_utilization}")
    print(f"[cfg] max_model_len            = {args.max_model_len}")
    print(f"[cfg] max_tokens               = {args.max_tokens}")
    print(f"[cfg] temperature              = {args.temperature}")
    print(f"[cfg] seed                     = {args.seed}")

    # --- Load dataset -----------------------------------------------------
    n = None if args.dataset_size < 0 else args.dataset_size
    print(f"[data] loading {args.dataset} (n={n})")
    problems = load_math_dataset(name=args.dataset, num_samples=n, shuffle=False)
    print(f"[data] loaded {len(problems)} problems")

    # --- Tokenizer & prompts ---------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = build_prompts(tokenizer, problems)
    if len(prompts) > 0:
        # Show first prompt (truncated) so users can eyeball the chat template.
        preview = prompts[0].replace("\n", "\\n")
        print(f"[data] first prompt (truncated): {preview[:240]}...")

    # --- Engine -----------------------------------------------------------
    llm_kwargs = dict(
        model=args.model_path,
        dtype=args.dtype,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        seed=args.seed,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    print("[vllm] initializing engine...")
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[vllm] engine ready in {time.time() - t0:.1f}s")

    sampling_params = SamplingParams(
        n=args.num_samples_per_problem,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # --- Output paths -----------------------------------------------------
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out.with_suffix(out.suffix + ".meta.json")

    run_meta = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "dataset_size": len(problems),
        "kv_dtype_cli": args.kv_dtype,
        "kv_cache_dtype_vllm": kv_cache_dtype,
        "dtype": args.dtype,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available() else None
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with meta_path.open("w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[out ] run metadata -> {meta_path}")

    # --- Generate ---------------------------------------------------------
    print(f"[gen ] running {len(prompts)} prompts through vLLM...")
    t0 = time.time()
    # vLLM processes the whole list in one scheduled batch and returns
    # RequestOutputs aligned with the input order.
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    gen_time = time.time() - t0
    print(f"[gen ] finished in {gen_time:.1f}s "
          f"({gen_time / max(len(prompts), 1):.1f}s / problem)")

    # --- Write JSONL ------------------------------------------------------
    n_written = 0
    with out.open("w") as fout:
        for problem, prompt, result in zip(problems, prompts, outputs, strict=False):
            n_prompt_tokens = len(result.prompt_token_ids)
            for sample_i, completion in enumerate(result.outputs):
                raw = completion.text
                parsed = parse_trace(raw, finish_reason=completion.finish_reason)
                record = {
                    # --- identity
                    "idx": problem.idx,
                    "sample_idx": sample_i,
                    "source": problem.source,
                    # --- input
                    "problem": problem.problem,
                    "ground_truth": problem.answer,
                    "prompt": prompt,
                    # --- output
                    "raw_output": raw,
                    "think": parsed.think,
                    "final_response": parsed.final_response,
                    "boxed_answer": parsed.boxed_answer,
                    "think_complete": parsed.think_complete,
                    "finish_reason": completion.finish_reason,
                    # --- accounting
                    "num_prompt_tokens": n_prompt_tokens,
                    "num_generated_tokens": len(completion.token_ids),
                    # --- run config stamp (so each record is self-describing)
                    "kv_dtype_cli": args.kv_dtype,
                    "kv_cache_dtype_vllm": kv_cache_dtype,
                    "dtype": args.dtype,
                    "model_path": args.model_path,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    # --- dataset metadata passthrough
                    "metadata": problem.metadata,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()  # flush per record so the file is usable mid-run
                n_written += 1

    print(f"[out ] wrote {n_written} records -> {out}")

    # --- Quick quality summary --------------------------------------------
    n_think_complete = 0
    n_has_boxed = 0
    n_truncated = 0
    with out.open() as fin:
        for line in fin:
            r = json.loads(line)
            if r.get("think_complete"):
                n_think_complete += 1
            if r.get("boxed_answer"):
                n_has_boxed += 1
            if r.get("finish_reason") == "length":
                n_truncated += 1
    print(f"[stat] think-closed: {n_think_complete}/{n_written}  "
          f"boxed-answer: {n_has_boxed}/{n_written}  "
          f"hit-max-tokens: {n_truncated}/{n_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
