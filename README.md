# DeepSeek-R1 KV Cache Quantization — Trace-Level Diagnostic Study

A minimal, reproducible pipeline for generating and comparing reasoning
traces from the **DeepSeek-R1-Distill-Qwen** models with and without KV
cache quantization, using **vLLM offline batched inference** on a single
**NVIDIA RTX 3090 (24 GB, Ampere)**.

The pipeline runs the same AIME-24 / MATH-500 problems through two
configurations — a BF16 baseline KV cache and an FP8 KV cache — and stores
the complete `<think>...</think>` trace, final response, boxed answer, and
full raw output side-by-side as JSONL, ready for counterfactual analysis.

---

## Hardware note: FP8 on Ampere

The RTX 3090 is Ampere (compute capability 8.6). **Ampere has no native FP8
tensor cores** — those only arrive in Hopper (H100) and Ada Lovelace (L40S,
RTX 4090/5090). What vLLM does on Ampere is *storage-only* quantization:
the KV cache is stored in FP8 but dequantized to BF16/FP16 at
attention-compute time. This still roughly halves KV cache memory, and it
induces exactly the kind of numerical drift a trace-level diagnostic study
is designed to measure.

### `--kv_dtype` choices

| CLI value | vLLM `kv_cache_dtype` | RTX 3090 status |
|---|---|---|
| `auto` | `auto` | **Baseline.** Matches model dtype (BF16/FP16). |
| `fp8_e5m2` | `fp8_e5m2` | **Recommended quantized config.** Wider dynamic range; the well-trodden Ampere path. |
| `fp8_e4m3` | `fp8_e4m3` | Works, but E4M3 is tuned for Hopper. Numerically less well-behaved on Ampere. |
| `fp8` | `fp8` | Alias for `fp8_e4m3` in current vLLM. |
| `int8` | *(not a valid vLLM value)* | Exposed for discoverability. The script raises a clear error pointing to `fp8_e5m2`. See note below. |

vLLM's `--kv-cache-dtype` runtime flag only accepts
`auto | fp8 | fp8_e5m2 | fp8_e4m3`. "INT8 KV cache" in vLLM proper means a
model-specific quantization recipe that goes through the model quantization
path, not this runtime flag — which is why the script surfaces a
descriptive error instead of silently falling back.

---

## Vast.ai setup

### Recommended Docker image

Either of these work on a CUDA 12.1+ host:

```text
# Option A: official vLLM image (everything pre-baked)
vllm/vllm-openai:v0.6.6

# Option B: PyTorch CUDA base, install vLLM from requirements.txt
pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
```

When renting on Vast.ai:

- **GPU**: 1× RTX 3090 (24 GB)
- **Disk**: ≥ 80 GB (for the 7B model + HF cache + JSONL outputs)
- **CUDA**: 12.1 or 12.4 host driver
- **Ports**: none required for this pipeline (offline inference only)

### Clone + install

```bash
git clone <your-repo>
cd deepseek-kv-trace-study

python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you want to pre-download the model weights (avoids mid-run surprise):

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --local-dir ./models/DeepSeek-R1-Distill-Qwen-7B
```

---

## Running the pipeline

### Baseline (unquantized KV cache)

```bash
python run_vllm_pipeline.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset aime-24 \
    --dataset_size 30 \
    --kv_dtype auto \
    --dtype bfloat16 \
    --output_file outputs/traces_fp16.jsonl
```

### Quantized (FP8 E5M2 — Ampere-friendly)

```bash
python run_vllm_pipeline.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset aime-24 \
    --dataset_size 30 \
    --kv_dtype fp8_e5m2 \
    --dtype bfloat16 \
    --output_file outputs/traces_fp8e5m2.jsonl
```

Both invocations are identical except for `--kv_dtype` and `--output_file`,
which makes the two JSONL files directly comparable line-by-line.

### Fast sanity run on the 1.5B distill

```bash
python run_vllm_pipeline.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset aime-24 \
    --dataset_size 4 \
    --kv_dtype auto \
    --output_file outputs/smoke_test.jsonl
```

---

## Memory tuning for 7B on 24 GB

Rough memory budget at BF16, `gpu_memory_utilization=0.90`:

| Item | Memory |
|---|---|
| 7B model weights (BF16) | ~14 GB |
| vLLM framework overhead | ~1–2 GB |
| Available for KV cache | ~6–7 GB |

At 32k context a single 7B sequence's KV cache is roughly 1.5 GB in BF16
and ~0.75 GB in FP8 — so FP8 is what makes long-context batched AIME runs
practical on this hardware.

If you OOM:

1. Drop `--gpu_memory_utilization` to `0.85`.
2. Cap context: `--max_model_len 16384` (or `8192` for sanity runs).
3. Lower `--max_tokens` — but AIME traces really can exceed 10k tokens;
   cutting too low corrupts the study.
4. Switch to FP8 KV cache (`--kv_dtype fp8_e5m2`) — but that becomes your
   quantized condition, not your baseline.
5. Fall back to the 1.5B distill for iteration.

---

## Output schema

Each line of the output JSONL is one completion:

```json
{
  "idx": 0,
  "sample_idx": 0,
  "source": "aime-24",
  "problem": "Let ...",
  "ground_truth": "204",
  "prompt": "... formatted via chat template, ending in <think>\\n ...",
  "raw_output": "<full generated text including </think> and answer>",
  "think": "<content before </think>>",
  "final_response": "<content after </think>>",
  "boxed_answer": "204",
  "think_complete": true,
  "finish_reason": "stop",
  "num_prompt_tokens": 87,
  "num_generated_tokens": 7412,
  "kv_dtype_cli": "fp8_e5m2",
  "kv_cache_dtype_vllm": "fp8_e5m2",
  "dtype": "bfloat16",
  "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "seed": 42,
  "temperature": 0.0,
  "metadata": { "...": "passthrough from the source dataset" }
}
```

Alongside the JSONL the script also writes `<output>.jsonl.meta.json`
capturing the full run configuration (model, KV dtype, seed, GPU, compute
capability, timestamp) so each run is self-describing.

### Parser semantics for `<think>` blocks

DeepSeek-R1 distills use `apply_chat_template(add_generation_prompt=True)`
to **pre-seed the assistant turn with `<think>\n`** — so the generated
text typically contains only the closing `</think>` tag. `trace_utils.py`
handles all four cases:

1. Both `<think>` and `</think>` present → standard block extraction.
2. Only `</think>` present → everything before it is the trace
   (the DeepSeek-R1 common case).
3. Only `<think>` present → output was truncated mid-trace;
   `think_complete=false`.
4. Neither present → no trace; the full text is the response.

`boxed_answer` uses a brace-matching scan (not a flat regex) so nested
LaTeX like `\boxed{\dfrac{22}{7}}` parses correctly.

---

## Reproducibility

- Greedy decoding: `temperature=0.0`, `top_p=1.0`
- `seed=42` is passed to both the vLLM engine and `SamplingParams`
- Chat template is taken verbatim from the model's tokenizer config
- `shuffle=False` by default — baseline and quantized runs see the exact
  same prompts in the exact same order

Note: bit-exact reproducibility across independent vLLM invocations is
*not* guaranteed — paged attention and chunked prefill can introduce tiny
floating-point non-determinism. In practice greedy + fixed seed gives very
high run-to-run similarity, which is sufficient for a comparative study.
DeepSeek-R1 itself recommends `temperature=0.6` for general use; using
`0.0` is a deliberate trade of output "quality" for determinism so that
any divergence between the baseline and quantized traces is attributable
to the KV cache perturbation and nothing else.

---

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── dataset_loader.py        # AIME-24 / MATH-500 -> MathProblem records
├── trace_utils.py           # <think> extraction + \boxed{} parser (+ self-tests)
└── run_vllm_pipeline.py     # the main entry point
```

Run `python trace_utils.py` as a quick self-test of the parser without
touching a GPU.
