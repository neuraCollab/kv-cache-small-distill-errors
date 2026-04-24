# KV Cache Quantization — Trace-Level Diagnostic Study

**Design document**
**Date:** 2026-04-25
**Status:** Approved — ready for implementation planning

---

## 1. Goal

Build a reproducible research pipeline that diagnoses **how** KV cache
quantization breaks reasoning in compact (1.5B–3B-class) reasoning models,
not just how much it degrades accuracy. Output is a paper-ready report
showing that different quantization methods produce characteristic failure
**signatures** across six error categories.

The deliverables are: a public HF dataset with all generated traces, First
Divergence Points (FDP), and Claude judgments; a markdown + JSON report
with statistics and plots; and full test coverage of every processing
stage. The paper itself is out of scope.

## 2. Scope

### In scope

- Data generation for 3 models × 5 quantization configs × 80 problems.
- Automatic First Divergence Point detection (hybrid token-level +
  semantic re-sync).
- LLM-as-a-judge classification into 6 error categories (A–F) via
  Anthropic Claude Sonnet 4.6 with prompt caching.
- Failure-signature analysis: confusion matrix, chi-square, Cramér's V,
  per-model breakdown.
- Idempotent 4-phase pipeline resumable from HuggingFace Hub snapshots.
- Unit + integration tests mapped to each of the four task areas; CPU-only
  CI; GPU and live-API tests behind pytest markers.

### Out of scope

- Implementing novel KV quantization algorithms (we use what vLLM and HF
  Transformers + HQQ provide natively).
- Paper writing, figure polishing, literature review.
- Models outside the 1.5B–7B range.
- Non-math reasoning datasets (GPQA, Big-Bench, etc.).
- Serving / real-time inference — everything is offline batch.

## 3. Hardware & budget

**GPU:** single RTX 4090 (Ada, CC 8.9) on Vast.ai, ~$0.40/hour. Ada is
chosen over Ampere because it has native FP8 tensor cores, making
`fp8_e5m2` and `fp8_e4m3` qualitatively distinct — this is essential to
the "different methods break differently" thesis.

**Budget envelope:**

| Line item | Cost |
|---|---|
| Compute (~14 GPU-hours @ $0.40/h) | ~$5.60 |
| Claude Sonnet 4.6 judge (~960 calls, prompt cached) | ~$3.00 |
| HuggingFace Hub (public dataset) | $0 |
| Contingency (OOM retries, re-runs) | ~$3.00 |
| **Total** | **~$12** |

Hard budget ceiling: $20 (user approved a modest increase over the
original $10). Fallback plan if burn rate exceeds $15 is documented in §10.

## 4. Models and quantization configurations

### Models

| Model ID | Purpose |
|---|---|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | primary compact reasoning baseline |
| `Qwen/Qwen3-1.7B` (thinking mode) | second training recipe for cross-model signature comparison |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | scale anchor for reviewer confidence |

### Quantization configs (all applied to **KV cache**, not weights)

| Config name | Engine | Mechanism |
|---|---|---|
| `bf16` | vLLM | no KV quantization — reference baseline |
| `fp8_e5m2` | vLLM | `kv_cache_dtype="fp8_e5m2"` — wider dynamic range |
| `fp8_e4m3` | vLLM | `kv_cache_dtype="fp8_e4m3"` — tighter precision |
| `hqq_int4` | HF Transformers | `QuantizedCacheConfig(backend="HQQ", nbits=4)` |
| `hqq_int2` | HF Transformers | `QuantizedCacheConfig(backend="HQQ", nbits=2)` |

Two generator engines are unavoidable: vLLM gives speed and native FP8,
HF Transformers is the only mainline path to INT4/INT2 KV through HQQ.
The `Generator` ABC hides this difference from downstream phases.

## 5. Dataset

80 problems total:

- **30 AIME-24** problems — long, dense, multi-step reasoning where
  divergences are most informative.
- **50 MATH-500** problems — medium difficulty, broader topic coverage,
  gives statistical power for the chi-square test.

Both are loaded via the existing `dataset_loader.py` module (extended to
expose MATH-500 via the first 50 rows of its `test` split with
`shuffle=False`; the selection is deterministic from ordering alone, no
seed required).

Baseline and quantized runs always see the **same 80 problems in the same
order** — this is the precondition for per-problem trace pairing.

## 6. Architecture

### 6.1 Repository layout

```
kv-trace-study/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── Makefile
├── config/
│   ├── models.yaml             # 3 models + their chat templates + engine hints
│   ├── quant_methods.yaml      # 5 configs with per-config overrides
│   └── pipeline.yaml           # dataset, seed, judge model, HF repo id, FDP params
├── src/kvtrace/
│   ├── __init__.py
│   ├── dataset_loader.py       # EXTENDED from baseline
│   ├── trace_utils.py          # REUSED from baseline (no changes)
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── base.py             # Generator ABC, GenerationResult dataclass
│   │   ├── vllm_gen.py         # bf16/fp8 variants
│   │   └── hf_gen.py           # HQQ INT4/INT2
│   ├── fdp/
│   │   ├── __init__.py
│   │   ├── tokenizer_align.py
│   │   ├── semantic_resync.py  # MiniLM-L6-v2 singleton, lazy load
│   │   └── finder.py           # hybrid algorithm
│   ├── judge/
│   │   ├── __init__.py
│   │   ├── taxonomy.py         # 6 categories A–F (TAXONOMY_V1)
│   │   ├── prompt.py           # versioned template, cache_control on taxonomy
│   │   ├── claude_judge.py     # Anthropic SDK client + on-disk SHA256 cache
│   │   └── golden_set.py       # 10 curated (trace_pair, gold category) for calibration
│   ├── hf_hub/
│   │   ├── __init__.py
│   │   ├── upload.py           # idempotent dataset revisions
│   │   └── download.py         # resume-from-hub helper
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── signatures.py       # confusion matrix, chi-square, Cramér's V
│   │   └── report.py           # markdown + JSON + plots
│   └── memory.py               # free_gpu() context manager
├── scripts/
│   ├── 01_generate_traces.py
│   ├── 02_find_fdps.py
│   ├── 03_judge_fdps.py
│   ├── 04_analyze.py
│   └── run_all.sh
├── tests/
│   ├── conftest.py
│   ├── fixtures/               # golden traces, mocked judge responses, tiny datasets
│   ├── test_trace_utils.py
│   ├── test_dataset_loader.py
│   ├── test_generators_vllm.py
│   ├── test_generators_hf.py
│   ├── test_generator_contract.py
│   ├── test_fdp_finder.py
│   ├── test_tokenizer_align.py
│   ├── test_semantic_resync.py
│   ├── test_taxonomy.py
│   ├── test_judge_prompt.py
│   ├── test_judge_response_parsing.py
│   ├── test_judge_cache.py
│   ├── test_judge_claude_mock.py
│   ├── test_judge_calibration.py   # marked live_api
│   ├── test_signatures.py
│   ├── test_report.py
│   ├── test_memory_free.py         # marked gpu
│   ├── test_hf_hub_upload.py
│   ├── test_vllm_smoke.py          # marked gpu
│   ├── test_hqq_int2_7b_fits.py    # marked gpu
│   └── test_end_to_end_smoke.py    # fully mocked, <30s, gating
├── outputs/                     # .gitignore'd
├── docs/
│   └── superpowers/specs/
└── .github/workflows/ci.yml
```

### 6.2 Dataflow

Four idempotent phases. GPU is held only during Phase 1 and released in
full before Phase 2 starts.

```
Phase 1 (GENERATE, GPU)
  for each (model, config):
    skip if HF revision `traces-{model}-{config}` already exists
    Generator.load() → generate(80 problems) → save JSONL locally
    HF upload with revision tag
    Generator.unload()   ← releases GPU fully
  yields 15 JSONL files

Phase 2 (FDP, CPU + ~80MB MiniLM)
  for each model:
    for each quant_config != bf16:
      read paired JSONL (bf16 vs quant)
      per problem: run hybrid FDP finder
      save FDPRecord JSONL + HF upload
  yields 12 JSONL files (3 models × 4 quant configs)

Phase 3 (JUDGE, CPU + Claude API)
  for each FDPRecord:
    construct prompt (taxonomy is prompt-cached)
    SHA256 → skip API call if cached
    Claude Sonnet 4.6 → JudgmentResult (pydantic-validated JSON)
    cache + save to outputs/judgments/*.jsonl
  yields 12 JSONL files

Phase 4 (ANALYZE, CPU only)
  aggregate judgments → confusion matrix (method × {A..F})
  chi-square, Cramér's V, per-model breakdown, plots
  yields outputs/report.md, outputs/report.json, outputs/plots/*.png
```

Each phase reads only from disk or HF — never from in-process state — so
a killed Vast.ai instance can be resumed with `bash scripts/run_all.sh`
without recomputing anything already uploaded to HF.

## 7. Component specifications

### 7.1 Generator ABC

```python
class Generator(ABC):
    @abstractmethod
    def load(self, model_id: str, quant_config: QuantConfig) -> None: ...
    @abstractmethod
    def generate(self, problems: list[MathProblem]) -> list[GenerationResult]: ...
    @abstractmethod
    def unload(self) -> None: ...
    def __enter__(self): self.load(...); return self
    def __exit__(self, *args): self.unload()
```

`GenerationResult` wraps the existing `ParsedTrace` plus `token_ids: list[int]`
(needed for FDP tokenizer alignment), `prompt_tokens: int`, and
`generated_tokens: int`.

`VLLMGenerator.unload` must:
1. `del self.llm`
2. `torch.cuda.empty_cache()`
3. `gc.collect()`
4. `vllm.distributed.parallel_state.destroy_model_parallel()` (avoids a
   well-known vLLM leak when recreating engines in the same process).

`HFGenerator` uses `AutoModelForCausalLM.from_pretrained(...,
torch_dtype=torch.bfloat16)` + `generation_config.cache_config =
QuantizedCacheConfig(backend="HQQ", nbits=quant_config.nbits)`.

### 7.2 FDP hybrid finder

Inputs: `baseline_tokens`, `quant_tokens` (same tokenizer — same model),
`baseline_text`, `quant_text`, `params: FDPParams`.

Algorithm:

1. Walk both token lists in lock-step; find first index `i0` where tokens
   differ.
2. If no `i0` found, return `FDPRecord(fdp_token_idx=None)`.
3. Re-sync check:
   a. Decode windows `[i0:i0+params.resync_lookahead]` in both traces.
   b. Split each into "reasoning units" by `\n\n` and transition markers
      (`Let me`, `Wait,`, `So,`, `Therefore`, step numbers).
   c. For the next 3 units in each trace, compute MiniLM cosine similarity
      between corresponding units.
   d. If ≥2 of 3 cosines exceed `params.cosmetic_cosine_threshold` (0.9),
      classify as cosmetic divergence:
      - find the first token position where traces re-align,
      - increment `cosmetic_skipped`,
      - recurse into step 1 starting from the re-align point.
   e. Otherwise this is a real FDP.
4. Stop after `params.max_cosmetic_skips` (default 5); beyond that we
   treat the first mismatch as the real FDP.
5. Return `FDPRecord`:
   - `fdp_token_idx: int | None`
   - `cosmetic_skipped: int`
   - `baseline_context: str` — decoded slice `[i0 - ctx : i0 + ctx]`,
     default ctx=200 tokens
   - `quant_context: str` — same
   - `common_prefix: str` — decoded `[:i0]`, tail-trimmed to 300 tokens
   - `boxed_match: Literal["both_correct","baseline_only","quant_only","both_wrong","no_boxed"]`
   - `baseline_truncated: bool`, `quant_truncated: bool`
   - `full_traces_href: dict` — HF dataset pointer, not inline (keeps
     JSONL small)

Explicitly tested edge cases: identical traces; first-token divergence;
cosmetic-only divergence; real divergence after 1–3 cosmetic skips;
baseline truncated before divergence; FDP in first 10 tokens (left-clip);
re-sync never happens within lookahead.

### 7.3 Judge prompt + taxonomy

Taxonomy (fixed, versioned as `TAXONOMY_V1`):

- **A. Arithmetic** — numerical or symbolic computation error; logic is
  correct, the arithmetic is not.
- **B. Logical** — correct premises, incorrect inference step.
- **C. Strategy-switch** — unmotivated abandonment of the current
  approach for a different one.
- **D. Hallucination** — invention of an irrelevant or nonexistent fact
  (a fake theorem, a wrong formula).
- **E. Premature-termination** — trace cuts off before producing a final
  boxed answer; includes `finish_reason=length` and giving-up text.
- **F. Repetition/loop** — the same reasoning step (paragraph or
  equation) repeated three or more times.

Each category in `taxonomy.py` carries a definition plus 2–3 one-line
micro-examples to anchor the judge.

Prompt structure (Anthropic SDK):

```
system + [CACHED BLOCK: system instruction + taxonomy + JSON schema]
                                    ↑ cache_control: {"type":"ephemeral"}
user  + [PER-REQUEST: problem, ground truth, baseline window, quant window]
```

Response schema (pydantic-validated):

```json
{
  "category": "A|B|C|D|E|F",
  "confidence": 0.0-1.0,
  "rationale": "up to 2 sentences",
  "affected_span": "3-15 word quote from the quantized trace"
}
```

Settings: `model="claude-sonnet-4-6"`, `temperature=0.0`, `max_tokens=400`.

On-disk cache at `.cache/judge/{sha256}.json` keyed by the full prompt
string; re-runs of Phase 3 are free.

Calibration: `golden_set.py` contains 10 curated `(trace_pair,
gold_category)` examples covering each category. `test_judge_calibration`
(marked `live_api`) requires ≥7/10 agreement; it is mandatory before any
production judge run.

### 7.4 HF Hub integration

One public dataset repo per study run, id read from
`HF_REPO_ID` env var (default: `{HF_USER}/kv-trace-study`). The user sets
`HF_USER` before running `run_all.sh`.

Each phase writes its outputs to a distinct dataset **revision tag**:
`traces-{model}-{config}`, `fdps-{model}-{config}`,
`judgments-{model}-{config}`. Re-upload with the same tag is a no-op
(idempotency gate).

If `HF_USER` is not set, everything still works locally — HF upload/download
is silently skipped and phases read from `outputs/` on disk.

### 7.5 Analysis

`signatures.py` builds a `[n_methods × 6]` matrix of judgment counts per
(quant_config, category). From there:

- Row-normalized signature per method (rows sum to 1).
- Chi-square test of independence (H0: method does not affect category
  distribution); reports p-value.
- Pairwise Cramér's V matrix between methods.
- Per-model break-down (do 1.5B, 1.7B, 7B have the same signatures, or
  does scale change the failure mode?).
- Correlation between `(boxed_match, cosmetic_skipped)` and category.

`report.py` emits:
- `outputs/report.md` — tables and inline plot references.
- `outputs/report.json` — full results machine-readable.
- `outputs/plots/*.png` — heatmap of signatures, stacked bars per model,
  chi-square summary.

Reports are deterministic: same inputs → identical bytes, so diffs are
meaningful in git.

## 8. Testing strategy

Three pytest markers:

| Marker | When it runs | Requirements |
|---|---|---|
| (unmarked) | CI and local default | CPU, no network, <2 min total |
| `@pytest.mark.gpu` | manually before Phase 1 | CUDA |
| `@pytest.mark.live_api` | manually before Phase 3 | `ANTHROPIC_API_KEY` |

CI invocation: `pytest -m "not gpu and not live_api" --cov=src/kvtrace
--cov-fail-under=85`.

Every one of the four task areas from the research spec has at least one
gating test:

- **Task 1 (trace collection):** `test_generators_vllm.py`,
  `test_generators_hf.py`, `test_generator_contract.py`,
  `test_trace_utils.py`, `test_dataset_loader.py`,
  `test_vllm_smoke.py` (gpu).
- **Task 2 (FDP):** `test_fdp_finder.py` (7 edge cases),
  `test_tokenizer_align.py`, `test_semantic_resync.py`.
- **Task 3 (LLM-as-judge):** `test_taxonomy.py`, `test_judge_prompt.py`,
  `test_judge_response_parsing.py`, `test_judge_cache.py`,
  `test_judge_claude_mock.py`, `test_judge_calibration.py` (live_api).
- **Task 4 (signatures):** `test_signatures.py`, `test_report.py`.

Cross-cutting: `test_memory_free.py` (gpu), `test_hf_hub_upload.py`
(mocked), `test_end_to_end_smoke.py` (fully mocked, full pipeline, <30 s;
this is the top-level CI gate).

TDD loop per ТЗ: write test → red → minimal implementation → green. When
a test fails, the code is rewritten, not the test (except if the failure
reveals a bug in the test itself, which must be flagged explicitly).

## 9. Reproducibility

- All CLI scripts accept `--seed 42` passed to vLLM engine,
  `SamplingParams`, `torch.manual_seed`, and dataset shuffling (when
  shuffling is enabled — defaults to off).
- Deterministic decoding: `temperature=0.0`, `top_p=1.0`.
- Chat templates pulled verbatim from each model's tokenizer; locked to
  the model's HF revision tag.
- `requirements.txt` pins exact versions where behavior is API-sensitive:
  `transformers==4.45.0`, `vllm==0.6.6`, `anthropic==0.40.0`.
- Judge prompt is versioned (`TAXONOMY_V1`, `PROMPT_V1`); changes produce
  new SHA256 cache keys so stale judgments are never silently reused.
- Each JSONL record is self-describing: model, config, seed, timestamp,
  prompt version are stamped per row.

## 10. Risk register and fallbacks

| # | Risk | Mitigation |
|---|---|---|
| R1 | 7B + HQQ INT2 OOMs on 24 GB at long context | `max_model_len` override to 16 k for HQQ-only configs. `test_hqq_int2_7b_fits` (gpu) probes this before the real run. |
| R2 | `QuantizedCacheConfig` changes across transformers versions | Pin `transformers==4.45.0`; CI test asserts the import. Fallback: swap HQQ backend → `quanto`. |
| R3 | Some (model, config) pairs produce mostly garbage traces (e.g. `Qwen3-1.7B + hqq_int2`) | Don't pre-skip — garbage traces are themselves diagnostic signal (likely category F). Report observes them explicitly. |
| R4 | Anthropic rate-limits on 960 sequential calls | tenacity exponential backoff; max 5 concurrent; SHA256 cache makes retries free. |
| R5 | Vast.ai instance death mid-run | Every phase reads from HF or disk, writes idempotently. `run_all.sh --resume` picks up from the last completed revision. |

Budget-cut order if burn exceeds $15 (from least to most painful):

1. Drop `hqq_int2` config (~$1.5 GPU + $0.6 judge saved).
2. Cut MATH-500 from 50 to 30 problems (~$1.5 saved).
3. Drop 7B anchor model (~$3 saved, loses scale argument).
4. Switch judge from Sonnet 4.6 to Haiku 4.5 (~$2 saved; only if
   `test_judge_calibration` still passes at ≥7/10).

The first two are pre-baked as `config/pipeline.yaml.light` for fast
iteration.

## 11. Deliverables

After `bash scripts/run_all.sh` completes on a fresh 4090:

- `outputs/report.md` — 6 tables, 4 plots, all numbers the paper needs.
- `outputs/report.json` — machine-readable version of the same.
- Public HF dataset `{HF_USER}/kv-trace-study` with 15 trace JSONL + 12
  FDP JSONL + 12 judgment JSONL, each under its own revision tag.
- `.cache/judge/` — complete set of cached judgments, making re-analysis
  with new slicing free.
- Test suite green on CPU CI (≥85 % coverage).

## 12. Open items requiring the user at runtime

- `HF_USER` environment variable (HuggingFace username for dataset
  ownership). Falls back to local-only mode if unset.
- `ANTHROPIC_API_KEY` environment variable, needed for Phase 3.
- Vast.ai account with ≥ $15 of credits; user provisions the 4090
  instance and runs `bash scripts/run_all.sh`.

None of these block spec or plan approval.
