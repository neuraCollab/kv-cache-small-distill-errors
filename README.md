# KV Cache Quantization — Trace-Level Diagnostic Study

Reproducible pipeline for diagnosing **how** KV cache quantization methods
break reasoning in compact (1.5B–7B) reasoning models, not just how much
they degrade accuracy. Output is a markdown + JSON report with failure
**signatures** for 5 quantization configurations across 3 models.

## Design documents

- Design spec: [docs/superpowers/specs/2026-04-25-kv-trace-study-design.md](docs/superpowers/specs/2026-04-25-kv-trace-study-design.md)
- Implementation plan: [docs/superpowers/plans/2026-04-25-kv-trace-study.md](docs/superpowers/plans/2026-04-25-kv-trace-study.md)

## What it does

Four idempotent phases:

1. **GENERATE** — run each of 3 models × 5 KV-cache configurations × 80
   math problems (30 AIME-24 + 50 MATH-500) through its own generator
   (vLLM for BF16 / FP8 E5M2 / FP8 E4M3, HuggingFace Transformers + HQQ
   for INT4 / INT2).
2. **FIND FDP** — for each quantized trace, locate the First Divergence
   Point from the BF16 baseline using token-level exact matching plus a
   MiniLM semantic re-sync filter to skip cosmetic divergences.
3. **JUDGE** — ask Claude Sonnet 4.6 (with Anthropic prompt caching and a
   local SHA256 prompt cache) to classify each FDP into one of six error
   categories: A-Arithmetic, B-Logical, C-Strategy-switch,
   D-Hallucination, E-Premature-termination, F-Repetition/loop.
4. **ANALYZE** — build a confusion matrix (method × category), run a
   chi-square independence test and Cramér's V, and emit a deterministic
   markdown + JSON report plus heatmap plots.

Each phase is resumable from HuggingFace Hub snapshots, so a Vast.ai
instance death in the middle of the run is cheap to recover from.

## Hardware

- **Single RTX 4090 (24 GB, Ada, CC 8.9)** on Vast.ai (~$0.40/hour).
  Ada is chosen over Ampere because it has native FP8 tensor cores,
  making `fp8_e5m2` and `fp8_e4m3` qualitatively distinct — this is
  essential for the "different methods break differently" thesis.
- ~14 GPU-hours total for the full 3-model × 5-config × 80-problem run.

## Budget

| Line item | Cost |
|---|---|
| Compute (Vast.ai, ~14 GPU-hours) | ~$5.60 |
| Claude Sonnet 4.6 judge (prompt cached) | ~$3.00 |
| HuggingFace Hub (public dataset) | $0 |
| Contingency | ~$3.00 |
| **Total** | **~$12** |

## Install

```bash
git clone <this repo>
cd kv-trace-study

python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Dev dependencies (tests, linters):

```bash
pip install -r requirements-dev.txt
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | for Phase 3 | Claude Sonnet 4.6 judge |
| `HF_REPO_ID` | optional | full HF dataset repo id (e.g. `me/my-kv-study`) |
| `HF_USER` | optional | HF username; dataset is `{HF_USER}/kv-trace-study` |
| `HF_TOKEN` | for HF upload | write access to the dataset repo |

If no HF variable is set the pipeline still runs locally — uploads are
silently skipped.

## Run the full study

```bash
# Set at least ANTHROPIC_API_KEY; HF_USER is strongly recommended.
export ANTHROPIC_API_KEY="sk-ant-..."
export HF_USER="your-hf-username"
export HF_TOKEN="hf_..."

# Calibrate the judge FIRST (takes ~30s on live API).
# Must pass ≥7/10 — if it fails, re-check the taxonomy prompt.
pytest -m live_api tests/test_judge_calibration.py -v

# Now the real thing.
bash scripts/run_all.sh

# Output:
#   outputs/report.md
#   outputs/report.json
#   outputs/plots/*.png
#   HF:  {HF_USER}/kv-trace-study with 15 trace + 12 FDP + 12 judgment revisions
```

For a faster iteration run that drops the most experimental config:

```bash
bash scripts/run_all.sh --light
```

## Run individual phases

```bash
# Phase 1 — one (model, config) at a time, resumable
python scripts/01_generate_traces.py \
    --model deepseek-r1-distill-qwen-1.5b \
    --config fp8_e5m2 \
    --resume

# Phase 2 — needs baseline bf16 already generated
python scripts/02_find_fdps.py --model deepseek-r1-distill-qwen-1.5b

# Phase 3 — all FDPs at once; cached
python scripts/03_judge_fdps.py

# Phase 4 — CPU only, <1 min
python scripts/04_analyze.py
```

## Testing

```bash
# CI default — no GPU, no live API, ≥85% coverage gate
make test

# GPU-dependent tests (run on the rented 4090 before Phase 1)
make test-gpu

# Live-API calibration (run before Phase 3)
make test-live
```

Three pytest markers:

| Marker | When to run |
|---|---|
| (none) | always; CI default |
| `@pytest.mark.gpu` | before renting GPU time |
| `@pytest.mark.live_api` | before each Phase 3 run (catches Anthropic drift) |

## Repository layout

```
kv-trace-study/
├── config/                   # 3 YAML files — models, quant methods, pipeline
├── src/kvtrace/
│   ├── generators/           # vLLM + HF (HQQ) backends behind one ABC
│   ├── fdp/                  # hybrid token + semantic re-sync finder
│   ├── judge/                # taxonomy, prompt, Claude client, golden set
│   ├── hf_hub/               # idempotent upload / download
│   └── analysis/             # signatures + markdown report
├── scripts/                  # 01…04 phase CLIs + run_all.sh
├── tests/                    # CPU, GPU, and live-API suites
└── outputs/                  # runtime artifacts (gitignored)
```

## Reproducibility

- Greedy decoding (`temperature=0.0`, `top_p=1.0`) + fixed `seed=42`.
- Chat templates taken verbatim from each model's HF tokenizer.
- `requirements.txt` pins `vllm==0.6.6`, `transformers==4.45.0`,
  `anthropic==0.40.0`.
- Prompt and taxonomy are versioned (`PROMPT_V1`, `TAXONOMY_V1`); a
  change bumps the SHA256 cache key, so stale judgments never silently
  reappear.
- Each JSONL row is self-describing — model, config, seed, timestamp,
  prompt version.

## Citation

The paper is not in this repository. The dataset and code here are the
*reproducibility appendix*: cite them via the HuggingFace dataset id and
the GitHub commit hash.
