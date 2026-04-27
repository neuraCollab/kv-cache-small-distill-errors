#!/usr/bin/env bash
#
# Usage:
#   bash scripts/run_all.sh [--light]
#
# --light  use a reduced config (see README).
#
# Assumes: ANTHROPIC_API_KEY and optionally HF_USER/HF_REPO_ID are set.
#
# Resilience: a single (model, config) pair that crashes — e.g. because
# transformers / vLLM doesn't recognise an architecture, or the GPU OOMs —
# logs the failure and continues to the next pair. Phase 2-4 then runs
# only over models that actually produced trace files. Without this, one
# bad config would abort the whole 8h+ run.

# NB: no `set -e`. We swallow per-config failures intentionally; structural
# bugs in this orchestrator surface via empty `outputs/traces/`.
set -uo pipefail

MODELS=(
  deepseek-r1-distill-qwen-1.5b
  qwen3-1.7b
  deepseek-r1-distill-qwen-7b
)

CONFIGS=(
  bf16
  fp8_e5m2
  fp8_e4m3
  hqq_int4
  hqq_int2
)

if [[ "${1:-}" == "--light" ]]; then
  CONFIGS=(bf16 fp8_e5m2 fp8_e4m3 hqq_int4)
  echo "[run_all] --light mode: dropping hqq_int2"
fi

mkdir -p outputs/traces logs
FAIL_LOG="logs/run_all_failures.log"
: > "$FAIL_LOG"

declare -a SUCCEEDED_MODELS=()
declare -A MODEL_HAS_TRACE=()

echo "=== Phase 1: generate traces ==="
for model in "${MODELS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    echo "--- model=$model config=$cfg ---"
    if python scripts/01_generate_traces.py --model "$model" --config "$cfg" --resume; then
      MODEL_HAS_TRACE["$model"]=1
    else
      rc=$?
      echo "[run_all] FAILED model=$model config=$cfg rc=$rc — continuing" | tee -a "$FAIL_LOG"
    fi
  done
done

# Build the set of models with at least one successful trace file. Phase 2
# scans `outputs/traces/{model}_*.jsonl`, so if a model has zero files
# `02_find_fdps.py` would either crash or silently emit empty pair stats.
for model in "${MODELS[@]}"; do
  if compgen -G "outputs/traces/${model}_*.jsonl" > /dev/null; then
    SUCCEEDED_MODELS+=("$model")
  else
    echo "[run_all] SKIP model=$model in Phase 2-4 (no trace files)" | tee -a "$FAIL_LOG"
  fi
done

if [[ ${#SUCCEEDED_MODELS[@]} -eq 0 ]]; then
  echo "[run_all] ABORT: no model produced any traces. See $FAIL_LOG."
  exit 1
fi

echo "=== Phase 2: find FDPs ==="
for model in "${SUCCEEDED_MODELS[@]}"; do
  python scripts/02_find_fdps.py --model "$model" \
    || echo "[run_all] FAILED phase2 model=$model — continuing" | tee -a "$FAIL_LOG"
done

echo "=== Phase 3: judge FDPs ==="
python scripts/03_judge_fdps.py \
  || echo "[run_all] FAILED phase3 — continuing" | tee -a "$FAIL_LOG"

echo "=== Phase 4: analyze ==="
python scripts/04_analyze.py \
  || echo "[run_all] FAILED phase4" | tee -a "$FAIL_LOG"

if [[ -s "$FAIL_LOG" ]]; then
  echo "[run_all] done with failures. See $FAIL_LOG and outputs/report.md"
  exit 0  # not a hard failure: the partial output is still useful
fi
echo "[run_all] done. See outputs/report.md"
