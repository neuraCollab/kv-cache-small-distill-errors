#!/usr/bin/env bash
#
# Usage:
#   bash scripts/run_all.sh [--light]
#
# --light  use a reduced config (see README).
#
# Assumes: ANTHROPIC_API_KEY and optionally HF_USER/HF_REPO_ID are set.

set -euo pipefail

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

echo "=== Phase 1: generate traces ==="
for model in "${MODELS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    echo "--- model=$model config=$cfg ---"
    python scripts/01_generate_traces.py --model "$model" --config "$cfg" --resume
  done
done

echo "=== Phase 2: find FDPs ==="
for model in "${MODELS[@]}"; do
  python scripts/02_find_fdps.py --model "$model"
done

echo "=== Phase 3: judge FDPs ==="
python scripts/03_judge_fdps.py

echo "=== Phase 4: analyze ==="
python scripts/04_analyze.py

echo "[run_all] done. See outputs/report.md"
