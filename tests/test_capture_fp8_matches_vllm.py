"""Golden-test: CPU FP8 simulator на первой задаче должен дать тот же
первый decode-токен, что vLLM в outputs/traces/qwen3-1.7b_fp8_e4m3.jsonl.

Помечен @pytest.mark.slow — требует загрузки модели И существующих
artefactов из основного эксперимента.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kvtrace.capture.cpu_runner import CaptureRunner

TRACES = Path("outputs/traces")


@pytest.mark.slow
def test_cpu_fp8_e4m3_matches_vllm_first_token():
    fp8_trace_path = TRACES / "qwen3-1.7b_fp8_e4m3.jsonl"
    bf16_trace_path = TRACES / "qwen3-1.7b_bf16.jsonl"
    if not fp8_trace_path.exists() or not bf16_trace_path.exists():
        pytest.skip("Existing vLLM traces not available")

    fp8_first = json.loads(fp8_trace_path.read_text().splitlines()[0])
    bf16_first = json.loads(bf16_trace_path.read_text().splitlines()[0])

    # bf16-trace prompt = первые num_prompt_tokens токенов
    prompt_len = bf16_first["num_prompt_tokens"]
    prompt_tokens = bf16_first["token_ids"][:prompt_len]
    expected_first_decode = fp8_first["token_ids"][prompt_len]

    runner = CaptureRunner()
    runner.load_model("Qwen/Qwen3-1.7B")

    # AR на 1 шаг с fp8_e4m3
    from kvtrace.capture.window import Window
    cap = runner.run_ar(
        prefix_token_ids=prompt_tokens,
        window=Window(ws=prompt_len, we=prompt_len + 1,
                      truncated_left=False, truncated_right=False),
        quant="fp8_e4m3",
        problem_id=0,
        fdp_token_idx=prompt_len,
        max_new_tokens=1,
    )
    actual_first_decode = cap.meta["gen_token_ids"][0]
    runner.unload()

    assert actual_first_decode == expected_first_decode, (
        f"CPU FP8 sim расходится с vLLM на первом decode-токене: "
        f"CPU={actual_first_decode} vs vLLM={expected_first_decode}. "
        "Симулятор арифметически неверен или хук подменяет K/V в неправильной точке."
    )
