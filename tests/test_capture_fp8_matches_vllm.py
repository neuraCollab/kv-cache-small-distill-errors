"""Golden-test: CPU FP8 simulator должен воспроизводить vLLM-trajectory
на первых N decode-шагах задачи #0 из outputs/traces/qwen3-1.7b_fp8_e4m3.jsonl.

Помечен @pytest.mark.slow — требует:
  - загрузки модели Qwen3-1.7B
  - существующих artefactов основного эксперимента в outputs/traces/

Логика:
  1. Реконструируем prompt через chat_template из problem-текста (трасса
     хранит только generated-tokens, не prompt).
  2. Прогоняем CPU модель с fp8_e4m3 quant на N_COMPARE decode-шагов.
  3. Сравниваем с первыми N_COMPARE generated-tokens из vLLM-трассы fp8_e4m3.

Почему N_COMPARE > 1: первый decode для reasoning Qwen3 это почти всегда
<think> (151667) — слишком тривиально. Берём 10, чтобы поймать
расхождения в нюансах квант-арифметики до того, как trajectory сойдёт
с FDP-точки (для Qwen3 FDP median ≈ позиция 540).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.window import Window

TRACES = Path("outputs/traces")
DEFAULT_USER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
N_COMPARE = 10  # сколько первых decode-tokens сверяем


@pytest.mark.slow
def test_cpu_fp8_e4m3_matches_vllm_first_decodes():
    fp8_trace_path = TRACES / "qwen3-1.7b_fp8_e4m3.jsonl"
    if not fp8_trace_path.exists():
        pytest.skip("Existing vLLM trace not available")

    fp8_first = json.loads(fp8_trace_path.read_text(encoding="utf-8").splitlines()[0])
    expected_first_decodes = fp8_first["token_ids"][:N_COMPARE]

    runner = CaptureRunner()
    runner.load_model("Qwen/Qwen3-1.7B")

    # Реконструируем prompt — трасса хранит problem-текст, но не сам prompt
    messages = [
        {"role": "user", "content": f"{fp8_first['problem']}\n\n{DEFAULT_USER_INSTRUCTION}"}
    ]
    prompt_ids = runner._tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
    )[0].tolist()
    prompt_len = len(prompt_ids)

    # AR с fp8_e4m3 на N_COMPARE шагов
    cap = runner.run_ar(
        prefix_token_ids=prompt_ids,
        window=Window(
            ws=prompt_len,
            we=prompt_len + N_COMPARE,
            truncated_left=False,
            truncated_right=False,
        ),
        quant="fp8_e4m3",
        problem_id=0,
        fdp_token_idx=prompt_len,
        max_new_tokens=N_COMPARE,
    )
    actual_first_decodes = cap.meta["gen_token_ids"]
    runner.unload()

    assert actual_first_decodes == expected_first_decodes, (
        f"CPU FP8 sim расходится с vLLM на первых {N_COMPARE} decode-токенах:\n"
        f"  CPU:  {actual_first_decodes}\n"
        f"  vLLM: {expected_first_decodes}\n"
        "Или симулятор арифметически неточен, или prompt реконструируется иначе чем в vLLM."
    )
