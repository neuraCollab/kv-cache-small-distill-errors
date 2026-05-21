"""Проверка, что хуки корректно работают когда кеш ≥ sliding_window."""
from __future__ import annotations

import pytest
import torch

from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.fp8_sim import fp8_e4m3
from kvtrace.capture.window import Window


@pytest.mark.slow
def test_hooks_work_with_long_context():
    runner = CaptureRunner()
    runner.load_model("Qwen/Qwen3-1.7B")

    # Qwen3-1.7B sliding_window обычно ≥ 4096; даём 5000 токенов чтобы
    # гарантированно перейти в sliding режим.
    long_input = list(range(5000))
    window = Window(ws=4500, we=4600, truncated_left=False, truncated_right=False)

    cap = runner.run_tf(
        input_token_ids=long_input,
        window=window,
        quant="fp8_e4m3",
        problem_id=0,
        fdp_token_idx=4550,
    )
    # Все 28 слоёв захвачены
    assert len(cap.q) == 28
    # Размер окна совпадает
    assert cap.q[0].shape[0] == window.size
    # K_post должен быть равен fp8_e4m3(K_pre) elementwise
    assert torch.equal(cap.k_post[0], fp8_e4m3(cap.k_pre[0]))
    runner.unload()
