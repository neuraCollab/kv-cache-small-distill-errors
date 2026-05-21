"""End-to-end smoke: 1 задача × bf16 × tf на реальной Qwen3-1.7B (CPU).

Помечен @pytest.mark.slow — в CI skipped (потребует загрузки ~3GB модели).
Запускается локально перед release: `pytest tests/test_capture_smoke.py -m slow -v`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.storage import load_capture, save_capture
from kvtrace.capture.window import compute_window


@pytest.mark.slow
def test_smoke_qwen3_1_7b_bf16_tf(tmp_path: Path):
    runner = CaptureRunner()
    runner.load_model("Qwen/Qwen3-1.7B")

    fake_input = list(range(50)) + [100, 200, 300]  # mock 53 tokens
    window = compute_window(fdp_idx=40, trace_len=len(fake_input), pre=10, post=10)
    cap = runner.run_tf(
        input_token_ids=fake_input,
        window=window,
        quant="bf16",
        problem_id=0,
        fdp_token_idx=40,
    )
    out = tmp_path / "smoke.safetensors"
    save_capture(cap, out)
    loaded = load_capture(out)
    assert loaded.meta["model"] == "qwen3-1.7b"
    assert loaded.q[0].shape[0] == window.size  # W positions
    assert len(loaded.q) == 28  # Qwen3-1.7B has 28 layers
    runner.unload()
