"""Unit-test для CaptureRunner.run_tf на синтетической модели."""
from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.window import Window


def _make_synthetic_runner(n_layers=2, hidden=8, vocab=16):
    """Создаёт CaptureRunner с заранее загруженной фейковой моделью."""
    from tests.test_capture_hooks import _FakeHFAttention

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_FakeHFAttention(i, hidden) for i in range(n_layers)])
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            self.config = MagicMock(num_hidden_layers=n_layers, hidden_size=hidden)

        def __call__(self, input_ids=None, past_key_values=None, **kw):
            h = nn.functional.one_hot(input_ids % hidden, num_classes=hidden).float()
            for layer in self.layers:
                h, _ = layer(h, past_key_value=past_key_values)
            logits = self.lm_head(h)
            out = MagicMock()
            out.logits = logits
            return out

    runner = CaptureRunner.__new__(CaptureRunner)
    runner._model = _Wrapper()
    runner._tokenizer = MagicMock(eos_token_id=0)
    runner._attention_modules = list(runner._model.layers)
    runner._n_layers = n_layers
    runner._vocab = vocab
    runner._model_revision_hash = ""
    return runner


def test_run_tf_returns_capture_data_with_correct_shapes():
    runner = _make_synthetic_runner(n_layers=2, hidden=8, vocab=16)
    input_token_ids = list(range(10))  # 10 токенов
    window = Window(ws=0, we=10, truncated_left=False, truncated_right=False)

    cap = runner.run_tf(
        input_token_ids=input_token_ids,
        window=window,
        quant="bf16",
        problem_id=42,
        fdp_token_idx=5,
    )

    assert cap.meta["problem_id"] == 42
    assert cap.meta["quant"] == "bf16"
    assert cap.meta["mode"] == "tf"
    assert cap.meta["fdp_token_idx"] == 5
    assert cap.meta["W"] == 10
    assert len(cap.q) == 2
    assert cap.q[0].shape[0] == 10  # W positions
    assert cap.logits.shape == (10, 16)


def test_run_tf_slices_to_window():
    runner = _make_synthetic_runner(n_layers=1, hidden=8, vocab=16)
    input_token_ids = list(range(20))
    window = Window(ws=5, we=15, truncated_left=False, truncated_right=False)

    cap = runner.run_tf(
        input_token_ids=input_token_ids,
        window=window,
        quant="bf16",
        problem_id=0,
        fdp_token_idx=10,
    )

    assert cap.meta["window_start"] == 5
    assert cap.meta["window_end"] == 15
    assert cap.q[0].shape[0] == 10  # 15-5
    assert cap.logits.shape == (10, 16)
