"""Unit-test для CaptureRunner.run_ar — autoregressive с KV-quant в loop."""
from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.window import Window


class _ARCache:
    """KV-cache that correctly concatenates along the seq dim (dim=1 for 4D tensors)."""

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            seq_dim = 1  # tensors are [B, seq, heads, dim]
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=seq_dim
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=seq_dim
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def _make_ar_runner(n_layers=1, hidden=8, vocab=16):
    """Минимальная модель + generate-метод для AR-теста."""
    from tests.test_capture_hooks import _FakeHFAttention

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_FakeHFAttention(i, hidden) for i in range(n_layers)])
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            self.config = MagicMock(num_hidden_layers=n_layers, vocab_size=vocab)

        def generate(self, input_ids, max_new_tokens, do_sample, temperature,
                     repetition_penalty, return_dict_in_generate, output_scores, **kw):
            cache = _ARCache()
            generated_logits = []
            ids = input_ids
            # First call: prefill with full prefix
            cur_input = ids
            for step in range(max_new_tokens):
                h = nn.functional.one_hot(cur_input % hidden, num_classes=hidden).float()
                for layer in self.layers:
                    h, _ = layer(h, past_key_value=cache)
                step_logits = self.lm_head(h[:, -1:])
                generated_logits.append(step_logits)
                next_id = step_logits.argmax(dim=-1)
                ids = torch.cat([ids, next_id], dim=-1)
                # Subsequent steps: only feed the new token
                cur_input = next_id
            out = MagicMock()
            out.sequences = ids
            out.scores = tuple(t.squeeze(1) for t in generated_logits)
            return out

    runner = CaptureRunner.__new__(CaptureRunner)
    runner._model = _Wrapper()
    runner._tokenizer = MagicMock(eos_token_id=999)
    runner._attention_modules = list(runner._model.layers)
    runner._n_layers = n_layers
    runner._vocab = vocab
    runner._model_revision_hash = "test"
    return runner


def test_run_ar_generates_expected_step_count():
    runner = _make_ar_runner(n_layers=1, hidden=8, vocab=16)
    prefix_token_ids = [1, 2, 3, 4, 5]
    window = Window(ws=3, we=8, truncated_left=False, truncated_right=False)

    cap = runner.run_ar(
        prefix_token_ids=prefix_token_ids,
        window=window,
        quant="bf16",
        problem_id=7,
        fdp_token_idx=5,
        max_new_tokens=3,
    )

    assert cap.meta["mode"] == "ar"
    assert cap.meta["problem_id"] == 7
    assert cap.meta["W"] == window.size
    # gen_token_ids — конкатенация последних prefix-позиций + сгенерированных
    assert len(cap.meta["gen_token_ids"]) == window.size
