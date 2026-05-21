"""Forward-хук installer для захвата Q/K_pre/K_post/V_pre/V_post.

Подмена K/V в кеше реализована через monkey-patch метода
`past_key_value.update(...)` ровно в момент attention forward. Это
работает и для HF DynamicCache, и для нашего FakeCache в тестах.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class CaptureHandle:
    """Контейнер захваченных тензоров + метод снятия хуков."""
    q: list[torch.Tensor] = field(default_factory=list)
    k_pre: list[torch.Tensor] = field(default_factory=list)
    v_pre: list[torch.Tensor] = field(default_factory=list)
    k_post: list[torch.Tensor] = field(default_factory=list)
    v_post: list[torch.Tensor] = field(default_factory=list)
    _hook_handles: list[Any] = field(default_factory=list)

    def remove(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def install_capture_hooks(
    model: nn.Module,
    attention_modules: list[nn.Module],
    quant_fn: Callable[[torch.Tensor], torch.Tensor],
) -> CaptureHandle:
    """Установить forward hooks на каждый attention-блок + q_proj.

    Стратегия:
      - Forward-hook на attention block: захватывает K/V из past_key_value
        и подменяет на quant-версию
      - Forward-hook на module.q_proj (если есть): захватывает Q post-projection
      - Если q_proj нет (наша FakeAttention из Task 5): берёт Q из
        outputs[1] tuple
    """
    handle = CaptureHandle()
    # Pre-allocate slots per layer so order is deterministic
    n_layers = len(attention_modules)
    handle.q = [None] * n_layers  # type: ignore[list-item]
    handle.k_pre = [None] * n_layers  # type: ignore[list-item]
    handle.v_pre = [None] * n_layers  # type: ignore[list-item]
    handle.k_post = [None] * n_layers  # type: ignore[list-item]
    handle.v_post = [None] * n_layers  # type: ignore[list-item]

    def _make_q_hook(layer_idx: int):
        def _hook(module, inputs, output):
            # output of q_proj is [bsz, seq, hidden] — захватываем как есть
            handle.q[layer_idx] = output.detach().clone()
        return _hook

    def _make_attn_hook(layer_idx: int):
        def _hook(module, inputs, outputs):
            pkv = inputs[1] if len(inputs) > 1 else None
            from_outputs = False

            if (
                isinstance(outputs, tuple)
                and len(outputs) == 2
                and isinstance(outputs[1], tuple)
                and len(outputs[1]) == 3
            ):
                q_tup, k_src, v_src = outputs[1]
                if handle.q[layer_idx] is None:
                    handle.q[layer_idx] = q_tup.detach().clone()
                from_outputs = True
            elif pkv is not None and hasattr(pkv, "key_cache"):
                k_src = pkv.key_cache[layer_idx]
                v_src = pkv.value_cache[layer_idx]
            else:
                # No K/V source found. If Q was already captured via q_proj hook,
                # skip K/V capture silently (Q-only scenario).
                if handle.q[layer_idx] is not None:
                    return
                raise RuntimeError(
                    f"Layer {layer_idx}: no Q/K/V source. "
                    f"output type={type(outputs)}, pkv={pkv}"
                )

            handle.k_pre[layer_idx] = k_src.detach().clone()
            handle.v_pre[layer_idx] = v_src.detach().clone()

            k_q = quant_fn(k_src)
            v_q = quant_fn(v_src)
            handle.k_post[layer_idx] = k_q.detach().clone()
            handle.v_post[layer_idx] = v_q.detach().clone()

            # Replace cache entries.
            # When k/v came from outputs tuple, k IS cache.key_cache[layer_idx]
            # (same object returned by cache.update). Mutate in-place so the
            # cache sees quantized values without needing a pkv reference.
            if from_outputs:
                k_src.copy_(k_q)
                v_src.copy_(v_q)
            elif pkv is not None and hasattr(pkv, "key_cache") and layer_idx < len(pkv.key_cache):
                pkv.key_cache[layer_idx] = k_q
                pkv.value_cache[layer_idx] = v_q

        return _hook

    for layer_idx, mod in enumerate(attention_modules):
        if hasattr(mod, "q_proj"):
            h = mod.q_proj.register_forward_hook(_make_q_hook(layer_idx))
            handle._hook_handles.append(h)
        h = mod.register_forward_hook(_make_attn_hook(layer_idx))
        handle._hook_handles.append(h)

    return handle
