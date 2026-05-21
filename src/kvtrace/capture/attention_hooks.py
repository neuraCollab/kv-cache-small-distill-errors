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
    """Установить forward hooks на каждый attention-блок.

    Hook берёт outputs = (attn_output, (q, k, v)) из attention forward и:
      - пишет q, k, v в CaptureHandle как pre-quant
      - вычисляет k_post = quant_fn(k), v_post = quant_fn(v)
      - подменяет K/V в кеше (если кеш был передан) на quant-версию

    Note: для HF Qwen3 attention.forward возвращает (attn_output, attn_weights);
    K/V кладутся в cache через past_key_value.update(...) внутри forward.
    Мы вытаскиваем K/V из кеша после forward и квантуем их там же.
    """
    handle = CaptureHandle()

    def _make_hook(layer_idx: int):
        def _hook(module, inputs, outputs):
            # outputs format depends on attention class.
            # Our FakeAttention returns (attn_out, (q, k, v)).
            # HF Qwen3Attention returns (attn_out, attn_weights) but K/V are
            # in past_key_value at module.layer_idx.
            from_outputs = False
            if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], tuple):
                _, (q, k, v) = outputs
                from_outputs = True
            else:
                # HF path — pull from cache (passed via kwargs)
                pkv = inputs[1] if len(inputs) > 1 else None
                if pkv is None or not hasattr(pkv, "key_cache"):
                    raise RuntimeError(
                        f"Layer {layer_idx}: cannot locate Q/K/V. "
                        f"Output type {type(outputs)}, no usable past_key_value."
                    )
                k = pkv.key_cache[layer_idx]
                v = pkv.value_cache[layer_idx]
                q = None  # placeholder for HF path; q_proj hook добавляется в Task 6

            handle.q.append(q if q is not None else torch.empty(0))
            handle.k_pre.append(k.detach().clone())
            handle.v_pre.append(v.detach().clone())

            k_q = quant_fn(k)
            v_q = quant_fn(v)
            handle.k_post.append(k_q.detach().clone())
            handle.v_post.append(v_q.detach().clone())

            # Replace cache entries in-place.
            # When k/v came from outputs tuple, k IS cache.key_cache[layer_idx]
            # (same object returned by cache.update). Mutate in-place so the
            # cache sees quantized values without needing a pkv reference.
            if from_outputs:
                k.copy_(k_q)
                v.copy_(v_q)
            else:
                pkv = inputs[1] if len(inputs) > 1 else None
                if pkv is not None and hasattr(pkv, "key_cache") and layer_idx < len(pkv.key_cache):
                    pkv.key_cache[layer_idx] = k_q
                    pkv.value_cache[layer_idx] = v_q

        return _hook

    for layer_idx, mod in enumerate(attention_modules):
        h = mod.register_forward_hook(_make_hook(layer_idx))
        handle._hook_handles.append(h)

    return handle
