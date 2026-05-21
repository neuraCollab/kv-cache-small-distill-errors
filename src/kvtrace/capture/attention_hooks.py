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
    # Tracks last-seen cache seq-length per layer (HF layout: seq at dim -2).
    # Used to capture ONLY the new positions added in this call — keeps
    # K_pre clean (un-quantized) per AR step.
    _prev_cache_sizes: dict[int, int] = field(default_factory=dict)

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
    # Append mode: lists grow with each hook invocation.
    # For a single forward pass with N layers, each list ends up with N entries
    # (one per layer) — same indexing as before.
    # For AR multi-step generation, lists grow by N per step, and
    # handle.q[i::n_layers] gives all captures for layer i across steps.

    def _make_q_hook():
        def _hook(module, inputs, output):
            # output of q_proj is [bsz, seq, hidden] — захватываем как есть
            handle.q.append(output.detach().clone())
        return _hook

    def _make_attn_hook(layer_idx: int):
        def _hook(module, inputs, kwargs, outputs):
            # past_key_value приходит как kwarg в HF Qwen3 и в FakeAttention/
            # FakeHFAttention. Fallback на inputs[1] — для позиционного вызова.
            pkv = (
                kwargs.get("past_key_value")
                or kwargs.get("past_key_values")
                or (inputs[1] if len(inputs) > 1 else None)
            )

            # Q-extraction from outputs[1] tuple — для FakeAttention path.
            # Если q_proj hook уже захватил Q для этого call'а
            # (len(handle.q) > len(handle.k_pre)), не дублируем.
            if (
                isinstance(outputs, tuple)
                and len(outputs) == 2
                and isinstance(outputs[1], tuple)
                and len(outputs[1]) == 3
            ):
                q_tup = outputs[1][0]
                if len(handle.q) <= len(handle.k_pre):
                    handle.q.append(q_tup.detach().clone())

            if pkv is None or not hasattr(pkv, "key_cache") or layer_idx >= len(pkv.key_cache):
                # Нет cache → soft-skip K/V если Q уже захвачен (Q-only scenario).
                if len(handle.q) > len(handle.k_pre):
                    return
                raise RuntimeError(
                    f"Layer {layer_idx}: no Q/K/V source. "
                    f"output type={type(outputs)}, pkv={pkv}"
                )

            # HF cache layout: [B, num_kv_heads, seq, head_dim] — seq at dim -2.
            # Захватываем ТОЛЬКО новые позиции этого call'а, чтобы K_pre
            # для старых позиций оставался чистым bf16 (не переквантованным).
            k_cache = pkv.key_cache[layer_idx]
            v_cache = pkv.value_cache[layer_idx]
            current_seq = k_cache.shape[-2]
            prev_seq = handle._prev_cache_sizes.get(layer_idx, 0)

            if current_seq <= prev_seq:
                # Странно — cache не вырос. Пропускаем, чтобы не падать на assertion.
                return

            # Slice new positions: [..., prev_seq:current_seq, :]
            k_pre_new = k_cache[..., prev_seq:current_seq, :].detach().clone()
            v_pre_new = v_cache[..., prev_seq:current_seq, :].detach().clone()
            handle.k_pre.append(k_pre_new)
            handle.v_pre.append(v_pre_new)

            k_q = quant_fn(k_pre_new)
            v_q = quant_fn(v_pre_new)
            handle.k_post.append(k_q.detach().clone())
            handle.v_post.append(v_q.detach().clone())

            # Write quantized values BACK to cache, ONLY at the new positions.
            # Старые позиции уже квантованы предыдущими вызовами, не трогаем.
            k_cache[..., prev_seq:current_seq, :] = k_q
            v_cache[..., prev_seq:current_seq, :] = v_q

            handle._prev_cache_sizes[layer_idx] = current_seq

        return _hook

    for layer_idx, mod in enumerate(attention_modules):
        if hasattr(mod, "q_proj"):
            h = mod.q_proj.register_forward_hook(_make_q_hook())
            handle._hook_handles.append(h)
        # with_kwargs=True needed to capture `past_key_value` (HF passes it as kwarg).
        h = mod.register_forward_hook(_make_attn_hook(layer_idx), with_kwargs=True)
        handle._hook_handles.append(h)

    return handle
