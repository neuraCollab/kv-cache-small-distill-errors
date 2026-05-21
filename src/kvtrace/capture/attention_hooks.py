"""Forward-хуки + monkey-patch для захвата Q/K_pre/K_post/V_pre/V_post.

КРИТИЧНО: квантование K/V применяется через monkey-patch
`past_key_value.update(...)`, который вызывается ВНУТРИ attention.forward
ДО того как attention читает cache. Так модель действительно видит
квантованный K/V (предыдущая попытка через post-forward hook не работала
— attention уже использовал bf16 K/V к моменту срабатывания хука).

Схема хуков:
  1. forward_pre_hook на каждом attention блоке — устанавливает
     monkey-patch на pkv.update при первом срабатывании per cache instance.
  2. forward_hook на module.q_proj — захватывает Q post-projection,
     pre-RoPE (для HF Qwen3-style моделей).
  3. forward_hook на attention block — fallback для FakeAttention path,
     которая не имеет q_proj и кладёт Q в outputs[1] tuple.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class CaptureHandle:
    """Контейнер захваченных тензоров + метод снятия хуков и анпатча."""
    q: list[torch.Tensor] = field(default_factory=list)
    k_pre: list[torch.Tensor] = field(default_factory=list)
    v_pre: list[torch.Tensor] = field(default_factory=list)
    k_post: list[torch.Tensor] = field(default_factory=list)
    v_post: list[torch.Tensor] = field(default_factory=list)
    _hook_handles: list[Any] = field(default_factory=list)
    _patched_caches: list[Any] = field(default_factory=list)

    def remove(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        # Восстановить оригинальный update на запатченных классах cache.
        # Patching на КЛАССЕ (не на instance), потому что некоторые места
        # в transformers вызывают update через class-level access
        # (Cls.update(self, ...)), что обходит instance-attribute patch.
        for cache_cls in self._patched_caches:
            if hasattr(cache_cls, "_kv_capture_original_update"):
                cache_cls.update = cache_cls._kv_capture_original_update
                delattr(cache_cls, "_kv_capture_original_update")
                delattr(cache_cls, "_kv_capture_patched")
        self._patched_caches.clear()


def install_capture_hooks(
    model: nn.Module,
    attention_modules: list[nn.Module],
    quant_fn: Callable[[torch.Tensor], torch.Tensor],
) -> CaptureHandle:
    """Установить хуки для захвата Q/K/V/quantized-K/V с реальным quant в forward."""
    handle = CaptureHandle()

    def _make_q_proj_hook():
        def _hook(module, inputs, output):
            handle.q.append(output.detach().clone())
        return _hook

    def _make_pre_hook(layer_idx: int):
        """Pre-hook on attention block: install monkey-patch on cache CLASS.

        Critical: использовать `is None`, не `or`-fallback, потому что
        DynamicCache.__len__() == 0 для пустого кэша → cache truthy-check
        возвращает False даже когда cache существует. Это пропускает
        первый слой (cache пуст до его cache.update).
        """
        def _pre_hook(module, args, kwargs):
            pkv = kwargs.get("past_key_value")
            if pkv is None:
                pkv = kwargs.get("past_key_values")
            if pkv is None and len(args) > 1:
                pkv = args[1]
            if pkv is None:
                return None
            cache_cls = type(pkv)
            if getattr(cache_cls, "_kv_capture_patched", False):
                return None  # класс уже запатчен
            _patch_cache_update(cache_cls, quant_fn, handle)
            handle._patched_caches.append(cache_cls)
            return None
        return _pre_hook

    def _make_outputs_q_hook():
        """Post-hook fallback: захватить Q из outputs[1] tuple для FakeAttention."""
        def _hook(module, inputs, outputs):
            if (
                isinstance(outputs, tuple)
                and len(outputs) == 2
                and isinstance(outputs[1], tuple)
                and len(outputs[1]) == 3
            ):
                q_tup = outputs[1][0]
                # Если q_proj хук уже захватил Q (len(handle.q) > len(handle.k_pre)),
                # не дублируем
                if len(handle.q) <= len(handle.k_pre):
                    handle.q.append(q_tup.detach().clone())
        return _hook

    for layer_idx, mod in enumerate(attention_modules):
        if hasattr(mod, "q_proj"):
            h_q = mod.q_proj.register_forward_hook(_make_q_proj_hook())
            handle._hook_handles.append(h_q)
        h_pre = mod.register_forward_pre_hook(
            _make_pre_hook(layer_idx), with_kwargs=True
        )
        handle._hook_handles.append(h_pre)
        h_post = mod.register_forward_hook(_make_outputs_q_hook())
        handle._hook_handles.append(h_post)

    return handle


def _patch_cache_update(cache_cls: type, quant_fn: Callable, handle: CaptureHandle) -> None:
    """Заменить cache_cls.update (метод КЛАССА) на квантующую версию.

    Class-level patch нужен потому что некоторые места в transformers вызывают
    `Cls.update(self, ...)` напрямую — это обходит instance-attribute patching.

    Эффект: attention.forward вызывает self.past_key_value.update(K, V) →
    наш патч сохраняет K_pre, квантует, сохраняет K_post, и передаёт
    квантованные K_q, V_q в оригинальный update. Cache хранит K_q,
    attention читает K_q, логиты получаются quant-quality.

    Restoration: handle.remove() возвращает оригинальный update класса.
    Идемпотентность: повторный patch при уже запатченном классе — no-op.
    """
    original_update = cache_cls.update

    def quantized_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # K/V pre-quant — снимок до любого преобразования
        handle.k_pre.append(key_states.detach().clone())
        handle.v_pre.append(value_states.detach().clone())

        # Квантуем
        k_q = quant_fn(key_states)
        v_q = quant_fn(value_states)
        handle.k_post.append(k_q.detach().clone())
        handle.v_post.append(v_q.detach().clone())

        # В cache летит quantized — attention увидит K_q
        return original_update(self, k_q, v_q, layer_idx, cache_kwargs)

    cache_cls.update = quantized_update
    cache_cls._kv_capture_patched = True
    cache_cls._kv_capture_original_update = original_update
