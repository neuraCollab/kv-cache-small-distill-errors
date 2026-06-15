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
    q: list[torch.Tensor] = field(default_factory=list)             # pre-RoPE
    q_post_rope: list[torch.Tensor] = field(default_factory=list)   # post-RoPE
    k_pre: list[torch.Tensor] = field(default_factory=list)         # post-RoPE (cache)
    v_pre: list[torch.Tensor] = field(default_factory=list)
    k_post: list[torch.Tensor] = field(default_factory=list)
    v_post: list[torch.Tensor] = field(default_factory=list)
    _hook_handles: list[Any] = field(default_factory=list)
    _patched_caches: list[Any] = field(default_factory=list)
    _patched_rope_modules: list[tuple[Any, Any]] = field(default_factory=list)

    def remove(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        # Восстановить cache.update (см. _patch_cache_update)
        for cache_cls in self._patched_caches:
            if hasattr(cache_cls, "_kv_capture_original_update"):
                cache_cls.update = cache_cls._kv_capture_original_update
                delattr(cache_cls, "_kv_capture_original_update")
                delattr(cache_cls, "_kv_capture_patched")
        self._patched_caches.clear()
        # Восстановить apply_rotary_pos_emb на модулях, где мы пропатчили
        for module, original_fn in self._patched_rope_modules:
            module.apply_rotary_pos_emb = original_fn
            if hasattr(module, "_kv_capture_rope_patched"):
                delattr(module, "_kv_capture_rope_patched")
        self._patched_rope_modules.clear()


def install_capture_hooks(
    model: nn.Module,
    attention_modules: list[nn.Module],
    quant_fn: Callable[[torch.Tensor], torch.Tensor],
    quant_only_layers: set[int] | None = None,
) -> CaptureHandle:
    """Установить хуки для захвата Q/K/V/quantized-K/V с реальным quant в forward.

    Args:
        quant_only_layers: если не None, квант применяется ТОЛЬКО к этим слоям
            (остальные используют bf16). Для layer-ablation studies.
    """
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
            _patch_cache_update(cache_cls, quant_fn, handle, quant_only_layers)
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

    # Patch apply_rotary_pos_emb для захвата post-RoPE Q.
    # Делаем это глобально на модуле transformers.models.qwen3.modeling_qwen3,
    # потому что Qwen3Attention.forward вызывает функцию через module-global lookup.
    _patch_rope_for_q_post(handle)

    return handle


def _patch_rope_for_q_post(handle: CaptureHandle) -> None:
    """Monkey-patch apply_rotary_pos_emb в Qwen3 modeling module чтобы захватывать
    post-RoPE Q. Необходимо для attention-map analysis (Q post-RoPE × K post-RoPE).

    Идемпотентно: повторный патч — no-op. Restoration через handle.remove().
    Если модуль qwen3 не импортирован — silently skip.
    """
    try:
        from transformers.models.qwen3 import modeling_qwen3
    except ImportError:
        return
    if getattr(modeling_qwen3, "_kv_capture_rope_patched", False):
        return
    original_arpe = modeling_qwen3.apply_rotary_pos_emb

    def patched_arpe(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_rot, k_rot = original_arpe(q, k, cos, sin, position_ids, unsqueeze_dim)
        handle.q_post_rope.append(q_rot.detach().clone())
        return q_rot, k_rot

    modeling_qwen3.apply_rotary_pos_emb = patched_arpe
    modeling_qwen3._kv_capture_rope_patched = True
    handle._patched_rope_modules.append((modeling_qwen3, original_arpe))


def _patch_cache_update(
    cache_cls: type,
    quant_fn: Callable,
    handle: CaptureHandle,
    quant_only_layers: set[int] | None = None,
) -> None:
    """Заменить cache_cls.update (метод КЛАССА) на квантующую версию.

    Class-level patch нужен потому что некоторые места в transformers вызывают
    `Cls.update(self, ...)` напрямую — это обходит instance-attribute patching.

    Если quant_only_layers задан, квант применяется ТОЛЬКО на этих слоях.
    На остальных вызывается оригинальный update без модификации K, V — bf16.
    Это для layer-ablation studies (см. scripts/10_layer_ablation.py).

    Restoration: handle.remove() возвращает оригинальный update класса.
    """
    original_update = cache_cls.update

    def quantized_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # K/V pre-quant — снимок до любого преобразования
        handle.k_pre.append(key_states.detach().clone())
        handle.v_pre.append(value_states.detach().clone())

        if quant_only_layers is not None and layer_idx not in quant_only_layers:
            # Этот слой не квантуем — пропускаем через оригинал
            handle.k_post.append(key_states.detach().clone())
            handle.v_post.append(value_states.detach().clone())
            return original_update(self, key_states, value_states, layer_idx, cache_kwargs)

        # quant_fn может быть либо одноаргументной (K) → K_q, либо
        # двухаргументной (K, layer_idx) → K_q для layer-aware defense.
        # Inspect via try/except — простая, robust к разным signature'ам.
        try:
            k_q = quant_fn(key_states, layer_idx)
            v_q = quant_fn(value_states, layer_idx)
        except TypeError:
            k_q = quant_fn(key_states)
            v_q = quant_fn(value_states)
        handle.k_post.append(k_q.detach().clone())
        handle.v_post.append(v_q.detach().clone())

        return original_update(self, k_q, v_q, layer_idx, cache_kwargs)

    cache_cls.update = quantized_update
    cache_cls._kv_capture_patched = True
    cache_cls._kv_capture_original_update = original_update
