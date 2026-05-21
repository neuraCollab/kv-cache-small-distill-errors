"""Unit-тесты для attention hooks на минимальной игрушечной модели.

Цель: убедиться, что хук собирает Q/K_pre/K_post/V_pre/V_post в правильном
порядке и при подмене K/V через quant_fn возвращает quantized версию.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kvtrace.capture.attention_hooks import CaptureHandle, install_capture_hooks
from kvtrace.capture.fp8_sim import fp8_e4m3


class _FakeCache:
    """Минимальный DynamicCache-like: list[Tensor] для K и V."""
    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class _FakeAttention(nn.Module):
    """Attention block с явными Q/K/V для проверки хука."""
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, hidden_states, past_key_value=None):
        # Игрушечные Q/K/V — три проекции = identity для простоты.
        # K/V в HF Qwen3 layout: [B, num_heads, seq, head_dim] (seq at dim -2).
        # Q остаётся в [B, seq, num_heads, head_dim] — после q_proj, до transpose.
        bsz, seq, dim = hidden_states.shape
        q = hidden_states.view(bsz, seq, 1, dim).contiguous()
        k = hidden_states.view(bsz, 1, seq, dim).contiguous() * 1.5  # HF layout
        v = hidden_states.view(bsz, 1, seq, dim).contiguous() * 2.0  # HF layout
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)
        # Dummy attention output — реальная attention-математика не нужна для
        # теста хуков; multiplying q*k*v fails when cache grows in AR mode
        # (sizes mismatch). Просто возвращаем hidden_states как заглушку.
        out = hidden_states
        return out, (q, k, v)


class _FakeModel(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_FakeAttention(i) for i in range(n_layers)])

    def forward(self, hidden_states, past_key_value=None):
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, past_key_value=past_key_value)
        return hidden_states


def test_hook_captures_q_k_v_per_layer():
    model = _FakeModel(n_layers=2)
    handle: CaptureHandle = install_capture_hooks(
        model,
        attention_modules=list(model.layers),
        quant_fn=lambda x: x,  # bf16 — identity
    )
    try:
        cache = _FakeCache()
        x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        model(x, past_key_value=cache)

        assert len(handle.q) == 2  # 2 layers
        assert len(handle.k_pre) == 2
        assert len(handle.v_post) == 2
        assert handle.q[0].shape == (1, 4, 1, 8)
    finally:
        handle.remove()


def test_hook_quantizes_kv_in_cache():
    model = _FakeModel(n_layers=1)
    handle = install_capture_hooks(
        model,
        attention_modules=list(model.layers),
        quant_fn=fp8_e4m3,
    )
    try:
        cache = _FakeCache()
        x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        model(x, past_key_value=cache)

        # k_pre — оригинальный K; k_post — после fp8_e4m3
        assert torch.equal(handle.k_post[0], fp8_e4m3(handle.k_pre[0]))
        # Кеш модели тоже содержит quantized версию
        assert torch.equal(cache.key_cache[0], handle.k_post[0])
    finally:
        handle.remove()


def test_remove_hooks_stops_capturing():
    model = _FakeModel(n_layers=1)
    handle = install_capture_hooks(
        model,
        attention_modules=list(model.layers),
        quant_fn=lambda x: x,
    )
    cache = _FakeCache()
    x = torch.randn(1, 2, 8, dtype=torch.bfloat16)
    model(x, past_key_value=cache)
    assert len(handle.q) == 1

    handle.remove()
    # После remove повторный forward не должен накопить ничего нового
    model(x, past_key_value=cache)
    assert len(handle.q) == 1


class _FakeHFAttention(nn.Module):
    """Имитирует HF Qwen3Attention: имеет .q_proj и кладёт K/V в кеш."""
    def __init__(self, layer_idx: int, dim: int = 8):
        super().__init__()
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states, past_key_value=None):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # HF Qwen3 K/V layout: [B, num_heads, seq, head_dim] (seq at dim -2).
        bsz, seq, dim = q.shape
        q = q.view(bsz, seq, 1, dim)
        k = k.view(bsz, 1, seq, dim)
        v = v.view(bsz, 1, seq, dim)
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)
        return (q.sum(-2), None)


class _FakeHFModel(nn.Module):
    def __init__(self, n_layers: int = 1, dim: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([_FakeHFAttention(i, dim) for i in range(n_layers)])

    def forward(self, hidden_states, past_key_value=None):
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, past_key_value=past_key_value)
        return hidden_states


def test_hook_captures_q_via_q_proj_on_hf_style_attention():
    model = _FakeHFModel(n_layers=1, dim=8)
    handle = install_capture_hooks(
        model,
        attention_modules=list(model.layers),
        quant_fn=lambda x: x,
    )
    try:
        cache = _FakeCache()
        x = torch.randn(1, 3, 8, dtype=torch.float32)
        model(x, past_key_value=cache)

        # Q должно быть захвачено через хук на q_proj
        assert len(handle.q) == 1
        assert handle.q[0].shape == (1, 3, 8)  # output q_proj без reshape
    finally:
        handle.remove()
