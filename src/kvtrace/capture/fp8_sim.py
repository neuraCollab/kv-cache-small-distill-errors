"""FP8 quant→dequant simulator (pure PyTorch, CPU-friendly).

Uses PyTorch's native float8_e4m3fn and float8_e5m2 dtypes (PyTorch >= 2.1).
Cast bf16 → fp8 → bf16 reproduces IEEE FP8 round-to-nearest-even, identical
to vLLM 0.7.3's default per-tensor KV-quant.

No per-channel scaling, no stochastic rounding — same defaults as the main
experiment (see config/quant_methods.yaml).
"""
from __future__ import annotations

from typing import Callable

import torch


def fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """bf16 → fp8_e4m3 → bf16. Narrow range (±448), high precision."""
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def fp8_e5m2(x: torch.Tensor) -> torch.Tensor:
    """bf16 → fp8_e5m2 → bf16. Wide range (±57344), lower precision."""
    return x.to(torch.float8_e5m2).to(x.dtype)


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _hqq_quant_dequant(
    x: torch.Tensor, n_bits: int, group_size: int = 64
) -> torch.Tensor:
    """Real HQQ (Half-Quadratic Quantization) через `hqq` пакет.

    Делает proper HQ optimization: minimize ||W - dequant(quant(W; s, z))||_p
    с p<2 (robust к outliers). Это то, что используется в
    `transformers.QuantizedCacheConfig(backend="hqq")`.

    Args:
        x: bf16 tensor any shape
        n_bits: 4 (INT4) или 2 (INT2)
        group_size: HQQ default 64
    Returns:
        bf16 tensor same shape, after quant→dequant
    """
    from hqq.core.quantize import Quantizer
    orig_shape = x.shape
    orig_dtype = x.dtype
    x_float = x.float().flatten()
    n = x_float.numel()
    # Pad to multiple of group_size (HQQ requires this)
    n_pad = (group_size - n % group_size) % group_size
    if n_pad > 0:
        x_float = torch.cat([x_float, torch.zeros(n_pad, dtype=x_float.dtype)])
    # HQQ.quantize ожидает 1D или 2D. 1D с group_size работает корректно.
    W_q, meta = Quantizer.quantize(
        x_float, nbits=n_bits, group_size=group_size, axis=0,
        optimize=True, device="cpu",
    )
    dq = Quantizer.dequantize(W_q, meta)  # flat shape
    return dq.flatten()[:n].view(orig_shape).to(orig_dtype)


def hqq_int4(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Real HQQ INT4 — proximal HQ optimization, group_size=64."""
    return _hqq_quant_dequant(x, n_bits=4, group_size=group_size)


def hqq_int2(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Real HQQ INT2."""
    return _hqq_quant_dequant(x, n_bits=2, group_size=group_size)


# Legacy names для backwards-compat с тестами / external scripts
int4_minmax = hqq_int4
int2_minmax = hqq_int2


QUANT_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "bf16": _identity,
    "fp8_e4m3": fp8_e4m3,
    "fp8_e5m2": fp8_e5m2,
    "hqq_int4": hqq_int4,
    "hqq_int2": hqq_int2,
}


def fp8_skip_outliers(
    K: torch.Tensor,
    outlier_channels: list[tuple[int, int]],
    base_fn: Callable[[torch.Tensor], torch.Tensor] = fp8_e4m3,
) -> torch.Tensor:
    """Quant K через base_fn, КРОМЕ перечисленных (head, channel) — те остаются bf16.

    Симулирует per-channel defense: top-N outlier-каналов сохраняются
    в bf16, остальные ~1014 каналов квантуются FP8.

    Args:
        K: [W, num_kv_heads, head_dim]
        outlier_channels: [(head, channel), ...] — индексы skip-quant
        base_fn: фоновая quant-функция (default fp8_e4m3)
    Returns:
        K_alt same shape, dtype = K.dtype
    """
    if not outlier_channels:
        return base_fn(K)
    out = base_fn(K).clone()
    for head, channel in outlier_channels:
        out[:, head, channel] = K[:, head, channel]
    return out


def identify_top_outlier_channels(
    K: torch.Tensor, top_n: int
) -> list[tuple[int, int]]:
    """Top-N каналов (head, channel) с наибольшим max|K[:, head, channel]|."""
    if top_n <= 0:
        return []
    max_per = K.abs().amax(dim=0)  # [num_heads, head_dim]
    num_heads, head_dim = max_per.shape
    flat = max_per.flatten()
    top_n = min(top_n, len(flat))
    _, idx = flat.topk(top_n)
    return [(int(i // head_dim), int(i % head_dim)) for i in idx.tolist()]
