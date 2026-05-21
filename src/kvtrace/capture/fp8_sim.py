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


QUANT_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "bf16": _identity,
    "fp8_e4m3": fp8_e4m3,
    "fp8_e5m2": fp8_e5m2,
}
