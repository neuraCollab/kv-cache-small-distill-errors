"""Unit-tests для FP8 quant→dequant симулятора (pure-PyTorch CPU)."""
from __future__ import annotations

import torch

from kvtrace.capture.fp8_sim import QUANT_FNS, fp8_e4m3, fp8_e5m2


def test_bf16_identity():
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    assert torch.equal(QUANT_FNS["bf16"](x), x)


def test_fp8_e4m3_idempotent():
    """qd(qd(x)) == qd(x): второе применение не меняет уже-в-fp8 значения."""
    x = torch.randn(64, dtype=torch.bfloat16)
    once = fp8_e4m3(x)
    twice = fp8_e4m3(once)
    assert torch.equal(once, twice)


def test_fp8_e5m2_idempotent():
    x = torch.randn(64, dtype=torch.bfloat16)
    once = fp8_e5m2(x)
    twice = fp8_e5m2(once)
    assert torch.equal(once, twice)


def test_preserves_dtype():
    x = torch.randn(4, dtype=torch.bfloat16)
    assert fp8_e4m3(x).dtype == torch.bfloat16
    assert fp8_e5m2(x).dtype == torch.bfloat16


def test_changes_values_for_random_input():
    """Гарантирует, что quant→dequant НЕ identity для произвольных bf16 значений."""
    torch.manual_seed(0)
    x = torch.randn(4096, dtype=torch.bfloat16)
    assert not torch.equal(fp8_e4m3(x), x)
    assert not torch.equal(fp8_e5m2(x), x)


def test_e4m3_more_precise_than_e5m2_within_range():
    """Внутри ±448 (диапазон e4m3) e4m3 точнее e5m2 — больше mantissa bits.

    e4m3: 4 expo / 3 mantissa → max ≈ 448, шаг внутри binade [256, 512) = 32
    e5m2: 5 expo / 2 mantissa → max ≈ 57344, шаг внутри binade [256, 512) = 64
    Поэтому в общем диапазоне ошибка квантования у e4m3 меньше.
    """
    torch.manual_seed(0)
    x = torch.randn(4096, dtype=torch.bfloat16) * 10  # range ~±30, well in both
    err_e4m3 = (fp8_e4m3(x) - x).abs().sum()
    err_e5m2 = (fp8_e5m2(x) - x).abs().sum()
    assert err_e4m3 < err_e5m2


def test_quant_fns_registry_keys():
    assert set(QUANT_FNS.keys()) == {"bf16", "fp8_e4m3", "fp8_e5m2"}
