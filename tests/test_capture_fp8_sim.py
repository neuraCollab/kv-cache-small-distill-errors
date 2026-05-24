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
    assert set(QUANT_FNS.keys()) == {"bf16", "fp8_e4m3", "fp8_e5m2", "hqq_int4", "hqq_int2"}


def test_int4_minmax_preserves_shape_and_dtype():
    from kvtrace.capture.fp8_sim import int4_minmax
    x = torch.randn(8, 4, dtype=torch.bfloat16)
    y = int4_minmax(x)
    assert y.shape == x.shape
    assert y.dtype == torch.bfloat16


def test_int4_minmax_has_at_most_16_unique_per_group():
    from kvtrace.capture.fp8_sim import int4_minmax
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.bfloat16)
    y = int4_minmax(x, group_size=64)
    assert len(torch.unique(y)) <= 16


def test_int2_minmax_has_at_most_4_unique_per_group():
    from kvtrace.capture.fp8_sim import int2_minmax
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.bfloat16)
    y = int2_minmax(x, group_size=64)
    assert len(torch.unique(y)) <= 4


def test_int_minmax_error_ordering():
    """fp8_e4m3 (7-bit precision) <= int4 (4-bit) <= int2 (2-bit)."""
    torch.manual_seed(0)
    x = torch.randn(2048, dtype=torch.bfloat16) * 10
    from kvtrace.capture.fp8_sim import int4_minmax, int2_minmax
    err_fp8 = (fp8_e4m3(x).float() - x.float()).norm()
    err_int4 = (int4_minmax(x).float() - x.float()).norm()
    err_int2 = (int2_minmax(x).float() - x.float()).norm()
    assert err_fp8 < err_int4 < err_int2
