import numpy as np
import pytest

from kvtrace.analysis.signatures import (
    aggregate_counts,
    chi_square_test,
    cramers_v,
    row_normalize,
)


def _judgments(method: str, counts: dict[str, int]) -> list[dict]:
    rows = []
    for cat, n in counts.items():
        for _ in range(n):
            rows.append({"quant_method": method, "category": cat})
    return rows


def test_aggregate_counts_shape_and_order():
    js = (
        _judgments("bf16", {"A": 10, "B": 2, "C": 0, "D": 0, "E": 3, "F": 0})
        + _judgments("fp8_e5m2", {"A": 5, "B": 4, "C": 2, "D": 0, "E": 1, "F": 3})
    )
    methods, matrix = aggregate_counts(js)
    assert methods == ["bf16", "fp8_e5m2"]
    assert matrix.shape == (2, 6)
    assert matrix[0].tolist() == [10, 2, 0, 0, 3, 0]
    assert matrix[1].tolist() == [5, 4, 2, 0, 1, 3]


def test_row_normalize_sums_to_one():
    m = np.array([[1, 2, 3, 0, 0, 4], [0, 0, 0, 0, 0, 10]], dtype=float)
    n = row_normalize(m)
    assert n.shape == m.shape
    assert np.allclose(n.sum(axis=1), [1.0, 1.0])


def test_row_normalize_zero_row_yields_zero():
    m = np.array([[0, 0, 0, 0, 0, 0]], dtype=float)
    n = row_normalize(m)
    assert np.all(n == 0.0)


def test_chi_square_detects_difference():
    # clearly different distributions per row
    m = np.array(
        [[30, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 30]], dtype=float
    )
    chi2, p, dof = chi_square_test(m)
    assert p < 0.001
    assert dof == 5


def test_chi_square_same_distribution_high_p():
    m = np.array(
        [[10, 10, 10, 10, 10, 10],
         [10, 10, 10, 10, 10, 10]], dtype=float
    )
    chi2, p, dof = chi_square_test(m)
    assert p > 0.5


def test_cramers_v_in_unit_interval():
    m = np.array([[10, 5], [2, 8]], dtype=float)
    v = cramers_v(m)
    assert 0.0 <= v <= 1.0


def test_aggregate_counts_empty_raises():
    with pytest.raises(ValueError):
        aggregate_counts([])
