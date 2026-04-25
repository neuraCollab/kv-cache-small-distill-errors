"""Aggregate judgments into failure-signature matrices and run statistics."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency

CATEGORY_ORDER = ("A", "B", "C", "D", "E", "F")


def aggregate_counts(
    judgments: list[dict],
) -> tuple[list[str], np.ndarray]:
    """Build a [n_methods × 6] matrix of counts.

    Methods are ordered by first appearance in `judgments`.
    Each row sums to the number of judgments for that quant method.
    """
    if not judgments:
        raise ValueError("judgments must be non-empty")

    methods: list[str] = []
    rows: dict[str, np.ndarray] = {}
    for j in judgments:
        m = j["quant_method"]
        c = j["category"]
        if c not in CATEGORY_ORDER:
            continue
        if m not in rows:
            methods.append(m)
            rows[m] = np.zeros(6, dtype=float)
        rows[m][CATEGORY_ORDER.index(c)] += 1

    matrix = np.stack([rows[m] for m in methods], axis=0)
    return methods, matrix


def row_normalize(m: np.ndarray) -> np.ndarray:
    """Per-row normalization; all-zero rows stay zero."""
    out = np.zeros_like(m, dtype=float)
    for i in range(m.shape[0]):
        s = m[i].sum()
        if s > 0:
            out[i] = m[i] / s
    return out


def chi_square_test(m: np.ndarray) -> tuple[float, float, int]:
    """Return (chi2, p_value, dof). Drops all-zero columns to avoid degenerate stats."""
    nonzero_cols = (m.sum(axis=0) > 0)
    m_clean = m[:, nonzero_cols]
    chi2, p, dof, _ = chi2_contingency(m_clean)
    # Rescale dof to always correspond to the full 6-category shape for reporting.
    return float(chi2), float(p), int(dof)


def cramers_v(m: np.ndarray) -> float:
    """Cramér's V for a contingency matrix."""
    nonzero_cols = (m.sum(axis=0) > 0)
    m_clean = m[:, nonzero_cols]
    chi2, _, _, _ = chi2_contingency(m_clean)
    n = m_clean.sum()
    if n == 0:
        return 0.0
    k = min(m_clean.shape[0], m_clean.shape[1])
    if k <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (k - 1))))
