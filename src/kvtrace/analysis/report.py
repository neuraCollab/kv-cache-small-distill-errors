"""Markdown + JSON report generation with matplotlib plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kvtrace.analysis.signatures import (  # noqa: E402
    CATEGORY_ORDER,
    aggregate_counts,
    chi_square_test,
    cramers_v,
    row_normalize,
)


def build_report(
    judgments: list[dict],
    *,
    md_path: Path,
    json_path: Path,
) -> None:
    methods, counts = aggregate_counts(judgments)
    signatures = row_normalize(counts)
    chi2, p, dof = chi_square_test(counts)
    v = cramers_v(counts)

    # --- JSON
    data = {
        "methods": methods,
        "categories": list(CATEGORY_ORDER),
        "counts": counts.astype(int).tolist(),
        "signatures": signatures.tolist(),
        "chi_square": chi2,
        "chi_square_p": p,
        "chi_square_dof": dof,
        "cramers_v": v,
    }
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    # --- Markdown
    lines: list[str] = []
    lines.append("# KV Cache Quantization — Failure-Signature Report\n")
    lines.append(f"Total judgments: **{int(counts.sum())}**\n")
    lines.append(f"Chi-square: chi2={chi2:.2f}, dof={dof}, **p={p:.3g}**\n")
    lines.append(f"Cramér's V: **{v:.3f}**\n")
    lines.append("")

    lines.append("## Raw counts (rows = quant method, cols = category A..F)\n")
    lines.append("| method | " + " | ".join(CATEGORY_ORDER) + " | total |")
    lines.append("|---|" + "|".join(["---"] * 6) + "|---|")
    for m, row in zip(methods, counts.astype(int), strict=False):
        lines.append(f"| {m} | " + " | ".join(str(x) for x in row) + f" | {int(row.sum())} |")

    lines.append("\n## Normalized failure signatures (rows sum to 1)\n")
    lines.append("| method | " + " | ".join(CATEGORY_ORDER) + " |")
    lines.append("|---|" + "|".join(["---"] * 6) + "|")
    for m, row in zip(methods, signatures, strict=False):
        lines.append(f"| {m} | " + " | ".join(f"{x:.2f}" for x in row) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_heatmap(matrix: np.ndarray, method_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, max(2, len(method_names) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(list(CATEGORY_ORDER))
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names)
    ax.set_xlabel("error category")
    ax.set_title("Failure signature per quantization method")
    fig.colorbar(im, ax=ax, label="fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
