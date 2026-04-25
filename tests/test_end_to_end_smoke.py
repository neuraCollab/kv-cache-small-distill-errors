"""Full pipeline on a synthetic in-memory dataset and mocked generators/judge.

This is the top-level CI gate. If it passes, the phases are wired up correctly.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kvtrace.dataset_loader import MathProblem
from kvtrace.fdp.finder import find_fdp
from kvtrace.generators.base import GenerationResult


class NoCosmeticChecker:
    def is_cosmetic_divergence(self, *a, **kw):
        return False


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(65 + (i % 26)) for i in ids)


def _fake_generation(idx: int, tokens: list[int], text: str, boxed: str | None) -> GenerationResult:
    return GenerationResult(
        idx=idx, raw=text, think=None, final_response=text, boxed_answer=boxed,
        think_complete=True, finish_reason="stop",
        token_ids=tokens, prompt_tokens=5, generated_tokens=len(tokens),
    )


def test_smoke_pipeline_without_network(tmp_path: Path):
    # Phase 1 (synthesized): 2 problems, 1 baseline + 1 quant config
    problems = [
        MathProblem(idx=0, problem="p0", answer="42", source="aime-24"),
        MathProblem(idx=1, problem="p1", answer="7", source="math-500"),
    ]
    baseline_rows = [
        _fake_generation(0, [1, 2, 3, 4, 5], "baseline-0", "42"),
        _fake_generation(1, [1, 2, 3, 4, 5], "baseline-1", "7"),
    ]
    quant_rows = [
        _fake_generation(0, [1, 2, 9, 4, 5], "quant-0", "41"),
        _fake_generation(1, [1, 2, 3, 4, 5], "quant-1", "7"),
    ]

    # Phase 2: run FDP
    fdp_records = []
    for base, quant, prob in zip(baseline_rows, quant_rows, problems):
        r = find_fdp(
            baseline_tokens=base.token_ids,
            quant_tokens=quant.token_ids,
            baseline_text=base.raw,
            quant_text=quant.raw,
            tokenizer=FakeTokenizer(),
            resync_checker=NoCosmeticChecker(),
            baseline_boxed=base.boxed_answer,
            quant_boxed=quant.boxed_answer,
            ground_truth=prob.answer,
        )
        fdp_records.append(r)

    # Only the first problem has a divergence
    assert fdp_records[0].fdp_token_idx == 2
    assert fdp_records[1].fdp_token_idx is None

    # Phase 3 (mock judge): fake judgments — arithmetic for prob 0, no judgment for prob 1
    judgments = [
        {"problem_idx": 0, "quant_method": "fp8_e5m2", "model": "m",
         "category": "A", "confidence": 0.9, "rationale": "r", "affected_span": "s"},
    ]

    # Phase 4: analyze
    from kvtrace.analysis.report import build_report
    md = tmp_path / "report.md"
    js = tmp_path / "report.json"
    build_report(judgments, md_path=md, json_path=js)

    text = md.read_text()
    assert "fp8_e5m2" in text
    assert "Chi-square" in text
    data = json.loads(js.read_text())
    assert data["methods"] == ["fp8_e5m2"]
