"""Live-API calibration: judge must classify ≥7/10 golden examples correctly."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from kvtrace.judge.claude_judge import ClaudeJudge
from kvtrace.judge.golden_set import GOLDEN_SET


@pytest.mark.live_api
def test_judge_passes_golden_set(tmp_path: Path) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    judge = ClaudeJudge(client=client, model="claude-sonnet-4-6", cache_dir=tmp_path)

    correct = 0
    errors: list[str] = []
    for ex in GOLDEN_SET:
        r = judge.judge(
            problem=ex.problem,
            ground_truth=ex.ground_truth,
            baseline_context=ex.baseline_context,
            quant_context=ex.quant_context,
            common_prefix=ex.common_prefix,
            quant_method_name="test",
        )
        if r.category == ex.gold_category:
            correct += 1
        else:
            errors.append(f"{ex.id}: gold={ex.gold_category} got={r.category} ({r.rationale[:60]})")

    accuracy = correct / len(GOLDEN_SET)
    msg = f"calibration {correct}/{len(GOLDEN_SET)} = {accuracy:.0%}. Errors:\n" + "\n".join(errors)
    assert accuracy >= 0.7, msg
