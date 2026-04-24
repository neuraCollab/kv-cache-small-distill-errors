"""Parse and validate Claude's JSON response into a JudgmentResult."""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from kvtrace.judge.taxonomy import lookup

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeParseError(ValueError):
    """Raised when Claude's response cannot be parsed as a valid judgment."""


class JudgmentResult(BaseModel):
    category: Literal["A", "B", "C", "D", "E", "F"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    affected_span: str

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f))


def parse_judge_response(text: str) -> JudgmentResult:
    """Extract JSON from Claude's response and validate it."""
    fence = _JSON_FENCE_RE.search(text)
    raw = fence.group(1) if fence else text
    m = _JSON_OBJECT_RE.search(raw)
    if not m:
        raise JudgeParseError(f"No JSON object found in response: {text[:200]!r}")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"Invalid JSON: {e}") from e

    # Validate category letter against the taxonomy BEFORE letting pydantic coerce.
    cat = str(data.get("category", "")).strip().upper()
    try:
        lookup(cat)
    except ValueError as e:
        raise JudgeParseError(str(e)) from e
    data["category"] = cat

    required = {"category", "confidence", "rationale", "affected_span"}
    missing = required - data.keys()
    if missing:
        raise JudgeParseError(f"Missing fields: {sorted(missing)}")

    try:
        return JudgmentResult(**data)
    except Exception as e:
        raise JudgeParseError(f"Schema validation failed: {e}") from e
