"""Shared pytest fixtures and helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def aime_tiny(fixtures_dir) -> list[dict]:
    with (fixtures_dir / "aime_tiny.json").open() as f:
        return json.load(f)


@pytest.fixture
def math500_tiny(fixtures_dir) -> list[dict]:
    with (fixtures_dir / "math500_tiny.json").open() as f:
        return json.load(f)


@pytest.fixture
def baseline_trace_text(fixtures_dir) -> str:
    return (fixtures_dir / "trace_baseline_sample.txt").read_text(encoding="utf-8")


@pytest.fixture
def quant_trace_text(fixtures_dir) -> str:
    return (fixtures_dir / "trace_quant_sample.txt").read_text(encoding="utf-8")
