from unittest.mock import patch

import pytest

from kvtrace.dataset_loader import (
    DATASET_CONFIGS,
    MathProblem,
    load_math_dataset,
    load_study_mix,
)


def test_math_problem_to_dict_roundtrip():
    p = MathProblem(idx=0, problem="x+1=2", answer="1", source="aime-24")
    d = p.to_dict()
    assert d["idx"] == 0
    assert d["problem"] == "x+1=2"
    assert d["ground_truth"] == "1"
    assert d["source"] == "aime-24"


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_math_dataset("gsm8k")


class _DS:
    def __init__(self, rows):
        self._rows = rows
    def __len__(self):
        return len(self._rows)
    def __iter__(self):
        return iter(self._rows)
    def select(self, indices):
        return _DS([self._rows[i] for i in indices])
    def shuffle(self, seed):
        return self


def test_load_math_dataset_aime(aime_tiny, monkeypatch):
    import kvtrace.dataset_loader as dl
    monkeypatch.setattr(dl, "load_dataset", lambda path, split: _DS(aime_tiny))
    problems = dl.load_math_dataset("aime-24", num_samples=2)
    assert len(problems) == 2
    assert problems[0].source == "aime-24"
    assert problems[0].answer == "40"
    assert "Find the smallest" in problems[0].problem


def test_load_math_dataset_math500(math500_tiny, monkeypatch):
    import kvtrace.dataset_loader as dl
    monkeypatch.setattr(dl, "load_dataset", lambda path, split: _DS(math500_tiny))
    problems = dl.load_math_dataset("math-500", num_samples=2)
    assert len(problems) == 2
    assert problems[0].answer == "4"


def test_load_study_mix_preserves_order(aime_tiny, math500_tiny, monkeypatch):
    """30 AIME first, then 50 MATH — but with tiny fixture, 2 + 2 = 4."""
    import kvtrace.dataset_loader as dl
    calls = {"aime": 0, "math": 0}

    def _fake(path, split):
        if "AIME" in path:
            calls["aime"] += 1
            return _DS(aime_tiny)
        calls["math"] += 1
        return _DS(math500_tiny)

    monkeypatch.setattr(dl, "load_dataset", _fake)
    problems = load_study_mix(aime_n=2, math_n=2)

    assert len(problems) == 4
    assert problems[0].source == "aime-24"
    assert problems[1].source == "aime-24"
    assert problems[2].source == "math-500"
    assert problems[3].source == "math-500"
    assert [p.idx for p in problems] == [0, 1, 2, 3]
