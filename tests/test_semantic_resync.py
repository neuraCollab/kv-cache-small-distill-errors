from unittest.mock import MagicMock, patch

import numpy as np

from kvtrace.fdp.semantic_resync import (
    ReSyncChecker,
    split_into_reasoning_units,
)


def test_split_by_double_newline():
    text = "First step.\n\nSecond step.\n\nThird step."
    units = split_into_reasoning_units(text)
    assert units == ["First step.", "Second step.", "Third step."]


def test_split_by_transition_markers():
    text = "Let me try x=1. Wait, that's wrong. So, try x=2."
    units = split_into_reasoning_units(text)
    assert any("Let me try" in u for u in units)
    assert any("Wait" in u for u in units)
    assert any("So," in u for u in units)


def test_split_handles_empty_string():
    assert split_into_reasoning_units("") == []
    assert split_into_reasoning_units("   ") == []


def test_checker_all_cosmetic(monkeypatch):
    checker = ReSyncChecker(threshold=0.9)
    fake_model = MagicMock()
    # Make every pair of embeddings identical → cosine=1.0
    fake_model.encode.side_effect = lambda texts, **kwargs: np.ones((len(texts), 4))
    monkeypatch.setattr(checker, "_model", fake_model)

    is_cosmetic = checker.is_cosmetic_divergence(
        "Let me try x=1. Wait, that fails. So x=2.",
        "Let's try x=1. Hmm, fails. So, x=2.",
    )
    assert is_cosmetic is True


def test_checker_real_divergence(monkeypatch):
    checker = ReSyncChecker(threshold=0.9)
    fake_model = MagicMock()
    # Orthogonal vectors → cosine=0
    def encode(texts, **kwargs):
        n = len(texts)
        v = np.eye(max(n, 4))[:n, :4]
        return v
    fake_model.encode.side_effect = encode
    monkeypatch.setattr(checker, "_model", fake_model)

    is_cosmetic = checker.is_cosmetic_divergence(
        "Let me compute 2+2=4.",
        "Let me compute 2+2=5.",
    )
    assert is_cosmetic is False


def test_checker_few_units_is_not_cosmetic(monkeypatch):
    checker = ReSyncChecker(threshold=0.9)
    fake_model = MagicMock()
    fake_model.encode.side_effect = lambda texts, **kwargs: np.ones((len(texts), 4))
    monkeypatch.setattr(checker, "_model", fake_model)

    # Only 1 unit in each → not enough to judge re-sync; must return False.
    assert checker.is_cosmetic_divergence("one.", "one!") is False
