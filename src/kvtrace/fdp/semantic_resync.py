"""Semantic re-sync via MiniLM cosine similarity."""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_TRANSITION_RE = re.compile(
    r"(?=(?:\bLet me\b|\bLet's\b|\bWait,|\bSo,|\bTherefore\b|\bHmm,|\bActually,|\bStep \d+))"
)


def split_into_reasoning_units(text: str) -> list[str]:
    """Split into 'reasoning units' via double-newlines and transition markers."""
    if not text or not text.strip():
        return []
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units: list[str] = []
    for p in raw_paragraphs:
        subs = _TRANSITION_RE.split(p)
        for s in subs:
            s = s.strip()
            if s:
                units.append(s)
    return units


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class ReSyncChecker:
    """Wraps a sentence-transformers model for unit-level cosine similarity.

    Model is loaded lazily and cached on the instance to keep CPU startup cheap
    for tests that monkey-patch `._model`.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.9,
        k: int = 3,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.k = k
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def is_cosmetic_divergence(self, baseline_window: str, quant_window: str) -> bool:
        b_units = split_into_reasoning_units(baseline_window)[: self.k]
        q_units = split_into_reasoning_units(quant_window)[: self.k]
        if len(b_units) < 2 or len(q_units) < 2:
            return False
        n = min(len(b_units), len(q_units))
        model = self._ensure_model()
        b_emb = model.encode(b_units[:n], convert_to_numpy=True)
        q_emb = model.encode(q_units[:n], convert_to_numpy=True)
        sims = [_cosine(b_emb[i], q_emb[i]) for i in range(n)]
        above = sum(1 for s in sims if s >= self.threshold)
        # "2 of 3" rule: at least majority above threshold
        return above >= max(2, (n + 1) // 2)
