"""Token-level alignment primitives."""
from __future__ import annotations

from typing import Any


def first_token_mismatch(a: list[int], b: list[int]) -> int | None:
    """Index of the first position where a[i] != b[i], or None if identical.

    When one list is a strict prefix of the other, returns min(len(a), len(b)).
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def decode_window(tokenizer: Any, tokens: list[int], center: int, context: int) -> str:
    """Decode tokens[center-context : center+context], clipping to bounds."""
    lo = max(0, center - context)
    hi = min(len(tokens), center + context + 1)
    return tokenizer.decode(tokens[lo:hi], skip_special_tokens=False)
