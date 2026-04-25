"""TAXONOMY_V1: the 6 error categories for reasoning-trace divergences."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TAXONOMY_VERSION = "v1"

CategoryLetter = Literal["A", "B", "C", "D", "E", "F"]


@dataclass(frozen=True)
class Category:
    letter: CategoryLetter
    name: str
    description: str
    examples: tuple[str, ...]


CATEGORIES: tuple[Category, ...] = (
    Category(
        letter="A",
        name="Arithmetic",
        description="Numerical or symbolic computation error; the strategy is sound but a computation step is wrong.",
        examples=(
            "Claims 7 * 8 = 54 (correct is 56).",
            "Expands (x+1)^2 = x^2 + 1 (missing 2x).",
            "Writes sqrt(48) = 4*sqrt(2) instead of 4*sqrt(3).",
        ),
    ),
    Category(
        letter="B",
        name="Logical",
        description="Premises are correct but the inference step drawn from them is invalid.",
        examples=(
            "From 'all primes > 2 are odd' concludes 'every odd number is prime'.",
            "From 'f(x)=0 has root x=2' concludes 'f is linear'.",
        ),
    ),
    Category(
        letter="C",
        name="Strategy-switch",
        description="Unmotivated abandonment of the current approach for a different one mid-derivation.",
        examples=(
            "Was using Vieta's formulas, suddenly starts expanding a polynomial fully.",
            "Was integrating by parts, abandons and restarts with substitution without reason.",
        ),
    ),
    Category(
        letter="D",
        name="Hallucination",
        description="Invention of a nonexistent or irrelevant fact (fake theorem, wrong formula, nonexistent identity).",
        examples=(
            "Cites 'the Smith-Jones theorem of 2019' to justify a step.",
            "Uses 'the well-known identity sin(x) + cos(x) = 1' (false).",
        ),
    ),
    Category(
        letter="E",
        name="Premature-termination",
        description="Trace cuts off before producing a boxed final answer: hit max tokens, gave up, or stopped mid-sentence.",
        examples=(
            "finish_reason=length and no \\boxed{} at the end.",
            "Text ends with 'I give up' or 'this is beyond me'.",
        ),
    ),
    Category(
        letter="F",
        name="Repetition/loop",
        description="The same reasoning step (paragraph or equation) is repeated three or more times without progress.",
        examples=(
            "Rewrites 'Let me try x=2' five times in a row.",
            "Repeats the same equation manipulation step in a loop.",
        ),
    ),
)


_BY_LETTER = {c.letter: c for c in CATEGORIES}


def lookup(letter: str) -> Category:
    key = letter.strip().upper()
    if key not in _BY_LETTER:
        raise ValueError(f"Unknown category {letter!r}. Expected one of A-F.")
    return _BY_LETTER[key]  # type: ignore[index]
