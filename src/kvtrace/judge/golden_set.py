"""Manually curated (trace_pair, gold_category) examples for judge calibration.

Each example deliberately exhibits ONE category A..F so that a correctly
working judge classifies each with ≥ calibration_threshold accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GoldenExample:
    id: str
    problem: str
    ground_truth: str
    baseline_context: str
    quant_context: str
    common_prefix: str
    gold_category: Literal["A", "B", "C", "D", "E", "F"]


GOLDEN_SET: tuple[GoldenExample, ...] = (
    GoldenExample(
        id="A1_arith_mul",
        problem="Compute 17 * 23.",
        ground_truth="391",
        common_prefix="Let me multiply 17 and 23. I'll break it up: 17*20 + 17*3.",
        baseline_context="17*20 = 340. 17*3 = 51. So 340+51 = 391.",
        quant_context="17*20 = 340. 17*3 = 51. So 340+51 = 381.",
        gold_category="A",
    ),
    GoldenExample(
        id="A2_arith_exp",
        problem="Find (x+2)^2.",
        ground_truth="x^2+4x+4",
        common_prefix="Expand (x+2)^2 step by step.",
        baseline_context="(x+2)^2 = x^2 + 2*2*x + 4 = x^2 + 4x + 4.",
        quant_context="(x+2)^2 = x^2 + 4.",
        gold_category="A",
    ),
    GoldenExample(
        id="B1_logic",
        problem="Is every even number > 2 composite?",
        ground_truth="Yes",
        common_prefix="Let n be even and n > 2. Then n = 2k for some k > 1.",
        baseline_context="Since 2 divides n and n > 2, n has a proper divisor, so n is composite.",
        quant_context="Since 2 divides n, n is composite. But wait, 2 is even and prime, so the claim is false.",
        gold_category="B",
    ),
    GoldenExample(
        id="C1_strategy",
        problem="Solve x^2 - 5x + 6 = 0.",
        ground_truth="x=2 or x=3",
        common_prefix="By Vieta's: sum of roots is 5, product is 6.",
        baseline_context="Roots are 2 and 3 since 2+3=5 and 2*3=6.",
        quant_context="Actually let me drop Vieta's and try the quadratic formula from scratch with different numbers.",
        gold_category="C",
    ),
    GoldenExample(
        id="D1_halluc",
        problem="Factor 143.",
        ground_truth="11 * 13",
        common_prefix="I need to find prime factors of 143.",
        baseline_context="143 / 11 = 13. So 143 = 11 * 13.",
        quant_context="By the Smith-Jones divisibility theorem of 2017, 143 factors as 7 * 21.",
        gold_category="D",
    ),
    GoldenExample(
        id="E1_trunc",
        problem="Sum 1+2+...+10.",
        ground_truth="55",
        common_prefix="I'll use the formula n(n+1)/2 with n=10.",
        baseline_context="10*11/2 = 55.",
        quant_context="10*11/2 = ... and then",
        gold_category="E",
    ),
    GoldenExample(
        id="F1_loop",
        problem="Find the smallest prime > 10.",
        ground_truth="11",
        common_prefix="Candidates: 11, 12, 13, ...",
        baseline_context="11 is prime. Answer: 11.",
        quant_context="Let me try 11. Let me try 11. Let me try 11. Let me try 11. Let me try 11.",
        gold_category="F",
    ),
    GoldenExample(
        id="A3_arith_frac",
        problem="Compute 1/2 + 1/3.",
        ground_truth="5/6",
        common_prefix="Common denominator is 6.",
        baseline_context="3/6 + 2/6 = 5/6.",
        quant_context="3/6 + 2/6 = 6/6 = 1.",
        gold_category="A",
    ),
    GoldenExample(
        id="B2_logic",
        problem="If p implies q, and q, does p follow?",
        ground_truth="No (affirming the consequent).",
        common_prefix="Given: p → q and q.",
        baseline_context="We cannot conclude p; this is affirming the consequent.",
        quant_context="From q and p → q we conclude p.",
        gold_category="B",
    ),
    GoldenExample(
        id="F2_loop",
        problem="Solve 2x=4.",
        ground_truth="x=2",
        common_prefix="Divide both sides by 2.",
        baseline_context="x = 2.",
        quant_context="Divide by 2. Divide by 2. Divide by 2. Divide by 2. Divide by 2.",
        gold_category="F",
    ),
)
