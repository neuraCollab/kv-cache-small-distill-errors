from unittest.mock import MagicMock

import pytest

from kvtrace.fdp.finder import FDPRecord, find_fdp, FDPParams


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(65 + (i % 26)) for i in ids)


class NoCosmeticChecker:
    def is_cosmetic_divergence(self, *a, **kw):
        return False


class AllCosmeticChecker:
    def is_cosmetic_divergence(self, *a, **kw):
        return True


def test_identical_traces_no_fdp():
    r = find_fdp(
        baseline_tokens=[1, 2, 3],
        quant_tokens=[1, 2, 3],
        baseline_text="same",
        quant_text="same",
        tokenizer=FakeTokenizer(),
        resync_checker=NoCosmeticChecker(),
    )
    assert r.fdp_token_idx is None


def test_first_token_divergence():
    r = find_fdp(
        baseline_tokens=[1, 2, 3, 4, 5],
        quant_tokens=[1, 2, 9, 4, 5],
        baseline_text="abcde",
        quant_text="abiZe",
        tokenizer=FakeTokenizer(),
        resync_checker=NoCosmeticChecker(),
    )
    assert r.fdp_token_idx == 2


def test_cosmetic_divergence_skipped_then_no_more():
    # Traces diverge at token 2, but checker flags as cosmetic.
    # After skip there is no further divergence → FDP is None but cosmetic_skipped=1.
    r = find_fdp(
        baseline_tokens=[1, 2, 3, 4, 5],
        quant_tokens=[1, 2, 9, 4, 5],
        baseline_text="abcde",
        quant_text="abiZe",
        tokenizer=FakeTokenizer(),
        resync_checker=AllCosmeticChecker(),
    )
    assert r.cosmetic_skipped >= 1
    # With max_cosmetic_skips default 5 and only one cosmetic point, FDP=None
    assert r.fdp_token_idx is None or r.fdp_token_idx > 2


def test_truncated_baseline_flag():
    # Baseline is shorter — hit max_tokens (finish_reason=length).
    r = find_fdp(
        baseline_tokens=[1, 2, 3],
        quant_tokens=[1, 2, 3, 4, 5],
        baseline_text="abc",
        quant_text="abcde",
        tokenizer=FakeTokenizer(),
        resync_checker=NoCosmeticChecker(),
        baseline_truncated=True,
    )
    assert r.fdp_token_idx == 3
    assert r.baseline_truncated is True


def test_boxed_match_attributes_are_captured():
    r = find_fdp(
        baseline_tokens=[1, 2],
        quant_tokens=[1, 9],
        baseline_text="ab",
        quant_text="ai",
        tokenizer=FakeTokenizer(),
        resync_checker=NoCosmeticChecker(),
        baseline_boxed="42",
        quant_boxed="41",
        ground_truth="42",
    )
    assert r.boxed_match == "baseline_only"


def test_max_cosmetic_skips_respected():
    # Always-cosmetic checker would loop forever without a cap.
    r = find_fdp(
        baseline_tokens=list(range(100)),
        quant_tokens=[i if i < 5 else (i + 1) for i in range(100)],
        baseline_text="x",
        quant_text="y",
        tokenizer=FakeTokenizer(),
        resync_checker=AllCosmeticChecker(),
        params=FDPParams(max_cosmetic_skips=2, resync_lookahead=10, context_window=3),
    )
    # After max skips, the first remaining mismatch becomes the real FDP.
    assert r.cosmetic_skipped == 2
    assert r.fdp_token_idx is not None


def test_context_window_clips_at_start():
    r = find_fdp(
        baseline_tokens=[1, 9, 3, 4],
        quant_tokens=[2, 9, 3, 4],
        baseline_text="abcd",
        quant_text="Xbcd",
        tokenizer=FakeTokenizer(),
        resync_checker=NoCosmeticChecker(),
        params=FDPParams(context_window=10, resync_lookahead=10, max_cosmetic_skips=0),
    )
    # FDP at index 0; baseline_context must not crash even with huge context.
    assert r.fdp_token_idx == 0
    assert isinstance(r.baseline_context, str)
    assert isinstance(r.quant_context, str)


def test_both_truncated_before_fdp():
    # Both finish by length but traces identical up to their shared tail.
    r = find_fdp(
        baseline_tokens=[1, 2, 3],
        quant_tokens=[1, 2, 3],
        baseline_text="abc",
        quant_text="abc",
        tokenizer=FakeTokenizer(),
        resync_checker=NoCosmeticChecker(),
        baseline_truncated=True,
        quant_truncated=True,
    )
    assert r.fdp_token_idx is None
    assert r.baseline_truncated and r.quant_truncated
