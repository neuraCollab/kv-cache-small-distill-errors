"""Hybrid First Divergence Point finder.

Token-level exact mismatch, then semantic re-sync to filter out cosmetic
divergences, then report the true FDP with ±context windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from kvtrace.fdp.tokenizer_align import decode_window, first_token_mismatch

BoxedMatch = Literal["both_correct", "baseline_only", "quant_only", "both_wrong", "no_boxed"]


class ReSyncProtocol(Protocol):
    def is_cosmetic_divergence(self, baseline_window: str, quant_window: str) -> bool: ...


@dataclass
class FDPParams:
    context_window: int = 200
    resync_lookahead: int = 500
    max_cosmetic_skips: int = 5


@dataclass
class FDPRecord:
    fdp_token_idx: int | None
    cosmetic_skipped: int
    baseline_context: str
    quant_context: str
    common_prefix: str
    boxed_match: BoxedMatch = "no_boxed"
    baseline_truncated: bool = False
    quant_truncated: bool = False
    metadata: dict = field(default_factory=dict)


def _boxed_match(baseline: str | None, quant: str | None, gt: str | None) -> BoxedMatch:
    if baseline is None and quant is None:
        return "no_boxed"
    gt_s = (gt or "").strip()
    b_ok = baseline is not None and baseline.strip() == gt_s
    q_ok = quant is not None and quant.strip() == gt_s
    if b_ok and q_ok:
        return "both_correct"
    if b_ok and not q_ok:
        return "baseline_only"
    if q_ok and not b_ok:
        return "quant_only"
    return "both_wrong"


def find_fdp(
    baseline_tokens: list[int],
    quant_tokens: list[int],
    baseline_text: str,
    quant_text: str,
    tokenizer: Any,
    resync_checker: ReSyncProtocol,
    params: FDPParams | None = None,
    baseline_truncated: bool = False,
    quant_truncated: bool = False,
    baseline_boxed: str | None = None,
    quant_boxed: str | None = None,
    ground_truth: str | None = None,
) -> FDPRecord:
    p = params or FDPParams()
    cosmetic_skipped = 0
    offset = 0

    while True:
        idx = first_token_mismatch(baseline_tokens[offset:], quant_tokens[offset:])
        if idx is None:
            # No (more) divergence.
            return FDPRecord(
                fdp_token_idx=None,
                cosmetic_skipped=cosmetic_skipped,
                baseline_context="",
                quant_context="",
                common_prefix=tokenizer.decode(
                    baseline_tokens[: min(len(baseline_tokens), 300)],
                    skip_special_tokens=False,
                ),
                boxed_match=_boxed_match(baseline_boxed, quant_boxed, ground_truth),
                baseline_truncated=baseline_truncated,
                quant_truncated=quant_truncated,
            )
        absolute_idx = offset + idx

        # Budget check: too many cosmetic skips already — treat this as real FDP.
        if cosmetic_skipped >= p.max_cosmetic_skips:
            return _build_real_fdp(
                absolute_idx, baseline_tokens, quant_tokens, tokenizer, p,
                cosmetic_skipped, baseline_truncated, quant_truncated,
                baseline_boxed, quant_boxed, ground_truth,
            )

        # Build re-sync windows.
        b_window = decode_window(tokenizer, baseline_tokens, absolute_idx, p.resync_lookahead)
        q_window = decode_window(tokenizer, quant_tokens, absolute_idx, p.resync_lookahead)
        is_cosmetic = resync_checker.is_cosmetic_divergence(b_window, q_window)

        if not is_cosmetic:
            return _build_real_fdp(
                absolute_idx, baseline_tokens, quant_tokens, tokenizer, p,
                cosmetic_skipped, baseline_truncated, quant_truncated,
                baseline_boxed, quant_boxed, ground_truth,
            )

        # Cosmetic: skip forward by resync_lookahead tokens and keep searching.
        cosmetic_skipped += 1
        offset = absolute_idx + p.resync_lookahead


def _build_real_fdp(
    absolute_idx: int,
    baseline_tokens: list[int],
    quant_tokens: list[int],
    tokenizer: Any,
    p: FDPParams,
    cosmetic_skipped: int,
    baseline_truncated: bool,
    quant_truncated: bool,
    baseline_boxed: str | None,
    quant_boxed: str | None,
    ground_truth: str | None,
) -> FDPRecord:
    b_ctx = decode_window(tokenizer, baseline_tokens, absolute_idx, p.context_window)
    q_ctx = decode_window(tokenizer, quant_tokens, absolute_idx, p.context_window)
    prefix_end = max(0, absolute_idx)
    prefix_start = max(0, prefix_end - 300)
    common_prefix = tokenizer.decode(
        baseline_tokens[prefix_start:prefix_end], skip_special_tokens=False
    )
    return FDPRecord(
        fdp_token_idx=absolute_idx,
        cosmetic_skipped=cosmetic_skipped,
        baseline_context=b_ctx,
        quant_context=q_ctx,
        common_prefix=common_prefix,
        boxed_match=_boxed_match(baseline_boxed, quant_boxed, ground_truth),
        baseline_truncated=baseline_truncated,
        quant_truncated=quant_truncated,
    )
