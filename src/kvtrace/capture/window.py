"""FDP-window slicing logic.

Window = [fdp_idx - pre, fdp_idx + post] inclusive (Python slice [ws:we]
where we = fdp_idx + post + 1). Truncation flags fire when window hits
trace boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    ws: int                # inclusive start
    we: int                # exclusive end (Python slice convention)
    truncated_left: bool
    truncated_right: bool

    @property
    def size(self) -> int:
        return self.we - self.ws


def compute_window(fdp_idx: int, trace_len: int, pre: int, post: int) -> Window:
    """Compute inclusive window [fdp_idx - pre, fdp_idx + post]."""
    if trace_len < 1:
        raise ValueError(f"trace_len must be >= 1, got {trace_len}")
    if fdp_idx < 0:
        raise ValueError(f"fdp_idx must be >= 0, got {fdp_idx}")
    if fdp_idx >= trace_len:
        raise ValueError(f"fdp_idx ({fdp_idx}) >= trace_len ({trace_len})")

    raw_ws = fdp_idx - pre
    raw_we = fdp_idx + post + 1  # exclusive end → +1 для inclusive [fdp_idx+post]

    ws = max(0, raw_ws)
    we = min(trace_len, raw_we)

    return Window(
        ws=ws,
        we=we,
        truncated_left=(raw_ws < 0),
        truncated_right=(raw_we > trace_len),
    )
