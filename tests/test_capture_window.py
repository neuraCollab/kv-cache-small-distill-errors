"""Edge cases для FDP-window slicing."""
from __future__ import annotations

import pytest

from kvtrace.capture.window import Window, compute_window


def test_centered_full_window():
    """FDP далеко от границ → полные 251 позиций."""
    w = compute_window(fdp_idx=500, trace_len=2000, pre=150, post=100)
    assert w.ws == 350
    assert w.we == 601
    assert w.size == 251
    assert not w.truncated_left
    assert not w.truncated_right


def test_fdp_near_start_truncated_left():
    """FDP=50, pre=150 → ws=0, truncated_left=True."""
    w = compute_window(fdp_idx=50, trace_len=2000, pre=150, post=100)
    assert w.ws == 0
    assert w.we == 151
    assert w.size == 151
    assert w.truncated_left
    assert not w.truncated_right


def test_fdp_near_end_truncated_right():
    """FDP=1980, trace_len=2000, post=100 → we=2000, truncated_right=True."""
    w = compute_window(fdp_idx=1980, trace_len=2000, pre=150, post=100)
    assert w.ws == 1830
    assert w.we == 2000
    assert w.truncated_right


def test_fdp_at_zero():
    w = compute_window(fdp_idx=0, trace_len=500, pre=150, post=100)
    assert w.ws == 0
    assert w.we == 101
    assert w.truncated_left


def test_fdp_at_last_position():
    """FDP=999, trace_len=1000 — последняя валидная позиция."""
    w = compute_window(fdp_idx=999, trace_len=1000, pre=150, post=100)
    assert w.ws == 849
    assert w.we == 1000
    assert w.truncated_right


def test_invalid_fdp_negative():
    with pytest.raises(ValueError, match="fdp_idx must be >= 0"):
        compute_window(fdp_idx=-1, trace_len=500, pre=150, post=100)


def test_invalid_fdp_beyond_trace():
    with pytest.raises(ValueError, match="fdp_idx .* >= trace_len"):
        compute_window(fdp_idx=500, trace_len=500, pre=150, post=100)


def test_zero_length_trace():
    with pytest.raises(ValueError, match="trace_len must be >= 1"):
        compute_window(fdp_idx=0, trace_len=0, pre=150, post=100)
