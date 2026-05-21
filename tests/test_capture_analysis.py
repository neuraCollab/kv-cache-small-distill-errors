"""Unit tests for kvtrace.capture.analysis."""
from __future__ import annotations

import torch

from kvtrace.capture.analysis import (
    align_captures_by_absolute_position,
    captures_share_window,
    compute_kv_value_stats_per_layer,
    compute_logits_kl_at_fdp,
    compute_relative_quant_error,
)
from kvtrace.capture.storage import CaptureData


def _make_cap(
    W: int = 10,
    n_layers: int = 2,
    n_kv_heads: int = 4,
    head_dim: int = 8,
    vocab: int = 32,
    quant: str = "bf16",
    window_start: int = 0,
    fdp: int = 5,
    seed: int = 0,
) -> CaptureData:
    g = torch.Generator().manual_seed(seed)
    q = [torch.randn(W, n_kv_heads * head_dim, dtype=torch.float16, generator=g)
         for _ in range(n_layers)]
    k_pre = [torch.randn(W, n_kv_heads, head_dim, dtype=torch.float16, generator=g)
             for _ in range(n_layers)]
    v_pre = [torch.randn(W, n_kv_heads, head_dim, dtype=torch.float16, generator=g)
             for _ in range(n_layers)]
    if quant == "bf16":
        k_post = [t.clone() for t in k_pre]
        v_post = [t.clone() for t in v_pre]
    else:
        # Simulate quant error: add small noise
        k_post = [t + 0.01 * torch.randn_like(t) for t in k_pre]
        v_post = [t + 0.01 * torch.randn_like(t) for t in v_pre]
    logits = torch.randn(W, vocab, dtype=torch.float16, generator=g)
    return CaptureData(
        meta={
            "model": "test", "quant": quant, "mode": "tf", "problem_id": 0,
            "fdp_token_idx": fdp,
            "window_start": window_start, "window_end": window_start + W, "W": W,
            "input_token_ids": list(range(W)), "gen_token_ids": list(range(W)),
            "truncated_left": False, "truncated_right": False, "early_eos": False,
            "pytorch_version": "test", "transformers_version": "test",
            "model_revision_hash": "test", "run_timestamp": "test",
        },
        q=q, k_pre=k_pre, v_pre=v_pre, k_post=k_post, v_post=v_post, logits=logits,
    )


def test_relative_quant_error_bf16_zero():
    cap = _make_cap(quant="bf16")
    err = compute_relative_quant_error(cap)
    assert err.k_relative_error.max().item() == 0.0
    assert err.v_relative_error.max().item() == 0.0
    assert err.k_relative_error.shape == (2, 4)


def test_relative_quant_error_quant_nonzero():
    cap = _make_cap(quant="fp8_e4m3")
    err = compute_relative_quant_error(cap)
    assert err.k_relative_error.max().item() > 0.0
    assert err.v_relative_error.max().item() > 0.0


def test_logits_kl_same_window():
    cap_a = _make_cap(seed=0, quant="bf16")
    cap_b = _make_cap(seed=1, quant="fp8_e4m3")
    out = compute_logits_kl_at_fdp(cap_a, cap_b)
    assert "kl_a_to_b" in out
    assert out["js_divergence"] >= 0
    assert isinstance(out["top1_match"], bool)


def test_logits_kl_different_window_raises():
    cap_a = _make_cap(window_start=0, fdp=5)
    cap_b = _make_cap(window_start=100, fdp=105)
    try:
        compute_logits_kl_at_fdp(cap_a, cap_b)
        assert False, "should have raised"
    except ValueError:
        pass


def test_kv_value_stats_keys():
    cap = _make_cap()
    stats = compute_kv_value_stats_per_layer(cap)
    assert len(stats) == 2  # n_layers
    expected_keys = {
        "layer", "k_max_abs", "k_mean_abs", "k_std",
        "k_outliers_pct_448", "k_outliers_pct_57344",
        "v_max_abs", "v_mean_abs", "v_std",
        "v_outliers_pct_448", "v_outliers_pct_57344",
    }
    assert set(stats[0].keys()) == expected_keys


def test_captures_share_window():
    cap_a = _make_cap(window_start=0, fdp=5)
    cap_b = _make_cap(window_start=0, fdp=5)
    cap_c = _make_cap(window_start=100, fdp=105)
    assert captures_share_window(cap_a, cap_b)
    assert not captures_share_window(cap_a, cap_c)


def test_align_captures_by_absolute_position():
    # window_a = [0, 10), window_b = [5, 15) → overlap [5, 10)
    cap_a = _make_cap(W=10, window_start=0)
    cap_b = _make_cap(W=10, window_start=5)
    sl_a, sl_b = align_captures_by_absolute_position(cap_a, cap_b)
    assert sl_a == slice(5, 10)
    assert sl_b == slice(0, 5)


def test_align_captures_no_overlap():
    cap_a = _make_cap(W=10, window_start=0)
    cap_b = _make_cap(W=10, window_start=20)
    assert align_captures_by_absolute_position(cap_a, cap_b) is None
