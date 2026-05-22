"""Unit tests for kvtrace.capture.analysis."""
from __future__ import annotations

import torch

from kvtrace.capture.analysis import (
    align_captures_by_absolute_position,
    bf16_margin_trajectory,
    captures_share_window,
    compute_kv_value_stats_per_layer,
    compute_logits_kl_at_fdp,
    compute_relative_quant_error,
    logit_kl_trajectory,
    per_position_kv_quant_noise,
    top_outlier_channels,
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


def test_logit_kl_trajectory_shape_and_zero_for_identical():
    cap_a = _make_cap(seed=0)
    cap_b = _make_cap(seed=0)  # identical
    kl = logit_kl_trajectory(cap_a, cap_b)
    assert kl.shape == (10,)  # W
    # Identical logits → KL ≈ 0 (allow tiny float noise)
    assert kl.abs().max() < 1e-3


def test_logit_kl_trajectory_nonzero_for_different():
    cap_a = _make_cap(seed=0)
    cap_b = _make_cap(seed=1)
    kl = logit_kl_trajectory(cap_a, cap_b)
    assert kl.shape == (10,)
    assert kl.max() > 0


def test_logit_kl_trajectory_raises_on_different_windows():
    cap_a = _make_cap(window_start=0)
    cap_b = _make_cap(window_start=100)
    try:
        logit_kl_trajectory(cap_a, cap_b)
        assert False, "should have raised"
    except ValueError:
        pass


def test_bf16_margin_trajectory():
    cap = _make_cap(W=10, vocab=32)
    margin = bf16_margin_trajectory(cap)
    assert margin.shape == (10,)
    # Margin = top1 - top2 ≥ 0 always
    assert (margin >= 0).all()


def test_per_position_kv_quant_noise_zero_for_bf16():
    cap = _make_cap(quant="bf16")
    noise = per_position_kv_quant_noise(cap)
    assert noise["k_noise"].shape == (2, 10)  # [n_layers, W]
    assert noise["v_noise"].shape == (2, 10)
    assert noise["k_noise"].max() == 0
    assert noise["v_noise"].max() == 0


def test_per_position_kv_quant_noise_nonzero_for_quant():
    cap = _make_cap(quant="fp8_e4m3")
    noise = per_position_kv_quant_noise(cap)
    assert noise["k_noise"].max() > 0
    assert noise["v_noise"].max() > 0


def test_top_outlier_channels_finds_high_values():
    cap = _make_cap()
    # Inject extreme values in layer 0, head 0, channel 0
    cap.k_pre[0][:, 0, 0] = 1000.0
    out = top_outlier_channels(cap, threshold=448.0, top_n_per_layer=3)
    # results contains entries per (layer, kind)
    layer0_k = next(r for r in out if r["layer"] == 0 and r["kind"] == "k")
    assert layer0_k["n_channels_above_threshold"] >= 1
    # The injected channel should be top
    assert layer0_k["top_channels"][0]["head"] == 0
    assert layer0_k["top_channels"][0]["channel"] == 0
    assert layer0_k["top_channels"][0]["max_abs"] >= 1000.0
