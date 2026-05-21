"""Roundtrip-тесты для CaptureData ↔ safetensors."""
from __future__ import annotations

from pathlib import Path

import torch

from kvtrace.capture.storage import CaptureData, load_capture, save_capture


def _make_dummy_capture(W=10, n_layers=2, n_q_heads=4, n_kv_heads=2, head_dim=8, vocab=100):
    return CaptureData(
        meta={
            "model": "test",
            "quant": "fp8_e4m3",
            "mode": "tf",
            "problem_id": 0,
            "fdp_token_idx": 5,
            "window_start": 0,
            "window_end": W,
            "W": W,
            "input_token_ids": list(range(W)),
            "gen_token_ids": list(range(W)),
            "truncated_left": False,
            "truncated_right": False,
            "early_eos": False,
            "pytorch_version": torch.__version__,
            "transformers_version": "test",
            "model_revision_hash": "deadbeef",
            "run_timestamp": "2026-05-21T00:00:00Z",
        },
        q=[torch.randn(W, n_q_heads, head_dim, dtype=torch.float16) for _ in range(n_layers)],
        k_pre=[torch.randn(W, n_kv_heads, head_dim, dtype=torch.float16) for _ in range(n_layers)],
        v_pre=[torch.randn(W, n_kv_heads, head_dim, dtype=torch.float16) for _ in range(n_layers)],
        k_post=[torch.randn(W, n_kv_heads, head_dim, dtype=torch.float16) for _ in range(n_layers)],
        v_post=[torch.randn(W, n_kv_heads, head_dim, dtype=torch.float16) for _ in range(n_layers)],
        logits=torch.randn(W, vocab, dtype=torch.float16),
    )


def test_roundtrip_preserves_shapes_and_values(tmp_path: Path):
    cap = _make_dummy_capture()
    out = tmp_path / "cap.safetensors"
    save_capture(cap, out)
    loaded = load_capture(out)

    assert loaded.meta == cap.meta
    assert len(loaded.q) == len(cap.q)
    for orig, got in zip(cap.q, loaded.q):
        assert torch.equal(orig, got)
    for orig, got in zip(cap.k_pre, loaded.k_pre):
        assert torch.equal(orig, got)
    for orig, got in zip(cap.v_post, loaded.v_post):
        assert torch.equal(orig, got)
    assert torch.equal(cap.logits, loaded.logits)


def test_roundtrip_preserves_dtype(tmp_path: Path):
    cap = _make_dummy_capture()
    out = tmp_path / "cap.safetensors"
    save_capture(cap, out)
    loaded = load_capture(out)
    assert loaded.q[0].dtype == torch.float16
    assert loaded.logits.dtype == torch.float16


def test_save_creates_parent_dirs(tmp_path: Path):
    cap = _make_dummy_capture()
    out = tmp_path / "a" / "b" / "c" / "cap.safetensors"
    save_capture(cap, out)
    assert out.exists()


def test_meta_includes_nested_lists(tmp_path: Path):
    cap = _make_dummy_capture(W=3)
    out = tmp_path / "cap.safetensors"
    save_capture(cap, out)
    loaded = load_capture(out)
    assert loaded.meta["input_token_ids"] == [0, 1, 2]
