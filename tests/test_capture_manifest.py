"""Тесты для _run_metadata.json и _skipped.jsonl writers."""
from __future__ import annotations

import json
from pathlib import Path

from kvtrace.capture.manifest import RunManifest, SkipEntry


def test_run_manifest_writes_metadata(tmp_path: Path):
    m = RunManifest(output_dir=tmp_path)
    m.write_metadata(
        model="qwen3-1.7b",
        model_revision_hash="abc123",
        git_commit="def456",
        quants=["bf16", "fp8_e4m3", "fp8_e5m2"],
        modes=["tf", "ar"],
        window_pre=150,
        window_post=100,
    )
    meta_path = tmp_path / "_run_metadata.json"
    assert meta_path.exists()
    data = json.loads(meta_path.read_text())
    assert data["model"] == "qwen3-1.7b"
    assert data["model_revision_hash"] == "abc123"
    assert data["git_commit"] == "def456"
    assert "run_timestamp" in data
    assert "pytorch_version" in data
    assert "transformers_version" in data


def test_skip_log_appends_entries(tmp_path: Path):
    m = RunManifest(output_dir=tmp_path)
    m.log_skip(SkipEntry(problem_id=5, quant="fp8_e4m3", mode="ar", reason="prefix_too_long"))
    m.log_skip(SkipEntry(problem_id=8, quant="bf16", mode="tf", reason="no_fdp"))

    skip_path = tmp_path / "_skipped.jsonl"
    assert skip_path.exists()
    lines = skip_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "problem_id": 5, "quant": "fp8_e4m3", "mode": "ar", "reason": "prefix_too_long"
    }
    assert json.loads(lines[1])["reason"] == "no_fdp"
