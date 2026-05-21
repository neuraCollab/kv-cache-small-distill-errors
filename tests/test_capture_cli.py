"""CLI smoke-тесты для scripts/06_capture_kv.py через subprocess."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_dry_run_prints_plan_without_running(tmp_path: Path, fixtures_dir: Path):
    """--dry-run должен напечатать (problem, quant, mode) комбинации без forward'ов."""
    # Мини-фикстуры: 2 problem records, 2 FDP records (e4m3 only)
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    (traces_dir / "qwen3-1.7b_bf16.jsonl").write_text(
        '\n'.join([
            json.dumps({"idx": 0, "token_ids": list(range(300))}),
            json.dumps({"idx": 1, "token_ids": list(range(300))}),
        ]) + "\n"
    )
    fdps_dir = tmp_path / "fdps"
    fdps_dir.mkdir()
    (fdps_dir / "qwen3-1.7b_fp8_e4m3.jsonl").write_text(
        '\n'.join([
            json.dumps({"problem_idx": 0, "fdp_token_idx": 100}),
            json.dumps({"problem_idx": 1, "fdp_token_idx": 150}),
        ]) + "\n"
    )

    result = subprocess.run(
        [sys.executable, "scripts/06_capture_kv.py",
         "--model", "qwen3-1.7b",
         "--quants", "bf16", "fp8_e4m3",
         "--modes", "tf",
         "--traces-dir", str(traces_dir),
         "--fdps-dir", str(fdps_dir),
         "--output-dir", str(tmp_path / "out"),
         "--dry-run"],
        capture_output=True, text=True, check=False,
        cwd="C:/Users/morro/prog/files",
    )
    assert result.returncode == 0, result.stderr
    assert "problem=0" in result.stdout
    assert "problem=1" in result.stdout
    assert "quant=bf16" in result.stdout
    assert "quant=fp8_e4m3" in result.stdout
    assert "Total captures planned: 4" in result.stdout  # 2 problems × 2 quants × 1 mode
