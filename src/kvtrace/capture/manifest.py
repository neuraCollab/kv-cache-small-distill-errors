"""Run metadata and skip-log writers."""
from __future__ import annotations

import datetime
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import transformers


@dataclass
class SkipEntry:
    problem_id: int
    quant: str
    mode: str
    reason: str


class RunManifest:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    def write_metadata(
        self,
        *,
        model: str,
        model_revision_hash: str,
        git_commit: str | None = None,
        quants: list[str],
        modes: list[str],
        window_pre: int,
        window_post: int,
    ) -> None:
        meta = {
            "model": model,
            "model_revision_hash": model_revision_hash,
            "git_commit": git_commit or _get_git_commit() or "unknown",
            "quants": quants,
            "modes": modes,
            "window_pre": window_pre,
            "window_post": window_post,
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        (self.output_dir / "_run_metadata.json").write_text(json.dumps(meta, indent=2))

    def log_skip(self, entry: SkipEntry) -> None:
        path = self.output_dir / "_skipped.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")


def _get_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
