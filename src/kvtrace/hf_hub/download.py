"""Download a dataset file pinned to a revision tag."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kvtrace.hf_hub.upload import resolve_repo_id

log = logging.getLogger(__name__)


def download_dataset_file(
    filename: str,
    *,
    revision_tag: str,
    api: Any | None = None,
    repo_id: str | None = None,
    local_dir: Path | str | None = None,
) -> Path | None:
    repo_id = repo_id or resolve_repo_id()
    if not repo_id:
        return None
    if api is None:  # pragma: no cover
        from huggingface_hub import HfApi
        api = HfApi()

    local_path = api.hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        revision=revision_tag,
        local_dir=str(local_dir) if local_dir else None,
    )
    return Path(local_path)
