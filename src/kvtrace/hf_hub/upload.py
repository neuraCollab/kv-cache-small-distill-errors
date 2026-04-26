"""Idempotent HF Hub dataset upload, keyed by revision tag."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def resolve_repo_id() -> str | None:
    """HF_REPO_ID overrides HF_USER; returns None if neither is set."""
    env_id = os.environ.get("HF_REPO_ID", "").strip()
    if env_id:
        return env_id
    user = os.environ.get("HF_USER", "").strip()
    if user:
        return f"{user}/kv-trace-study"
    return None


def _branch_exists(api: Any, repo_id: str, branch: str) -> bool:
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset")
    except Exception as e:  # pragma: no cover — network path not exercised in CI
        log.warning("list_repo_refs failed: %s", e)
        return False
    for b in getattr(refs, "branches", []) or []:
        if getattr(b, "name", None) == branch:
            return True
    return False


def _ensure_dataset_repo(api: Any, repo_id: str) -> bool:
    """Idempotently create the dataset repo. Returns True if it now exists."""
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        return True
    except Exception as e:
        log.warning("create_repo failed for %s: %s", repo_id, e)
        return False


def upload_dataset_file(
    local_path: Path,
    *,
    revision_tag: str,
    api: Any | None = None,
    repo_id: str | None = None,
    path_in_repo: str | None = None,
) -> bool:
    """Upload a file to a HuggingFace dataset repo under a revision branch.

    Best-effort: never raises. Returns True if an upload was performed;
    False if skipped (no repo configured, repo creation failed, revision
    already present, or any upload-time error). Trace generation is the
    expensive step — a failed upload must not crash the pipeline.
    """
    repo_id = repo_id or resolve_repo_id()
    if not repo_id:
        log.info("no HF_REPO_ID / HF_USER; skipping upload of %s", local_path)
        return False

    if api is None:  # pragma: no cover — import path
        from huggingface_hub import HfApi
        api = HfApi()

    # First-run convenience: auto-create the dataset repo if it doesn't
    # exist yet. Without this, list_repo_refs / create_branch return 404
    # and the caller sees a stack trace mid-pipeline.
    if not _ensure_dataset_repo(api, repo_id):
        log.warning("could not ensure repo %s exists; skipping upload of %s", repo_id, local_path)
        return False

    if _branch_exists(api, repo_id, revision_tag):
        log.info("revision %s already on %s; skipping upload", revision_tag, repo_id)
        return False

    try:
        api.create_branch(
            repo_id=repo_id, repo_type="dataset", branch=revision_tag, exist_ok=True
        )
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo or local_path.name,
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision_tag,
        )
    except Exception as e:
        log.warning("upload failed for %s -> %s@%s: %s", local_path, repo_id, revision_tag, e)
        return False
    log.info("uploaded %s -> %s@%s", local_path, repo_id, revision_tag)
    return True
