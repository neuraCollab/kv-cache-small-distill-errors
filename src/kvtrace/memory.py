"""GPU memory cleanup helpers."""
from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from contextlib import contextmanager

log = logging.getLogger(__name__)


def _destroy_vllm_parallel_state() -> None:
    """Call vLLM's destroy_model_parallel if it is importable. No-op otherwise."""
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
    except Exception:
        return
    try:
        destroy_model_parallel()
    except Exception as e:  # pragma: no cover — safety net
        log.warning("destroy_model_parallel raised: %s", e)


def free_gpu() -> None:
    """Release vLLM engine state + torch cache + run GC.

    Call this between (model, config) pairs in Phase 1. Idempotent.
    """
    _destroy_vllm_parallel_state()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:  # pragma: no cover
        log.warning("torch.cuda.empty_cache failed: %s", e)
    gc.collect()


@contextmanager
def gpu_scope() -> Iterator[None]:
    """Use as `with gpu_scope(): ...` — frees GPU on exit even on exception."""
    try:
        yield
    finally:
        free_gpu()
