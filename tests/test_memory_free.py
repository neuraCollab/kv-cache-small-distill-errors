from unittest.mock import MagicMock, patch

import pytest

from kvtrace.memory import free_gpu, _destroy_vllm_parallel_state


def test_free_gpu_calls_torch_and_gc(monkeypatch):
    calls = []
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.empty_cache.side_effect = lambda: calls.append("empty_cache")
    fake_gc = MagicMock()
    fake_gc.collect.side_effect = lambda: calls.append("gc.collect")

    with patch.dict("sys.modules", {"torch": fake_torch, "gc": fake_gc}):
        import importlib
        import kvtrace.memory as m
        importlib.reload(m)
        m.free_gpu()

    assert "empty_cache" in calls
    assert "gc.collect" in calls


def test_destroy_vllm_parallel_state_swallows_import_error(monkeypatch):
    # If vLLM isn't installed, this must not raise.
    monkeypatch.setitem(__import__("sys").modules, "vllm", None)
    _destroy_vllm_parallel_state()  # should not raise


@pytest.mark.gpu
def test_free_gpu_actually_frees():
    import torch
    assert torch.cuda.is_available()
    x = torch.randn(1024, 1024, device="cuda")
    before = torch.cuda.memory_allocated()
    del x
    free_gpu()
    after = torch.cuda.memory_allocated()
    assert after < before
