# KV-matrix capture для Qwen3-1.7B на CPU — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать Phase 6 pipeline — захват полных тензоров Q, K (pre/post quant), V (pre/post quant) и logits в окне `[FDP−150, FDP+100]` для Qwen3-1.7B на CPU под bf16/fp8_e4m3/fp8_e5m2, в режимах teacher-forced и autoregressive.

**Architecture:** Forward-хуки на HF `Qwen3Attention`-блоках + monkey-patch `DynamicCache.update` для подмены K/V на quant→dequant версию. Симулятор FP8 — `tensor.to(torch.float8_e4m3fn).to(bf16)` (PyTorch native). Сериализация — `safetensors` по одному файлу на `(problem, quant, mode)`. CLI script `06_capture_kv.py` оркестрирует над существующими `outputs/traces/` и `outputs/fdps/`.

**Tech Stack:** Python 3.11+, PyTorch >= 2.1 (native fp8 dtypes), transformers 4.51.x (pinned, см. requirements.txt), safetensors, pytest. Никаких новых runtime-зависимостей.

**Spec:** [docs/superpowers/specs/2026-05-21-kv-matrix-capture-design.md](../specs/2026-05-21-kv-matrix-capture-design.md)

---

## File Structure

**Создаются:**
```
src/kvtrace/capture/
  __init__.py             # re-exports public API
  fp8_sim.py              # QUANT_FNS = {bf16, fp8_e4m3, fp8_e5m2}
  window.py               # compute_window(fdp_idx, T, pre, post)
  storage.py              # save_capture, load_capture, CaptureData dataclass
  attention_hooks.py      # install_capture_hooks(model, quant_fn) -> handle
  cpu_runner.py           # CaptureRunner: load_model, run_tf, run_ar
  manifest.py             # _run_metadata.json, _skipped.jsonl writers

scripts/
  06_capture_kv.py        # CLI entry point

tests/
  test_capture_fp8_sim.py
  test_capture_window.py
  test_capture_storage.py
  test_capture_hooks.py
  test_capture_runner_tf.py
  test_capture_runner_ar.py
  test_capture_manifest.py
  test_capture_cli.py
  test_capture_smoke.py                # @pytest.mark.slow
  test_capture_fp8_matches_vllm.py     # @pytest.mark.slow
  test_capture_sliding_window.py       # @pytest.mark.slow
  fixtures/
    qwen3_fdp_sample.json              # 2-3 FDP records для unit-тестов
    qwen3_trace_sample.json            # соответствующие токен-стримы
```

**Модифицируются:**
- `scripts/run_all.sh` — добавление Phase 6 (опционально, после получения первых результатов)

---

## Task 1: Module scaffold + sanity check

**Files:**
- Create: `src/kvtrace/capture/__init__.py`
- Verify: `requirements.txt` PyTorch availability

- [ ] **Step 1.1: Создать пустой пакет**

```python
# src/kvtrace/capture/__init__.py
"""KV-cache capture pipeline (Phase 6).

Captures Q, K_pre, K_post, V_pre, V_post, logits during forward pass on CPU
under bf16 / fp8_e4m3 / fp8_e5m2 KV-cache quantization. See
docs/superpowers/specs/2026-05-21-kv-matrix-capture-design.md.
"""
from __future__ import annotations
```

- [ ] **Step 1.2: Проверить, что torch >= 2.1 доступен (фактически — naличие fp8 dtypes)**

Run:
```bash
python -c "import torch; assert hasattr(torch, 'float8_e4m3fn') and hasattr(torch, 'float8_e5m2'), f'PyTorch {torch.__version__} missing fp8'"
```
Expected: silent success. Если fail — установить `torch>=2.1` через pip.

- [ ] **Step 1.3: Существующие тесты не сломались**

Run: `pytest tests/ -x --ignore=tests/test_generators_vllm.py -q`
Expected: PASS (vLLM-specific тесты могут не работать на CPU-only машине, поэтому игнорируем).

- [ ] **Step 1.4: Commit**

```bash
git add src/kvtrace/capture/__init__.py
git commit -m "feat(capture): scaffold capture module for Phase 6"
```

---

## Task 2: FP8 simulators

**Files:**
- Create: `src/kvtrace/capture/fp8_sim.py`
- Test: `tests/test_capture_fp8_sim.py`

- [ ] **Step 2.1: Написать падающий тест**

```python
# tests/test_capture_fp8_sim.py
"""Unit-tests для FP8 quant→dequant симулятора (pure-PyTorch CPU)."""
from __future__ import annotations

import torch

from kvtrace.capture.fp8_sim import QUANT_FNS, fp8_e4m3, fp8_e5m2


def test_bf16_identity():
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    assert torch.equal(QUANT_FNS["bf16"](x), x)


def test_fp8_e4m3_idempotent():
    """qd(qd(x)) == qd(x): второе применение не меняет уже-в-fp8 значения."""
    x = torch.randn(64, dtype=torch.bfloat16)
    once = fp8_e4m3(x)
    twice = fp8_e4m3(once)
    assert torch.equal(once, twice)


def test_fp8_e5m2_idempotent():
    x = torch.randn(64, dtype=torch.bfloat16)
    once = fp8_e5m2(x)
    twice = fp8_e5m2(once)
    assert torch.equal(once, twice)


def test_preserves_dtype():
    x = torch.randn(4, dtype=torch.bfloat16)
    assert fp8_e4m3(x).dtype == torch.bfloat16
    assert fp8_e5m2(x).dtype == torch.bfloat16


def test_changes_values_for_random_input():
    """Гарантирует, что quant→dequant НЕ identity для произвольных bf16 значений."""
    torch.manual_seed(0)
    x = torch.randn(4096, dtype=torch.bfloat16)
    assert not torch.equal(fp8_e4m3(x), x)
    assert not torch.equal(fp8_e5m2(x), x)


def test_e4m3_narrower_range_than_e5m2():
    """e4m3 имеет меньший max; большие значения должны клипаться сильнее."""
    big = torch.tensor([400.0, -400.0], dtype=torch.bfloat16)
    out_e4m3 = fp8_e4m3(big)
    out_e5m2 = fp8_e5m2(big)
    # e4m3 max ≈ 448, e5m2 max ≈ 57344 → e5m2 ближе к оригиналу
    assert (out_e5m2 - big).abs().sum() < (out_e4m3 - big).abs().sum()


def test_quant_fns_registry_keys():
    assert set(QUANT_FNS.keys()) == {"bf16", "fp8_e4m3", "fp8_e5m2"}
```

- [ ] **Step 2.2: Запустить тест — должен упасть на ImportError**

Run: `pytest tests/test_capture_fp8_sim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kvtrace.capture.fp8_sim'`

- [ ] **Step 2.3: Реализовать симулятор**

```python
# src/kvtrace/capture/fp8_sim.py
"""FP8 quant→dequant simulator (pure PyTorch, CPU-friendly).

Uses PyTorch's native float8_e4m3fn and float8_e5m2 dtypes (PyTorch >= 2.1).
Cast bf16 → fp8 → bf16 reproduces IEEE FP8 round-to-nearest-even, identical
to vLLM 0.7.3's default per-tensor KV-quant.

No per-channel scaling, no stochastic rounding — same defaults as the main
experiment (see config/quant_methods.yaml).
"""
from __future__ import annotations

from typing import Callable

import torch


def fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """bf16 → fp8_e4m3 → bf16. Narrow range (±448), high precision."""
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def fp8_e5m2(x: torch.Tensor) -> torch.Tensor:
    """bf16 → fp8_e5m2 → bf16. Wide range (±57344), lower precision."""
    return x.to(torch.float8_e5m2).to(x.dtype)


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


QUANT_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "bf16": _identity,
    "fp8_e4m3": fp8_e4m3,
    "fp8_e5m2": fp8_e5m2,
}
```

- [ ] **Step 2.4: Тесты должны пройти**

Run: `pytest tests/test_capture_fp8_sim.py -v`
Expected: 7 passed.

- [ ] **Step 2.5: Commit**

```bash
git add src/kvtrace/capture/fp8_sim.py tests/test_capture_fp8_sim.py
git commit -m "feat(capture): fp8_e4m3/fp8_e5m2 quant→dequant simulator"
```

---

## Task 3: Window slicing logic

**Files:**
- Create: `src/kvtrace/capture/window.py`
- Test: `tests/test_capture_window.py`

- [ ] **Step 3.1: Написать падающий тест**

```python
# tests/test_capture_window.py
"""Edge cases для FDP-window slicing."""
from __future__ import annotations

import pytest

from kvtrace.capture.window import Window, compute_window


def test_centered_full_window():
    """FDP далеко от границ → полные 251 позиций."""
    w = compute_window(fdp_idx=500, trace_len=2000, pre=150, post=100)
    assert w.ws == 350
    assert w.we == 601
    assert w.size == 251
    assert not w.truncated_left
    assert not w.truncated_right


def test_fdp_near_start_truncated_left():
    """FDP=50, pre=150 → ws=0, truncated_left=True."""
    w = compute_window(fdp_idx=50, trace_len=2000, pre=150, post=100)
    assert w.ws == 0
    assert w.we == 151
    assert w.size == 151
    assert w.truncated_left
    assert not w.truncated_right


def test_fdp_near_end_truncated_right():
    """FDP=1980, trace_len=2000, post=100 → we=2000, truncated_right=True."""
    w = compute_window(fdp_idx=1980, trace_len=2000, pre=150, post=100)
    assert w.ws == 1830
    assert w.we == 2000
    assert w.truncated_right


def test_fdp_at_zero():
    w = compute_window(fdp_idx=0, trace_len=500, pre=150, post=100)
    assert w.ws == 0
    assert w.we == 101
    assert w.truncated_left


def test_fdp_at_last_position():
    """FDP=999, trace_len=1000 — последняя валидная позиция."""
    w = compute_window(fdp_idx=999, trace_len=1000, pre=150, post=100)
    assert w.ws == 849
    assert w.we == 1000
    assert w.truncated_right


def test_invalid_fdp_negative():
    with pytest.raises(ValueError, match="fdp_idx must be >= 0"):
        compute_window(fdp_idx=-1, trace_len=500, pre=150, post=100)


def test_invalid_fdp_beyond_trace():
    with pytest.raises(ValueError, match="fdp_idx .* >= trace_len"):
        compute_window(fdp_idx=500, trace_len=500, pre=150, post=100)


def test_zero_length_trace():
    with pytest.raises(ValueError, match="trace_len must be >= 1"):
        compute_window(fdp_idx=0, trace_len=0, pre=150, post=100)
```

- [ ] **Step 3.2: Запустить тест — должен упасть**

Run: `pytest tests/test_capture_window.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3.3: Реализовать window.py**

```python
# src/kvtrace/capture/window.py
"""FDP-window slicing logic.

Window = [fdp_idx - pre, fdp_idx + post] inclusive (Python slice [ws:we]
where we = fdp_idx + post + 1). Truncation flags fire when window hits
trace boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    ws: int                # inclusive start
    we: int                # exclusive end (Python slice convention)
    truncated_left: bool
    truncated_right: bool

    @property
    def size(self) -> int:
        return self.we - self.ws


def compute_window(fdp_idx: int, trace_len: int, pre: int, post: int) -> Window:
    """Compute inclusive window [fdp_idx - pre, fdp_idx + post]."""
    if trace_len < 1:
        raise ValueError(f"trace_len must be >= 1, got {trace_len}")
    if fdp_idx < 0:
        raise ValueError(f"fdp_idx must be >= 0, got {fdp_idx}")
    if fdp_idx >= trace_len:
        raise ValueError(f"fdp_idx ({fdp_idx}) >= trace_len ({trace_len})")

    raw_ws = fdp_idx - pre
    raw_we = fdp_idx + post + 1  # exclusive end → +1 для inclusive [fdp_idx+post]

    ws = max(0, raw_ws)
    we = min(trace_len, raw_we)

    return Window(
        ws=ws,
        we=we,
        truncated_left=(raw_ws < 0),
        truncated_right=(raw_we > trace_len),
    )
```

- [ ] **Step 3.4: Тесты должны пройти**

Run: `pytest tests/test_capture_window.py -v`
Expected: 8 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/kvtrace/capture/window.py tests/test_capture_window.py
git commit -m "feat(capture): FDP-window slicing with truncation flags"
```

---

## Task 4: Storage layer (safetensors writer/reader)

**Files:**
- Create: `src/kvtrace/capture/storage.py`
- Test: `tests/test_capture_storage.py`

- [ ] **Step 4.1: Написать падающий тест**

```python
# tests/test_capture_storage.py
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
```

- [ ] **Step 4.2: Запустить тест — должен упасть**

Run: `pytest tests/test_capture_storage.py -v`
Expected: FAIL on import.

- [ ] **Step 4.3: Реализовать storage.py**

```python
# src/kvtrace/capture/storage.py
"""Safetensors-backed storage для CaptureData.

Layout (внутри файла):
  q_l{ℓ}, k_pre_l{ℓ}, v_pre_l{ℓ}, k_post_l{ℓ}, v_post_l{ℓ}  — per layer
  logits

Meta хранится в safetensors `__metadata__` как JSON-string (safetensors
требует str→str). Поэтому при load распарсиваем обратно из JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


@dataclass
class CaptureData:
    meta: dict[str, Any]
    q: list[torch.Tensor]            # per layer
    k_pre: list[torch.Tensor]
    v_pre: list[torch.Tensor]
    k_post: list[torch.Tensor]
    v_post: list[torch.Tensor]
    logits: torch.Tensor


def save_capture(cap: CaptureData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    n_layers = len(cap.q)
    for layer in range(n_layers):
        tensors[f"q_l{layer}"] = cap.q[layer].contiguous()
        tensors[f"k_pre_l{layer}"] = cap.k_pre[layer].contiguous()
        tensors[f"v_pre_l{layer}"] = cap.v_pre[layer].contiguous()
        tensors[f"k_post_l{layer}"] = cap.k_post[layer].contiguous()
        tensors[f"v_post_l{layer}"] = cap.v_post[layer].contiguous()
    tensors["logits"] = cap.logits.contiguous()

    meta_with_layers = {**cap.meta, "n_layers": n_layers}
    metadata_json = {"json": json.dumps(meta_with_layers)}
    save_file(tensors, str(path), metadata=metadata_json)


def load_capture(path: Path) -> CaptureData:
    # safetensors split — load tensors first, then header metadata via the
    # safe_open context. Simpler: load all + re-parse header.
    raw = load_file(str(path))

    # Read metadata via low-level API.
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        meta_str = f.metadata().get("json")
    if meta_str is None:
        raise ValueError(f"Capture file {path} missing 'json' metadata key")
    meta = json.loads(meta_str)
    n_layers = meta.pop("n_layers")

    q = [raw[f"q_l{ℓ}"] for ℓ in range(n_layers)]
    k_pre = [raw[f"k_pre_l{ℓ}"] for ℓ in range(n_layers)]
    v_pre = [raw[f"v_pre_l{ℓ}"] for ℓ in range(n_layers)]
    k_post = [raw[f"k_post_l{ℓ}"] for ℓ in range(n_layers)]
    v_post = [raw[f"v_post_l{ℓ}"] for ℓ in range(n_layers)]
    logits = raw["logits"]

    return CaptureData(meta=meta, q=q, k_pre=k_pre, v_pre=v_pre,
                       k_post=k_post, v_post=v_post, logits=logits)
```

- [ ] **Step 4.4: Тесты должны пройти**

Run: `pytest tests/test_capture_storage.py -v`
Expected: 4 passed.

- [ ] **Step 4.5: Commit**

```bash
git add src/kvtrace/capture/storage.py tests/test_capture_storage.py
git commit -m "feat(capture): CaptureData dataclass + safetensors roundtrip"
```

---

## Task 5: Attention hooks (toy-model unit tests)

**Files:**
- Create: `src/kvtrace/capture/attention_hooks.py`
- Test: `tests/test_capture_hooks.py`

- [ ] **Step 5.1: Написать падающий тест с фиктивной моделью**

```python
# tests/test_capture_hooks.py
"""Unit-тесты для attention hooks на минимальной игрушечной модели.

Цель: убедиться, что хук собирает Q/K_pre/K_post/V_pre/V_post в правильном
порядке и при подмене K/V через quant_fn возвращает quantized версию.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kvtrace.capture.attention_hooks import CaptureHandle, install_capture_hooks
from kvtrace.capture.fp8_sim import fp8_e4m3


class _FakeCache:
    """Минимальный DynamicCache-like: list[Tensor] для K и V."""
    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class _FakeAttention(nn.Module):
    """Attention block с явными Q/K/V для проверки хука."""
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, hidden_states, past_key_value=None):
        # Игрушечные Q/K/V — три проекции = identity для простоты
        bsz, seq, dim = hidden_states.shape
        q = hidden_states.view(bsz, seq, 1, dim).contiguous()
        k = hidden_states.view(bsz, seq, 1, dim).contiguous() * 1.5
        v = hidden_states.view(bsz, seq, 1, dim).contiguous() * 2.0
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)
        # attention_output (нам не важно для теста)
        out = (q * k * v).sum(dim=-2)
        return out, (q, k, v)


class _FakeModel(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_FakeAttention(i) for i in range(n_layers)])

    def forward(self, hidden_states, past_key_value=None):
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, past_key_value=past_key_value)
        return hidden_states


def test_hook_captures_q_k_v_per_layer():
    model = _FakeModel(n_layers=2)
    handle: CaptureHandle = install_capture_hooks(
        model,
        attention_modules=list(model.layers),
        quant_fn=lambda x: x,  # bf16 — identity
    )
    cache = _FakeCache()
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    model(x, past_key_value=cache)

    assert len(handle.q) == 2  # 2 layers
    assert len(handle.k_pre) == 2
    assert len(handle.v_post) == 2
    assert handle.q[0].shape == (1, 4, 1, 8)


def test_hook_quantizes_kv_in_cache():
    model = _FakeModel(n_layers=1)
    handle = install_capture_hooks(
        model,
        attention_modules=list(model.layers),
        quant_fn=fp8_e4m3,
    )
    cache = _FakeCache()
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    model(x, past_key_value=cache)

    # k_pre — оригинальный K; k_post — после fp8_e4m3
    assert torch.equal(handle.k_post[0], fp8_e4m3(handle.k_pre[0]))
    # Кеш модели тоже содержит quantized версию
    assert torch.equal(cache.key_cache[0], handle.k_post[0])


def test_remove_hooks_stops_capturing():
    model = _FakeModel(n_layers=1)
    handle = install_capture_hooks(
        model,
        attention_modules=list(model.layers),
        quant_fn=lambda x: x,
    )
    cache = _FakeCache()
    x = torch.randn(1, 2, 8, dtype=torch.bfloat16)
    model(x, past_key_value=cache)
    assert len(handle.q) == 1

    handle.remove()
    # После remove повторный forward не должен накопить ничего нового
    model(x, past_key_value=cache)
    assert len(handle.q) == 1
```

- [ ] **Step 5.2: Запустить тест — должен упасть**

Run: `pytest tests/test_capture_hooks.py -v`
Expected: FAIL on import.

- [ ] **Step 5.3: Реализовать attention_hooks.py**

```python
# src/kvtrace/capture/attention_hooks.py
"""Forward-хук installer для захвата Q/K_pre/K_post/V_pre/V_post.

Подмена K/V в кеше реализована через monkey-patch метода
`past_key_value.update(...)` ровно в момент attention forward. Это
работает и для HF DynamicCache, и для нашего FakeCache в тестах.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class CaptureHandle:
    """Контейнер захваченных тензоров + метод снятия хуков."""
    q: list[torch.Tensor] = field(default_factory=list)
    k_pre: list[torch.Tensor] = field(default_factory=list)
    v_pre: list[torch.Tensor] = field(default_factory=list)
    k_post: list[torch.Tensor] = field(default_factory=list)
    v_post: list[torch.Tensor] = field(default_factory=list)
    _hook_handles: list[Any] = field(default_factory=list)

    def remove(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def install_capture_hooks(
    model: nn.Module,
    attention_modules: list[nn.Module],
    quant_fn: Callable[[torch.Tensor], torch.Tensor],
) -> CaptureHandle:
    """Установить forward hooks на каждый attention-блок.

    Hook берёт outputs = (attn_output, (q, k, v)) из attention forward и:
      - пишет q, k, v в CaptureHandle как pre-quant
      - вычисляет k_post = quant_fn(k), v_post = quant_fn(v)
      - подменяет K/V в кеше (если кеш был передан) на quant-версию

    Note: для HF Qwen3 attention.forward возвращает (attn_output, attn_weights);
    K/V кладутся в cache через past_key_value.update(...) внутри forward.
    Мы вытаскиваем K/V из кеша после forward и квантуем их там же.
    """
    handle = CaptureHandle()

    def _make_hook(layer_idx: int):
        def _hook(module, inputs, outputs):
            # outputs format depends on attention class.
            # Our FakeAttention returns (attn_out, (q, k, v)).
            # HF Qwen3Attention returns (attn_out, attn_weights) but K/V are
            # in past_key_value at module.layer_idx.
            if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], tuple):
                _, (q, k, v) = outputs
            else:
                # HF path — pull from cache (passed via kwargs)
                pkv = inputs[1] if len(inputs) > 1 else None
                if pkv is None or not hasattr(pkv, "key_cache"):
                    raise RuntimeError(
                        f"Layer {layer_idx}: cannot locate Q/K/V. "
                        f"Output type {type(outputs)}, no usable past_key_value."
                    )
                k = pkv.key_cache[layer_idx]
                v = pkv.value_cache[layer_idx]
                q = None  # placeholder for HF path; см. Task 7

            handle.q.append(q if q is not None else torch.empty(0))
            handle.k_pre.append(k.detach().clone())
            handle.v_pre.append(v.detach().clone())

            k_q = quant_fn(k)
            v_q = quant_fn(v)
            handle.k_post.append(k_q.detach().clone())
            handle.v_post.append(v_q.detach().clone())

            # Replace cache entries in-place
            pkv = inputs[1] if len(inputs) > 1 else None
            if pkv is not None and hasattr(pkv, "key_cache") and layer_idx < len(pkv.key_cache):
                pkv.key_cache[layer_idx] = k_q
                pkv.value_cache[layer_idx] = v_q

        return _hook

    for layer_idx, mod in enumerate(attention_modules):
        h = mod.register_forward_hook(_make_hook(layer_idx))
        handle._hook_handles.append(h)

    return handle
```

- [ ] **Step 5.4: Тесты должны пройти**

Run: `pytest tests/test_capture_hooks.py -v`
Expected: 3 passed.

- [ ] **Step 5.5: Commit**

```bash
git add src/kvtrace/capture/attention_hooks.py tests/test_capture_hooks.py
git commit -m "feat(capture): forward hooks + quant_fn KV-cache replacement"
```

---

## Task 6: Q-capture для HF Qwen3 path

HF `Qwen3Attention.forward` не возвращает Q напрямую — Q вычисляется внутри и используется в SDPA. Нужен отдельный хук на projection Q.

**Files:**
- Modify: `src/kvtrace/capture/attention_hooks.py`
- Test: `tests/test_capture_hooks.py` (расширение)

- [ ] **Step 6.1: Добавить тест для Q-projection захвата**

Дополнить `tests/test_capture_hooks.py`:

```python
class _FakeHFAttention(nn.Module):
    """Имитирует HF Qwen3Attention: имеет .q_proj и кладёт K/V в кеш."""
    def __init__(self, layer_idx: int, dim: int = 8):
        super().__init__()
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states, past_key_value=None):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # Reshape to [bsz, seq, n_heads=1, head_dim] for cache
        bsz, seq, dim = q.shape
        q = q.view(bsz, seq, 1, dim)
        k = k.view(bsz, seq, 1, dim)
        v = v.view(bsz, seq, 1, dim)
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)
        # Возвращаем attn_output + None (как HF при output_attentions=False)
        return (q.sum(-2), None)


class _FakeHFModel(nn.Module):
    def __init__(self, n_layers: int = 1, dim: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([_FakeHFAttention(i, dim) for i in range(n_layers)])

    def forward(self, hidden_states, past_key_value=None):
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, past_key_value=past_key_value)
        return hidden_states


def test_hook_captures_q_via_q_proj_on_hf_style_attention():
    model = _FakeHFModel(n_layers=1, dim=8)
    handle = install_capture_hooks(
        model,
        attention_modules=list(model.layers),
        quant_fn=lambda x: x,
    )
    cache = _FakeCache()
    x = torch.randn(1, 3, 8, dtype=torch.float32)
    model(x, past_key_value=cache)

    # Q должно быть захвачено через хук на q_proj
    assert len(handle.q) == 1
    assert handle.q[0].shape == (1, 3, 8)  # output q_proj без reshape
```

- [ ] **Step 6.2: Запустить — должен упасть на assertion**

Run: `pytest tests/test_capture_hooks.py::test_hook_captures_q_via_q_proj_on_hf_style_attention -v`
Expected: FAIL (либо Q пуст, либо размер не тот).

- [ ] **Step 6.3: Расширить attention_hooks.py — добавить q_proj хук**

Модифицировать `install_capture_hooks` в `src/kvtrace/capture/attention_hooks.py`. Заменить функцию целиком:

```python
def install_capture_hooks(
    model: nn.Module,
    attention_modules: list[nn.Module],
    quant_fn: Callable[[torch.Tensor], torch.Tensor],
) -> CaptureHandle:
    """Установить forward hooks на каждый attention-блок + q_proj.

    Стратегия:
      - Forward-hook на attention block: захватывает K/V из past_key_value
        и подменяет на quant-версию
      - Forward-hook на module.q_proj (если есть): захватывает Q post-projection
      - Если q_proj нет (наша FakeAttention из Task 5): берёт Q из
        outputs[1] tuple
    """
    handle = CaptureHandle()
    # Pre-allocate slots per layer so order is deterministic
    n_layers = len(attention_modules)
    handle.q = [None] * n_layers  # type: ignore[list-item]
    handle.k_pre = [None] * n_layers  # type: ignore[list-item]
    handle.v_pre = [None] * n_layers  # type: ignore[list-item]
    handle.k_post = [None] * n_layers  # type: ignore[list-item]
    handle.v_post = [None] * n_layers  # type: ignore[list-item]

    def _make_q_hook(layer_idx: int):
        def _hook(module, inputs, output):
            # output of q_proj is [bsz, seq, hidden] — захватываем как есть
            handle.q[layer_idx] = output.detach().clone()
        return _hook

    def _make_attn_hook(layer_idx: int):
        def _hook(module, inputs, outputs):
            pkv = inputs[1] if len(inputs) > 1 else None

            # Try to extract Q/K/V from outputs (FakeAttention path)
            if (
                isinstance(outputs, tuple)
                and len(outputs) == 2
                and isinstance(outputs[1], tuple)
                and len(outputs[1]) == 3
            ):
                q_tup, k_tup, v_tup = outputs[1]
                if handle.q[layer_idx] is None:
                    handle.q[layer_idx] = q_tup.detach().clone()
                k_src, v_src = k_tup, v_tup
            elif pkv is not None and hasattr(pkv, "key_cache"):
                k_src = pkv.key_cache[layer_idx]
                v_src = pkv.value_cache[layer_idx]
            else:
                raise RuntimeError(
                    f"Layer {layer_idx}: no Q/K/V source. "
                    f"output type={type(outputs)}, pkv={pkv}"
                )

            handle.k_pre[layer_idx] = k_src.detach().clone()
            handle.v_pre[layer_idx] = v_src.detach().clone()

            k_q = quant_fn(k_src)
            v_q = quant_fn(v_src)
            handle.k_post[layer_idx] = k_q.detach().clone()
            handle.v_post[layer_idx] = v_q.detach().clone()

            if pkv is not None and hasattr(pkv, "key_cache") and layer_idx < len(pkv.key_cache):
                pkv.key_cache[layer_idx] = k_q
                pkv.value_cache[layer_idx] = v_q
        return _hook

    for layer_idx, mod in enumerate(attention_modules):
        if hasattr(mod, "q_proj"):
            h = mod.q_proj.register_forward_hook(_make_q_hook(layer_idx))
            handle._hook_handles.append(h)
        h = mod.register_forward_hook(_make_attn_hook(layer_idx))
        handle._hook_handles.append(h)

    return handle
```

- [ ] **Step 6.4: Все тесты hooks должны пройти**

Run: `pytest tests/test_capture_hooks.py -v`
Expected: 4 passed.

- [ ] **Step 6.5: Commit**

```bash
git add src/kvtrace/capture/attention_hooks.py tests/test_capture_hooks.py
git commit -m "feat(capture): q_proj hook for HF Qwen3-style attention"
```

---

## Task 7: CPU runner — model load + TF mode

**Files:**
- Create: `src/kvtrace/capture/cpu_runner.py`
- Test: `tests/test_capture_runner_tf.py`

- [ ] **Step 7.1: Написать падающий тест (мок модели)**

```python
# tests/test_capture_runner_tf.py
"""Unit-test для CaptureRunner.run_tf на синтетической модели."""
from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.window import Window


def _make_synthetic_runner(n_layers=2, hidden=8, vocab=16):
    """Создаёт CaptureRunner с заранее загруженной фейковой моделью."""
    from tests.test_capture_hooks import _FakeHFAttention

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_FakeHFAttention(i, hidden) for i in range(n_layers)])
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            self.config = MagicMock(num_hidden_layers=n_layers, hidden_size=hidden)

        def __call__(self, input_ids=None, past_key_values=None, **kw):
            h = nn.functional.one_hot(input_ids, num_classes=hidden).float()
            for layer in self.layers:
                h, _ = layer(h, past_key_value=past_key_values)
            logits = self.lm_head(h)
            out = MagicMock()
            out.logits = logits
            return out

    runner = CaptureRunner.__new__(CaptureRunner)
    runner._model = _Wrapper()
    runner._tokenizer = MagicMock(eos_token_id=0)
    runner._attention_modules = list(runner._model.layers)
    runner._n_layers = n_layers
    runner._vocab = vocab
    return runner


def test_run_tf_returns_capture_data_with_correct_shapes():
    runner = _make_synthetic_runner(n_layers=2, hidden=8, vocab=16)
    input_token_ids = list(range(10))  # 10 токенов
    window = Window(ws=0, we=10, truncated_left=False, truncated_right=False)

    cap = runner.run_tf(
        input_token_ids=input_token_ids,
        window=window,
        quant="bf16",
        problem_id=42,
        fdp_token_idx=5,
    )

    assert cap.meta["problem_id"] == 42
    assert cap.meta["quant"] == "bf16"
    assert cap.meta["mode"] == "tf"
    assert cap.meta["fdp_token_idx"] == 5
    assert cap.meta["W"] == 10
    assert len(cap.q) == 2
    assert cap.q[0].shape[0] == 10  # W positions
    assert cap.logits.shape == (10, 16)


def test_run_tf_slices_to_window():
    runner = _make_synthetic_runner(n_layers=1, hidden=8, vocab=16)
    input_token_ids = list(range(20))
    window = Window(ws=5, we=15, truncated_left=False, truncated_right=False)

    cap = runner.run_tf(
        input_token_ids=input_token_ids,
        window=window,
        quant="bf16",
        problem_id=0,
        fdp_token_idx=10,
    )

    assert cap.meta["window_start"] == 5
    assert cap.meta["window_end"] == 15
    assert cap.q[0].shape[0] == 10  # 15-5
    assert cap.logits.shape == (10, 16)
```

- [ ] **Step 7.2: Запустить — должен упасть**

Run: `pytest tests/test_capture_runner_tf.py -v`
Expected: FAIL on import.

- [ ] **Step 7.3: Реализовать cpu_runner.py — load + run_tf**

```python
# src/kvtrace/capture/cpu_runner.py
"""Capture pipeline orchestrator на CPU.

Загружает Qwen3-1.7B в bf16 на CPU, запускает forward с hook'ами
для teacher-forced или autoregressive захвата.
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

import torch

from kvtrace.capture.attention_hooks import install_capture_hooks
from kvtrace.capture.fp8_sim import QUANT_FNS
from kvtrace.capture.storage import CaptureData
from kvtrace.capture.window import Window

log = logging.getLogger(__name__)


class CaptureRunner:
    """Single-model lifecycle: load → many captures → unload."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._attention_modules: list[Any] = []
        self._n_layers: int = 0
        self._vocab: int = 0
        self._model_revision_hash: str = ""

    def load_model(self, hf_id: str, trust_remote_code: bool = True) -> None:
        """Load Qwen3-1.7B in bf16 on CPU with eager attention."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16,
            device_map={"": "cpu"},
            trust_remote_code=trust_remote_code,
            attn_implementation="eager",
        )
        self._model.eval()
        # Find attention modules. Qwen3: model.model.layers[i].self_attn
        layers = self._model.model.layers
        self._attention_modules = [layer.self_attn for layer in layers]
        self._n_layers = len(self._attention_modules)
        self._vocab = self._model.config.vocab_size
        self._model_revision_hash = getattr(self._model.config, "_commit_hash", "unknown")

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._attention_modules = []

    def run_tf(
        self,
        input_token_ids: list[int],
        window: Window,
        quant: str,
        problem_id: int,
        fdp_token_idx: int,
    ) -> CaptureData:
        """One forward pass on input_token_ids[0:window.we], slice to [ws:we]."""
        assert self._model is not None, "call load_model() first"
        if quant not in QUANT_FNS:
            raise ValueError(f"Unknown quant {quant!r}; expected one of {list(QUANT_FNS)}")
        quant_fn = QUANT_FNS[quant]

        handle = install_capture_hooks(
            self._model,
            attention_modules=self._attention_modules,
            quant_fn=quant_fn,
        )
        try:
            # Feed tokens [0:we]
            feed_ids = torch.tensor([input_token_ids[: window.we]], dtype=torch.long)
            with torch.no_grad():
                out = self._model(input_ids=feed_ids, use_cache=True)
            logits_full = out.logits[0]  # [we, vocab]

            ws, we = window.ws, window.we
            q_sliced = [_slice_seq_dim(t, ws, we) for t in handle.q]
            k_pre_sliced = [_slice_seq_dim(t, ws, we) for t in handle.k_pre]
            v_pre_sliced = [_slice_seq_dim(t, ws, we) for t in handle.v_pre]
            k_post_sliced = [_slice_seq_dim(t, ws, we) for t in handle.k_post]
            v_post_sliced = [_slice_seq_dim(t, ws, we) for t in handle.v_post]
            logits_sliced = logits_full[ws:we]

            return _build_capture(
                quant=quant,
                mode="tf",
                problem_id=problem_id,
                fdp_token_idx=fdp_token_idx,
                window=window,
                input_token_ids=input_token_ids[ws:we],
                gen_token_ids=input_token_ids[ws:we],
                early_eos=False,
                model_revision_hash=self._model_revision_hash,
                q=q_sliced,
                k_pre=k_pre_sliced,
                v_pre=v_pre_sliced,
                k_post=k_post_sliced,
                v_post=v_post_sliced,
                logits=logits_sliced,
            )
        finally:
            handle.remove()


def _slice_seq_dim(t: torch.Tensor, ws: int, we: int) -> torch.Tensor:
    """Slice seq dimension regardless of [B, seq, ...] or [seq, ...] shape."""
    if t.dim() >= 3 and t.shape[0] == 1:
        # [B=1, seq, ...] → squeeze B then slice
        return t[0, ws:we].to(torch.float16).contiguous()
    if t.dim() >= 2:
        return t[ws:we].to(torch.float16).contiguous()
    return t


def _build_capture(
    *,
    quant: str,
    mode: str,
    problem_id: int,
    fdp_token_idx: int,
    window: Window,
    input_token_ids: list[int],
    gen_token_ids: list[int],
    early_eos: bool,
    model_revision_hash: str,
    q: list[torch.Tensor],
    k_pre: list[torch.Tensor],
    v_pre: list[torch.Tensor],
    k_post: list[torch.Tensor],
    v_post: list[torch.Tensor],
    logits: torch.Tensor,
) -> CaptureData:
    import transformers

    meta = {
        "model": "qwen3-1.7b",
        "quant": quant,
        "mode": mode,
        "problem_id": problem_id,
        "fdp_token_idx": fdp_token_idx,
        "window_start": window.ws,
        "window_end": window.we,
        "W": window.size,
        "input_token_ids": list(input_token_ids),
        "gen_token_ids": list(gen_token_ids),
        "truncated_left": window.truncated_left,
        "truncated_right": window.truncated_right,
        "early_eos": early_eos,
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_revision_hash": model_revision_hash,
        "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    return CaptureData(
        meta=meta,
        q=q,
        k_pre=k_pre,
        v_pre=v_pre,
        k_post=k_post,
        v_post=v_post,
        logits=logits.to(torch.float16).contiguous(),
    )
```

- [ ] **Step 7.4: Тесты должны пройти**

Run: `pytest tests/test_capture_runner_tf.py -v`
Expected: 2 passed.

- [ ] **Step 7.5: Commit**

```bash
git add src/kvtrace/capture/cpu_runner.py tests/test_capture_runner_tf.py
git commit -m "feat(capture): CaptureRunner with TF mode + model loader"
```

---

## Task 8: CPU runner — AR mode

**Files:**
- Modify: `src/kvtrace/capture/cpu_runner.py`
- Test: `tests/test_capture_runner_ar.py`

- [ ] **Step 8.1: Написать тест AR-режима**

```python
# tests/test_capture_runner_ar.py
"""Unit-test для CaptureRunner.run_ar — autoregressive с KV-quant в loop."""
from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.window import Window


def _make_ar_runner(n_layers=1, hidden=8, vocab=16):
    """Минимальная модель + generate-метод для AR-теста."""
    from tests.test_capture_hooks import _FakeCache, _FakeHFAttention

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_FakeHFAttention(i, hidden) for i in range(n_layers)])
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            self.config = MagicMock(num_hidden_layers=n_layers, vocab_size=vocab)

        def generate(self, input_ids, max_new_tokens, do_sample, temperature,
                     repetition_penalty, return_dict_in_generate, output_scores, **kw):
            cache = _FakeCache()
            generated_logits = []
            ids = input_ids
            for _ in range(max_new_tokens):
                h = nn.functional.one_hot(ids, num_classes=hidden).float()
                for layer in self.layers:
                    h, _ = layer(h, past_key_value=cache)
                step_logits = self.lm_head(h[:, -1:])
                generated_logits.append(step_logits)
                next_id = step_logits.argmax(dim=-1)
                ids = torch.cat([ids, next_id], dim=-1)
            out = MagicMock()
            out.sequences = ids
            out.scores = tuple(t.squeeze(1) for t in generated_logits)
            return out

    runner = CaptureRunner.__new__(CaptureRunner)
    runner._model = _Wrapper()
    runner._tokenizer = MagicMock(eos_token_id=999)
    runner._attention_modules = list(runner._model.layers)
    runner._n_layers = n_layers
    runner._vocab = vocab
    runner._model_revision_hash = "test"
    return runner


def test_run_ar_generates_expected_step_count():
    runner = _make_ar_runner(n_layers=1, hidden=8, vocab=16)
    prefix_token_ids = [1, 2, 3, 4, 5]
    window = Window(ws=3, we=8, truncated_left=False, truncated_right=False)

    cap = runner.run_ar(
        prefix_token_ids=prefix_token_ids,
        window=window,
        quant="bf16",
        problem_id=7,
        fdp_token_idx=5,
        max_new_tokens=3,
    )

    assert cap.meta["mode"] == "ar"
    assert cap.meta["problem_id"] == 7
    assert cap.meta["W"] == window.size
    # gen_token_ids — конкатенация последних prefix-позиций + сгенерированных
    assert len(cap.meta["gen_token_ids"]) == window.size
```

- [ ] **Step 8.2: Запустить — должен упасть на `runner.run_ar` undefined**

Run: `pytest tests/test_capture_runner_ar.py -v`
Expected: FAIL with `AttributeError: 'CaptureRunner' object has no attribute 'run_ar'`.

- [ ] **Step 8.3: Добавить метод run_ar в `cpu_runner.py`**

Дописать в класс `CaptureRunner`:

```python
    def run_ar(
        self,
        prefix_token_ids: list[int],
        window: Window,
        quant: str,
        problem_id: int,
        fdp_token_idx: int,
        max_new_tokens: int = 250,
        repetition_penalty: float = 1.05,
    ) -> CaptureData:
        """Prefill prefix → greedy decode max_new_tokens с KV-quant в loop."""
        assert self._model is not None, "call load_model() first"
        if quant not in QUANT_FNS:
            raise ValueError(f"Unknown quant {quant!r}")
        quant_fn = QUANT_FNS[quant]

        handle = install_capture_hooks(
            self._model,
            attention_modules=self._attention_modules,
            quant_fn=quant_fn,
        )
        try:
            input_ids = torch.tensor([prefix_token_ids], dtype=torch.long)
            with torch.no_grad():
                out = self._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,  # ignored when do_sample=False
                    repetition_penalty=repetition_penalty,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            full_sequence: list[int] = out.sequences[0].tolist()
            # Logits per generated step: out.scores is tuple of [B, vocab]
            gen_logits = torch.stack(list(out.scores), dim=0)  # [n_gen, B, vocab] or [n_gen, vocab]
            if gen_logits.dim() == 3:
                gen_logits = gen_logits[:, 0, :]
            n_generated = gen_logits.shape[0]
            early_eos = n_generated < max_new_tokens

            # Build aligned tensor sequence: prefix (with hook-captured K/V at
            # each prefill position) + per-step generated K/V.
            # Hooks collected per-call; for prefill it's one call with seq=len(prefix);
            # for each decode step seq=1 → concat.
            q_all = [_concat_seq(handle.q[i::self._n_layers]) for i in range(self._n_layers)]
            k_pre_all = [_concat_seq(handle.k_pre[i::self._n_layers]) for i in range(self._n_layers)]
            v_pre_all = [_concat_seq(handle.v_pre[i::self._n_layers]) for i in range(self._n_layers)]
            k_post_all = [_concat_seq(handle.k_post[i::self._n_layers]) for i in range(self._n_layers)]
            v_post_all = [_concat_seq(handle.v_post[i::self._n_layers]) for i in range(self._n_layers)]

            ws, we = window.ws, window.we
            q_sliced = [_slice_seq_dim(t, ws, we) for t in q_all]
            k_pre_sliced = [_slice_seq_dim(t, ws, we) for t in k_pre_all]
            v_pre_sliced = [_slice_seq_dim(t, ws, we) for t in v_pre_all]
            k_post_sliced = [_slice_seq_dim(t, ws, we) for t in k_post_all]
            v_post_sliced = [_slice_seq_dim(t, ws, we) for t in v_post_all]

            # Logits: для AR имеем только generated-positions; для prefix позиций
            # их нет (модель не возвращает prefill logits через generate).
            # Заполняем prefill-окно нулями + actual generated logits.
            prefix_len = len(prefix_token_ids)
            full_logits = torch.zeros(prefix_len + n_generated, self._vocab, dtype=gen_logits.dtype)
            full_logits[prefix_len : prefix_len + n_generated] = gen_logits
            logits_sliced = full_logits[ws:we]

            return _build_capture(
                quant=quant,
                mode="ar",
                problem_id=problem_id,
                fdp_token_idx=fdp_token_idx,
                window=window,
                input_token_ids=prefix_token_ids + [0] * n_generated,
                gen_token_ids=full_sequence[ws:we],
                early_eos=early_eos,
                model_revision_hash=self._model_revision_hash,
                q=q_sliced,
                k_pre=k_pre_sliced,
                v_pre=v_pre_sliced,
                k_post=k_post_sliced,
                v_post=v_post_sliced,
                logits=logits_sliced,
            )
        finally:
            handle.remove()


def _concat_seq(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    """Concat per-call captures along seq dim. Каждый call даёт [1, seq, ...]."""
    if not tensor_list:
        return torch.empty(0)
    # Захваты разной длины (prefill = N, decode = 1); концатенируем seq dim.
    norm = [t if t.dim() >= 2 else t.unsqueeze(0) for t in tensor_list]
    # Если 4D [B, seq, h, d] — squeeze B
    norm = [t[0] if t.dim() == 4 and t.shape[0] == 1 else t for t in norm]
    return torch.cat(norm, dim=0)
```

- [ ] **Step 8.4: Тесты должны пройти**

Run: `pytest tests/test_capture_runner_ar.py -v`
Expected: 1 passed.

- [ ] **Step 8.5: Также все предыдущие должны проходить**

Run: `pytest tests/test_capture_runner_tf.py tests/test_capture_runner_ar.py tests/test_capture_hooks.py -v`
Expected: все passed.

- [ ] **Step 8.6: Commit**

```bash
git add src/kvtrace/capture/cpu_runner.py tests/test_capture_runner_ar.py
git commit -m "feat(capture): autoregressive run_ar with KV-quant in loop"
```

---

## Task 9: Manifest writer (run metadata + skip log)

**Files:**
- Create: `src/kvtrace/capture/manifest.py`
- Test: `tests/test_capture_manifest.py`

- [ ] **Step 9.1: Написать падающий тест**

```python
# tests/test_capture_manifest.py
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
```

- [ ] **Step 9.2: Запустить — должен упасть**

Run: `pytest tests/test_capture_manifest.py -v`
Expected: FAIL on import.

- [ ] **Step 9.3: Реализовать manifest.py**

```python
# src/kvtrace/capture/manifest.py
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
```

- [ ] **Step 9.4: Тесты должны пройти**

Run: `pytest tests/test_capture_manifest.py -v`
Expected: 2 passed.

- [ ] **Step 9.5: Commit**

```bash
git add src/kvtrace/capture/manifest.py tests/test_capture_manifest.py
git commit -m "feat(capture): RunManifest writes _run_metadata.json + _skipped.jsonl"
```

---

## Task 10: CLI script `scripts/06_capture_kv.py`

**Files:**
- Create: `scripts/06_capture_kv.py`
- Test: `tests/test_capture_cli.py`

- [ ] **Step 10.1: Написать тест с `--dry-run`**

```python
# tests/test_capture_cli.py
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
    )
    assert result.returncode == 0, result.stderr
    assert "problem=0" in result.stdout
    assert "problem=1" in result.stdout
    assert "quant=bf16" in result.stdout
    assert "quant=fp8_e4m3" in result.stdout
    assert "Total captures planned: 4" in result.stdout  # 2 problems × 2 quants × 1 mode
```

- [ ] **Step 10.2: Запустить — должен упасть (script not found)**

Run: `cd C:/Users/morro/prog/files && pytest tests/test_capture_cli.py -v`
Expected: FAIL (subprocess returns nonzero).

- [ ] **Step 10.3: Реализовать `scripts/06_capture_kv.py`**

```python
# scripts/06_capture_kv.py
"""Phase 6: capture Q/K/V/logits для Qwen3-1.7B на CPU.

Читает existing artifacts:
  - outputs/traces/<model>_bf16.jsonl       — bf16 reference token streams
  - outputs/fdps/<model>_<quant>.jsonl      — FDP indices per (problem, quant)

Пишет:
  - <output-dir>/<model>/<quant>_<mode>/<problem>.safetensors
  - <output-dir>/<model>/_run_metadata.json
  - <output-dir>/<model>/_skipped.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.manifest import RunManifest, SkipEntry
from kvtrace.capture.storage import save_capture
from kvtrace.capture.window import compute_window

log = logging.getLogger("capture")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_HF_IDS = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6 KV-matrix capture")
    p.add_argument("--model", default="qwen3-1.7b", choices=list(MODEL_HF_IDS))
    p.add_argument("--quants", nargs="+", default=["bf16", "fp8_e4m3", "fp8_e5m2"])
    p.add_argument("--modes", nargs="+", default=["tf", "ar"], choices=["tf", "ar"])
    p.add_argument("--problems", default="all",
                   help="'all' или '0,3,5' для подмножества")
    p.add_argument("--window-pre", type=int, default=150)
    p.add_argument("--window-post", type=int, default=100)
    p.add_argument("--traces-dir", type=Path, default=Path("outputs/traces"))
    p.add_argument("--fdps-dir", type=Path, default=Path("outputs/fdps"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/kv_capture"))
    p.add_argument("--dry-run", action="store_true",
                   help="Распечатать план без запуска forward'ов")
    p.add_argument("--resume", action="store_true",
                   help="Skip captures, чьи safetensors уже на диске")
    p.add_argument("--smoke", action="store_true",
                   help="Shortcut: --problems 0,1 --quants bf16 --modes tf (быстрый CI)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.smoke:
        args.problems = "0,1"
        args.quants = ["bf16"]
        args.modes = ["tf"]

    bf16_trace_path = args.traces_dir / f"{args.model}_bf16.jsonl"
    if not bf16_trace_path.exists():
        log.error("Missing bf16 trace: %s", bf16_trace_path)
        return 2
    bf16_traces = _load_jsonl(bf16_trace_path)

    # FDP: один файл per (model, fp8 quant). bf16 заимствует FDP от fp8_e4m3.
    fdp_by_quant: dict[str, dict[int, int]] = {}
    for q in args.quants:
        if q == "bf16":
            continue
        fdp_path = args.fdps_dir / f"{args.model}_{q}.jsonl"
        if not fdp_path.exists():
            log.error("Missing FDPs for quant=%s: %s", q, fdp_path)
            return 2
        fdp_by_quant[q] = {
            r["problem_idx"]: r["fdp_token_idx"]
            for r in _load_jsonl(fdp_path)
            if r.get("fdp_token_idx") is not None
        }
    # bf16 uses fp8_e4m3 FDPs as reference coordinate
    if "bf16" in args.quants:
        ref_quant = "fp8_e4m3" if "fp8_e4m3" in fdp_by_quant else next(iter(fdp_by_quant))
        fdp_by_quant["bf16"] = fdp_by_quant[ref_quant]

    if args.problems == "all":
        problem_ids = sorted({t["idx"] for t in bf16_traces})
    else:
        problem_ids = [int(p) for p in args.problems.split(",")]

    out_root = args.output_dir / args.model
    manifest = RunManifest(output_dir=out_root)

    plan: list[tuple[int, str, str]] = []
    for pid in problem_ids:
        for q in args.quants:
            modes_for_q = ["tf"] if q == "bf16" else args.modes
            for m in modes_for_q:
                plan.append((pid, q, m))

    print(f"Total captures planned: {len(plan)}")
    for pid, q, m in plan:
        print(f"  problem={pid} quant={q} mode={m}")

    if args.dry_run:
        return 0

    # Real run: load model once.
    runner = CaptureRunner()
    log.info("Loading %s on CPU...", args.model)
    runner.load_model(MODEL_HF_IDS[args.model])
    manifest.write_metadata(
        model=args.model,
        model_revision_hash=runner._model_revision_hash,
        quants=args.quants,
        modes=args.modes,
        window_pre=args.window_pre,
        window_post=args.window_post,
    )

    trace_by_id = {t["idx"]: t for t in bf16_traces}
    for pid, quant, mode in plan:
        out_path = out_root / f"{quant}_{mode}" / f"{pid}.safetensors"
        if args.resume and out_path.exists():
            log.info("Skipping existing: %s", out_path)
            continue

        trace = trace_by_id.get(pid)
        if trace is None:
            manifest.log_skip(SkipEntry(pid, quant, mode, "trace_not_found"))
            continue
        fdp_idx = fdp_by_quant[quant].get(pid)
        if fdp_idx is None:
            manifest.log_skip(SkipEntry(pid, quant, mode, "no_fdp"))
            continue

        token_ids = trace["token_ids"]
        try:
            window = compute_window(
                fdp_idx=fdp_idx,
                trace_len=len(token_ids),
                pre=args.window_pre,
                post=args.window_post,
            )
        except ValueError as e:
            manifest.log_skip(SkipEntry(pid, quant, mode, f"window_invalid: {e}"))
            continue

        try:
            if mode == "tf":
                cap = runner.run_tf(
                    input_token_ids=token_ids,
                    window=window,
                    quant=quant,
                    problem_id=pid,
                    fdp_token_idx=fdp_idx,
                )
            else:  # ar
                prefix_end = window.ws + (fdp_idx - window.ws)  # prefix up to fdp_idx
                prefix = token_ids[: max(1, fdp_idx - args.window_pre)]
                cap = runner.run_ar(
                    prefix_token_ids=prefix,
                    window=window,
                    quant=quant,
                    problem_id=pid,
                    fdp_token_idx=fdp_idx,
                    max_new_tokens=args.window_post + 150,
                )
        except Exception as e:
            log.exception("Capture failed for problem=%d quant=%s mode=%s", pid, quant, mode)
            manifest.log_skip(SkipEntry(pid, quant, mode, f"runtime_error: {e!r}"))
            continue

        save_capture(cap, out_path)
        log.info("Saved %s", out_path)

    runner.unload()
    return 0


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 10.4: Запустить тест dry-run**

Run: `cd C:/Users/morro/prog/files && pytest tests/test_capture_cli.py -v`
Expected: PASS.

- [ ] **Step 10.5: Commit**

```bash
git add scripts/06_capture_kv.py tests/test_capture_cli.py
git commit -m "feat(capture): scripts/06_capture_kv.py CLI entry"
```

---

## Task 11: Smoke test end-to-end (slow, real Qwen3-1.7B)

**Files:**
- Create: `tests/test_capture_smoke.py`

- [ ] **Step 11.1: Написать slow-тест**

```python
# tests/test_capture_smoke.py
"""End-to-end smoke: 1 задача × bf16 × tf на реальной Qwen3-1.7B (CPU).

Помечен @pytest.mark.slow — в CI skipped (потребует загрузки ~3GB модели).
Запускается локально перед release: `pytest tests/test_capture_smoke.py -m slow -v`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.storage import load_capture, save_capture
from kvtrace.capture.window import compute_window


@pytest.mark.slow
def test_smoke_qwen3_1_7b_bf16_tf(tmp_path: Path):
    runner = CaptureRunner()
    runner.load_model("Qwen/Qwen3-1.7B")

    fake_input = list(range(50)) + [100, 200, 300]  # mock 53 tokens
    window = compute_window(fdp_idx=40, trace_len=len(fake_input), pre=10, post=10)
    cap = runner.run_tf(
        input_token_ids=fake_input,
        window=window,
        quant="bf16",
        problem_id=0,
        fdp_token_idx=40,
    )
    out = tmp_path / "smoke.safetensors"
    save_capture(cap, out)
    loaded = load_capture(out)
    assert loaded.meta["model"] == "qwen3-1.7b"
    assert loaded.q[0].shape[0] == window.size  # W positions
    assert len(loaded.q) == 28  # Qwen3-1.7B has 28 layers
    runner.unload()
```

- [ ] **Step 11.2: Проверить, что pytest корректно skip'ает без `-m slow`**

Run: `pytest tests/test_capture_smoke.py -v`
Expected: 1 deselected (или skipped, в зависимости от pytest config).

- [ ] **Step 11.3: Запустить с `-m slow` локально (опционально — занимает 5-10 минут с загрузкой модели)**

Run: `pytest tests/test_capture_smoke.py -m slow -v`
Expected: PASS (или skip если модель не загружается).

- [ ] **Step 11.4: Commit**

```bash
git add tests/test_capture_smoke.py
git commit -m "test(capture): slow smoke test for Qwen3-1.7B real load"
```

---

## Task 12: Golden FP8-vs-vLLM verification

**Files:**
- Create: `tests/test_capture_fp8_matches_vllm.py`

- [ ] **Step 12.1: Написать тест, сверяющий CPU FP8 sim с существующей vLLM-трассой**

```python
# tests/test_capture_fp8_matches_vllm.py
"""Golden-test: CPU FP8 simulator на первой задаче должен дать тот же
первый decode-токен, что vLLM в outputs/traces/qwen3-1.7b_fp8_e4m3.jsonl.

Помечен @pytest.mark.slow — требует загрузки модели И существующих
artefactов из основного эксперимента.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kvtrace.capture.cpu_runner import CaptureRunner

TRACES = Path("outputs/traces")


@pytest.mark.slow
def test_cpu_fp8_e4m3_matches_vllm_first_token():
    fp8_trace_path = TRACES / "qwen3-1.7b_fp8_e4m3.jsonl"
    bf16_trace_path = TRACES / "qwen3-1.7b_bf16.jsonl"
    if not fp8_trace_path.exists() or not bf16_trace_path.exists():
        pytest.skip("Existing vLLM traces not available")

    fp8_first = json.loads(fp8_trace_path.read_text().splitlines()[0])
    bf16_first = json.loads(bf16_trace_path.read_text().splitlines()[0])

    # bf16-trace prompt = первые num_prompt_tokens токенов
    prompt_len = bf16_first["num_prompt_tokens"]
    prompt_tokens = bf16_first["token_ids"][:prompt_len]
    expected_first_decode = fp8_first["token_ids"][prompt_len]

    runner = CaptureRunner()
    runner.load_model("Qwen/Qwen3-1.7B")

    # AR на 1 шаг с fp8_e4m3
    from kvtrace.capture.window import Window
    cap = runner.run_ar(
        prefix_token_ids=prompt_tokens,
        window=Window(ws=prompt_len, we=prompt_len + 1,
                      truncated_left=False, truncated_right=False),
        quant="fp8_e4m3",
        problem_id=0,
        fdp_token_idx=prompt_len,
        max_new_tokens=1,
    )
    actual_first_decode = cap.meta["gen_token_ids"][0]
    runner.unload()

    assert actual_first_decode == expected_first_decode, (
        f"CPU FP8 sim расходится с vLLM на первом decode-токене: "
        f"CPU={actual_first_decode} vs vLLM={expected_first_decode}. "
        "Симулятор арифметически неверен или хук подменяет K/V в неправильной точке."
    )
```

- [ ] **Step 12.2: Запустить (опционально, требует artefactов)**

Run: `pytest tests/test_capture_fp8_matches_vllm.py -m slow -v`
Expected: PASS если artefacты есть; иначе skipped.

- [ ] **Step 12.3: Commit**

```bash
git add tests/test_capture_fp8_matches_vllm.py
git commit -m "test(capture): golden FP8 sim vs vLLM first-decode-token"
```

---

## Task 13: Sliding-window attention verification (slow)

**Files:**
- Create: `tests/test_capture_sliding_window.py`

- [ ] **Step 13.1: Тест на длинный prompt, требующий sliding window**

```python
# tests/test_capture_sliding_window.py
"""Проверка, что хуки корректно работают когда кеш ≥ sliding_window."""
from __future__ import annotations

import pytest
import torch

from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.fp8_sim import fp8_e4m3
from kvtrace.capture.window import Window


@pytest.mark.slow
def test_hooks_work_with_long_context():
    runner = CaptureRunner()
    runner.load_model("Qwen/Qwen3-1.7B")

    # Qwen3-1.7B sliding_window обычно ≥ 4096; даём 5000 токенов чтобы
    # гарантированно перейти в sliding режим.
    long_input = list(range(5000))
    window = Window(ws=4500, we=4600, truncated_left=False, truncated_right=False)

    cap = runner.run_tf(
        input_token_ids=long_input,
        window=window,
        quant="fp8_e4m3",
        problem_id=0,
        fdp_token_idx=4550,
    )
    # Все 28 слоёв захвачены
    assert len(cap.q) == 28
    # Размер окна совпадает
    assert cap.q[0].shape[0] == window.size
    # K_post должен быть равен fp8_e4m3(K_pre) elementwise
    assert torch.equal(cap.k_post[0], fp8_e4m3(cap.k_pre[0]))
    runner.unload()
```

- [ ] **Step 13.2: Запустить (опционально)**

Run: `pytest tests/test_capture_sliding_window.py -m slow -v`

- [ ] **Step 13.3: Commit**

```bash
git add tests/test_capture_sliding_window.py
git commit -m "test(capture): sliding-window attention regression"
```

---

## Task 14: Документация и phase 6 в run_all.sh

**Files:**
- Modify: `scripts/run_all.sh`
- Modify: `README.md` (если есть упоминание phases)

- [ ] **Step 14.1: Проверить, что есть `scripts/run_all.sh`**

Run: `ls scripts/run_all.sh`
Expected: file exists.

- [ ] **Step 14.2: Дописать Phase 6 в run_all.sh**

Открыть `scripts/run_all.sh` и добавить в конец (перед последним `exit`):

```bash
# Phase 6: KV-matrix capture (CPU-only, optional)
if [[ "${RUN_PHASE_6:-0}" == "1" ]]; then
    echo "=== Phase 6: KV-matrix capture (CPU) ==="
    python scripts/06_capture_kv.py \
        --model qwen3-1.7b \
        --quants bf16 fp8_e4m3 fp8_e5m2 \
        --modes tf ar \
        --output-dir outputs/kv_capture/ \
        --resume
fi
```

- [ ] **Step 14.3: Smoke-test, что bash-синтаксис валиден**

Run: `bash -n scripts/run_all.sh`
Expected: silent success.

- [ ] **Step 14.4: Commit**

```bash
git add scripts/run_all.sh
git commit -m "feat(capture): optional Phase 6 in run_all.sh (RUN_PHASE_6=1)"
```

---

## Final verification

- [ ] **Step F.1: Все тесты проходят (без slow)**

Run: `pytest tests/ -x --ignore=tests/test_generators_vllm.py -q -m "not slow"`
Expected: все passed.

- [ ] **Step F.2: Coverage по новому коду ≥ 85%**

Run: `pytest tests/ --cov=src/kvtrace/capture --cov-report=term-missing -m "not slow"`
Expected: `TOTAL ... 85%+`.

- [ ] **Step F.3: ruff/mypy чисто**

Run: `ruff check src/kvtrace/capture scripts/06_capture_kv.py tests/test_capture_*.py`
Run: `mypy src/kvtrace/capture`
Expected: no errors.

- [ ] **Step F.4: Локально запустить smoke**

Run: `python scripts/06_capture_kv.py --model qwen3-1.7b --quants bf16 --modes tf --problems 0 --output-dir /tmp/cap_test`
Expected: появляется `/tmp/cap_test/qwen3-1.7b/bf16_tf/0.safetensors`.

- [ ] **Step F.5: Final commit**

```bash
git commit --allow-empty -m "feat(capture): Phase 6 KV-matrix capture complete"
```

---

## Self-review checklist (для исполнителя)

Перед сдачей работы:
- Каждое имя поля в `CaptureData.meta` совпадает с layout в spec — особенно `model`, `quant`, `mode`, `problem_id`, `fdp_token_idx`, `window_start`, `window_end`, `W`
- Все `quant_fn` ссылки берут `QUANT_FNS[quant]` — нет hard-coded fp8_e4m3
- `_slice_seq_dim` корректно обрабатывает 3D и 4D тензоры — добавьте debug-print при первом запуске real model
- Hook на `q_proj` берёт output до reshape в num_heads — это сырой `[B, seq, hidden]`; downstream метрики должны учитывать это
- В AR-режиме prefix берётся как `token_ids[: max(1, fdp_idx - window_pre)]` — поправьте если fdp_idx − window_pre < 1
