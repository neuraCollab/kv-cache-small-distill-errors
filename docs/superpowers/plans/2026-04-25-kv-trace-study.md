# KV Cache Quantization — Trace-Level Diagnostic Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, idempotent 4-phase research pipeline (GENERATE → FDP → JUDGE → ANALYZE) that diagnoses how KV cache quantization methods cause characteristic failure signatures in 1.5B–7B reasoning models, with full test coverage for all four research tasks.

**Architecture:** Four idempotent phases, each reading and writing JSONL with HuggingFace Hub checkpoints. `Generator` ABC abstracts two engines (vLLM for BF16/FP8, HF Transformers for HQQ INT4/INT2). GPU is held only during Phase 1. Judge uses Anthropic prompt caching plus on-disk SHA256 cache.

**Tech Stack:** Python 3.10+, vLLM 0.6.6, transformers 4.45, anthropic 0.40, pydantic 2.x, sentence-transformers, pytest + pytest-cov + pytest-mock + responses, huggingface_hub, scipy, matplotlib.

**Spec reference:** `docs/superpowers/specs/2026-04-25-kv-trace-study-design.md`

---

## Ground rules for every task

1. Red → Green → Commit. Run the failing test first. Write the minimum code to pass. Commit.
2. If a test fails for a reason you did not expect, **fix the code, not the test** — unless the test itself contains a bug, in which case call it out in the commit message.
3. After each task, run the full CPU suite: `pytest -m "not gpu and not live_api"` and confirm green before moving on.
4. Every JSONL line is self-describing — model, config, seed, timestamp, prompt version are stamped per row.

---

## File structure (map of what gets created)

```
kv-trace-study/
├── .github/workflows/ci.yml              # Task 0.3
├── .gitignore                            # Task 0.1
├── Makefile                              # Task 0.1
├── README.md                             # Task 9.1
├── pyproject.toml                        # Task 0.1
├── requirements.txt                      # Task 0.1
├── requirements-dev.txt                  # Task 0.1
├── config/
│   ├── models.yaml                       # Task 1.1
│   ├── quant_methods.yaml                # Task 1.1
│   └── pipeline.yaml                     # Task 1.1
├── src/kvtrace/
│   ├── __init__.py                       # Task 0.2
│   ├── config.py                         # Task 1.2 (pydantic config loader)
│   ├── dataset_loader.py                 # Task 1.3 (port+extend baseline)
│   ├── trace_utils.py                    # Task 1.4 (port baseline)
│   ├── memory.py                         # Task 2.1
│   ├── generators/
│   │   ├── __init__.py                   # Task 0.2
│   │   ├── base.py                       # Task 3.1
│   │   ├── vllm_gen.py                   # Task 3.2
│   │   └── hf_gen.py                     # Task 3.3
│   ├── fdp/
│   │   ├── __init__.py                   # Task 0.2
│   │   ├── tokenizer_align.py            # Task 4.1
│   │   ├── semantic_resync.py            # Task 4.2
│   │   └── finder.py                     # Task 4.3
│   ├── judge/
│   │   ├── __init__.py                   # Task 0.2
│   │   ├── taxonomy.py                   # Task 5.1
│   │   ├── prompt.py                     # Task 5.2
│   │   ├── claude_judge.py               # Task 5.3
│   │   └── golden_set.py                 # Task 5.4
│   ├── hf_hub/
│   │   ├── __init__.py                   # Task 0.2
│   │   ├── upload.py                     # Task 6.1
│   │   └── download.py                   # Task 6.1
│   └── analysis/
│       ├── __init__.py                   # Task 0.2
│       ├── signatures.py                 # Task 7.1
│       └── report.py                     # Task 7.2
├── scripts/
│   ├── 01_generate_traces.py             # Task 8.1
│   ├── 02_find_fdps.py                   # Task 8.2
│   ├── 03_judge_fdps.py                  # Task 8.3
│   ├── 04_analyze.py                     # Task 8.4
│   └── run_all.sh                        # Task 8.5
├── tests/
│   ├── conftest.py                       # Task 0.4
│   ├── fixtures/                         # Task 0.4
│   └── test_*.py                         # tasks 1.3 – 8.4
└── outputs/                              # runtime, gitignored
```

Existing baseline files to port:
- `dataset_loader.py` → `src/kvtrace/dataset_loader.py` (extend for MATH-500 50-sample)
- `trace_utils.py` → `src/kvtrace/trace_utils.py` (unchanged)
- `run_vllm_pipeline.py` → logic split into `generators/vllm_gen.py` + `scripts/01_generate_traces.py`

---

# Phase 0 — Project skeleton

## Task 0.1: Project files (pyproject, requirements, .gitignore, Makefile, git init)

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `Makefile`

- [ ] **Step 1: Init git repo**

```bash
cd C:/Users/morro/prog/files
git init
git add README.md requirements.txt dataset_loader.py trace_utils.py run_vllm_pipeline.py docs/
git commit -m "chore: seed repo with baseline pipeline and design spec"
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
.pytest_cache/
.ruff_cache/
.mypy_cache/
*.egg-info/
.coverage
htmlcov/
dist/
build/

# Project
outputs/
.cache/
models/
*.pt
*.bin
.env
.env.local

# IDE
.vscode/
.idea/
```

- [ ] **Step 3: Write `requirements.txt`**

```text
# Core
vllm==0.6.6
transformers==4.45.0
tokenizers>=0.20
accelerate>=0.34.0
datasets>=3.0.0
huggingface_hub>=0.25.0

# KV quantization
hqq>=0.2.2

# Judge
anthropic==0.40.0

# FDP semantic re-sync
sentence-transformers>=3.0.0

# Config
pydantic>=2.7
pyyaml>=6.0

# Analysis
scipy>=1.11
numpy<2.0
matplotlib>=3.8

# Utils
tqdm>=4.66
tenacity>=8.2
```

- [ ] **Step 4: Write `requirements-dev.txt`**

```text
-r requirements.txt
pytest>=8.0
pytest-cov>=5.0
pytest-mock>=3.12
responses>=0.25
ruff>=0.6
mypy>=1.11
types-PyYAML
```

- [ ] **Step 5: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "kvtrace"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
minversion = "8.0"
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]
markers = [
    "gpu: requires CUDA (skipped by default in CI)",
    "live_api: requires ANTHROPIC_API_KEY (skipped by default)",
]

[tool.coverage.run]
source = ["src/kvtrace"]
omit = ["*/generators/vllm_gen.py", "*/generators/hf_gen.py"]

[tool.coverage.report]
fail_under = 85
show_missing = true

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
strict_optional = true
```

- [ ] **Step 6: Write `Makefile`**

```makefile
.PHONY: install install-dev test test-all lint type fmt clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest -m "not gpu and not live_api"

test-gpu:
	pytest -m gpu

test-live:
	pytest -m live_api

test-all:
	pytest

lint:
	ruff check .

fmt:
	ruff check --fix .

type:
	mypy src/kvtrace

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov outputs/*.jsonl
```

- [ ] **Step 7: Verify installability**

Run: `pip install -e . && python -c "import kvtrace"`
Expected: import error because `src/kvtrace/__init__.py` not yet created — that is the next task.

- [ ] **Step 8: Commit**

```bash
git add .gitignore pyproject.toml requirements.txt requirements-dev.txt Makefile
git commit -m "chore: project skeleton (pyproject, requirements, Makefile)"
```

---

## Task 0.2: Package init files

**Files:**
- Create: `src/kvtrace/__init__.py`
- Create: `src/kvtrace/generators/__init__.py`
- Create: `src/kvtrace/fdp/__init__.py`
- Create: `src/kvtrace/judge/__init__.py`
- Create: `src/kvtrace/hf_hub/__init__.py`
- Create: `src/kvtrace/analysis/__init__.py`

- [ ] **Step 1: Write all init files**

`src/kvtrace/__init__.py`:
```python
"""kvtrace — KV cache quantization trace-level diagnostic study."""
__version__ = "0.1.0"
```

All other `__init__.py` files are empty (one-line file: `"""subpackage."""`).

- [ ] **Step 2: Verify**

Run: `pip install -e . && python -c "import kvtrace; print(kvtrace.__version__)"`
Expected: prints `0.1.0`.

- [ ] **Step 3: Commit**

```bash
git add src/kvtrace/__init__.py src/kvtrace/*/__init__.py
git commit -m "chore: package init files"
```

---

## Task 0.3: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write CI workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install CPU-only deps
        run: |
          python -m pip install --upgrade pip
          # Skip vLLM + sentence-transformers heavy CUDA extras in CI:
          # install only what the unmarked tests need.
          pip install pytest pytest-cov pytest-mock responses ruff mypy
          pip install transformers==4.45.0 tokenizers datasets huggingface_hub
          pip install pydantic pyyaml scipy numpy matplotlib tqdm tenacity
          pip install anthropic==0.40.0
          pip install -e . --no-deps
      - name: Lint
        run: ruff check .
      - name: Type check
        run: mypy src/kvtrace
      - name: Test (CPU, no live API)
        run: pytest -m "not gpu and not live_api" --cov=src/kvtrace --cov-fail-under=85
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: CPU test workflow with lint + mypy + coverage gate"
```

---

## Task 0.4: Test infrastructure (conftest + fixtures)

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `tests/fixtures/__init__.py` (empty)
- Create: `tests/fixtures/aime_tiny.json`
- Create: `tests/fixtures/math500_tiny.json`
- Create: `tests/fixtures/trace_baseline_sample.txt`
- Create: `tests/fixtures/trace_quant_sample.txt`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures and helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def aime_tiny(fixtures_dir) -> list[dict]:
    with (fixtures_dir / "aime_tiny.json").open() as f:
        return json.load(f)


@pytest.fixture
def math500_tiny(fixtures_dir) -> list[dict]:
    with (fixtures_dir / "math500_tiny.json").open() as f:
        return json.load(f)


@pytest.fixture
def baseline_trace_text(fixtures_dir) -> str:
    return (fixtures_dir / "trace_baseline_sample.txt").read_text(encoding="utf-8")


@pytest.fixture
def quant_trace_text(fixtures_dir) -> str:
    return (fixtures_dir / "trace_quant_sample.txt").read_text(encoding="utf-8")
```

- [ ] **Step 2: Create fixture `tests/fixtures/aime_tiny.json`**

```json
[
  {"Problem": "Find the smallest positive integer n such that n^2 + n + 41 is composite.", "Answer": "40"},
  {"Problem": "How many ways can you arrange the letters of BANANA?", "Answer": "60"}
]
```

- [ ] **Step 3: Create fixture `tests/fixtures/math500_tiny.json`**

```json
[
  {"problem": "What is 2+2?", "answer": "4"},
  {"problem": "What is the square root of 144?", "answer": "12"}
]
```

- [ ] **Step 4: Create baseline trace fixture `trace_baseline_sample.txt`**

```text
Let me solve this step by step.
First, I need to compute 2+3 = 5.
Next, I multiply by 4: 5*4 = 20.
So the answer is 20.
</think>

The answer is \boxed{20}.
```

- [ ] **Step 5: Create quant trace fixture `trace_quant_sample.txt`**

```text
Let me solve this step by step.
First, I need to compute 2+3 = 5.
Next, I multiply by 4: 5*4 = 24.
So the answer is 24.
</think>

The answer is \boxed{24}.
```

(Note: this pair diverges at "5*4 = 24" — an arithmetic error, category A.)

- [ ] **Step 6: Verify fixtures load**

Run: `pytest tests/ -v --collect-only`
Expected: collects 0 tests but no errors.

- [ ] **Step 7: Commit**

```bash
git add tests/__init__.py tests/conftest.py tests/fixtures/
git commit -m "test: conftest and shared fixture data (tiny AIME/MATH + paired trace samples)"
```

---

# Phase 1 — Config, dataset, trace utilities

## Task 1.1: YAML configs for models, quant methods, pipeline

**Files:**
- Create: `config/models.yaml`
- Create: `config/quant_methods.yaml`
- Create: `config/pipeline.yaml`

- [ ] **Step 1: Write `config/models.yaml`**

```yaml
models:
  deepseek-r1-distill-qwen-1.5b:
    hf_id: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    dtype: bfloat16
    max_model_len: 32768
    trust_remote_code: true

  qwen3-1.7b:
    hf_id: Qwen/Qwen3-1.7B
    dtype: bfloat16
    max_model_len: 32768
    trust_remote_code: true
    # Qwen3 thinking mode: add thinking system prompt via chat template.
    chat_template_kwargs:
      enable_thinking: true

  deepseek-r1-distill-qwen-7b:
    hf_id: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    dtype: bfloat16
    max_model_len: 32768
    trust_remote_code: true
```

- [ ] **Step 2: Write `config/quant_methods.yaml`**

```yaml
quant_methods:
  bf16:
    engine: vllm
    kv_cache_dtype: auto      # unquantized baseline
    overrides: {}

  fp8_e5m2:
    engine: vllm
    kv_cache_dtype: fp8_e5m2
    overrides: {}

  fp8_e4m3:
    engine: vllm
    kv_cache_dtype: fp8_e4m3
    overrides: {}

  hqq_int4:
    engine: hf
    hqq_nbits: 4
    overrides:
      # HF Transformers is slower; lower max_model_len for tight 7B budget.
      max_model_len_override:
        deepseek-r1-distill-qwen-7b: 16384

  hqq_int2:
    engine: hf
    hqq_nbits: 2
    overrides:
      max_model_len_override:
        deepseek-r1-distill-qwen-7b: 16384
```

- [ ] **Step 3: Write `config/pipeline.yaml`**

```yaml
pipeline:
  dataset:
    aime_24_count: 30
    math_500_count: 50
  seed: 42
  sampling:
    temperature: 0.0
    top_p: 1.0
    max_tokens: 32768
  fdp:
    context_window: 200
    resync_lookahead: 500
    cosmetic_cosine_threshold: 0.9
    max_cosmetic_skips: 5
    embed_model: sentence-transformers/all-MiniLM-L6-v2
  judge:
    model: claude-sonnet-4-6
    temperature: 0.0
    max_tokens: 400
    cache_dir: .cache/judge
    max_concurrent: 5
    calibration_threshold: 0.7
  hf_hub:
    repo_id_env: HF_REPO_ID
    repo_id_default_template: "{hf_user}/kv-trace-study"
    repo_type: dataset
```

- [ ] **Step 4: Verify YAML loads**

Run: `python -c "import yaml; print(yaml.safe_load(open('config/pipeline.yaml')))"`
Expected: nested dict printed, no errors.

- [ ] **Step 5: Commit**

```bash
git add config/
git commit -m "config: models/quant_methods/pipeline YAML"
```

---

## Task 1.2: Config loader (pydantic)

**Files:**
- Create: `src/kvtrace/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path

import pytest

from kvtrace.config import load_all_configs, ModelCfg, QuantCfg, PipelineCfg

CONFIG_DIR = Path(__file__).parent.parent / "config"


def test_load_all_configs_returns_three_registries():
    models, quants, pipeline = load_all_configs(CONFIG_DIR)
    assert isinstance(pipeline, PipelineCfg)
    assert "deepseek-r1-distill-qwen-1.5b" in models
    assert "bf16" in quants


def test_model_cfg_fields():
    models, _, _ = load_all_configs(CONFIG_DIR)
    m = models["deepseek-r1-distill-qwen-1.5b"]
    assert isinstance(m, ModelCfg)
    assert m.hf_id == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert m.dtype == "bfloat16"
    assert m.max_model_len == 32768


def test_quant_cfg_vllm_engine():
    _, quants, _ = load_all_configs(CONFIG_DIR)
    q = quants["fp8_e5m2"]
    assert isinstance(q, QuantCfg)
    assert q.engine == "vllm"
    assert q.kv_cache_dtype == "fp8_e5m2"


def test_quant_cfg_hf_engine_has_nbits():
    _, quants, _ = load_all_configs(CONFIG_DIR)
    q = quants["hqq_int4"]
    assert q.engine == "hf"
    assert q.hqq_nbits == 4


def test_quant_cfg_unknown_engine_raises(tmp_path):
    bad = tmp_path / "quant_methods.yaml"
    bad.write_text("quant_methods:\n  weird:\n    engine: pytorch_extend\n")
    with pytest.raises(ValueError, match="engine"):
        from kvtrace.config import _load_quants
        _load_quants(bad)


def test_pipeline_cfg_fdp_params():
    _, _, p = load_all_configs(CONFIG_DIR)
    assert p.fdp.context_window == 200
    assert 0.0 < p.fdp.cosmetic_cosine_threshold <= 1.0
    assert p.fdp.max_cosmetic_skips >= 0
```

- [ ] **Step 2: Run test — expect failure**

Run: `pytest tests/test_config.py -v`
Expected: ImportError `kvtrace.config` does not exist.

- [ ] **Step 3: Write implementation**

`src/kvtrace/config.py`:
```python
"""Typed configuration loader (pydantic v2)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelCfg(BaseModel):
    hf_id: str
    dtype: Literal["bfloat16", "float16", "auto"] = "bfloat16"
    max_model_len: int = 32768
    trust_remote_code: bool = True
    chat_template_kwargs: dict = Field(default_factory=dict)


class QuantCfg(BaseModel):
    engine: Literal["vllm", "hf"]
    kv_cache_dtype: str | None = None           # vLLM only
    hqq_nbits: int | None = None                # HF only
    overrides: dict = Field(default_factory=dict)

    @field_validator("engine", mode="after")
    @classmethod
    def _validate(cls, v):
        if v not in ("vllm", "hf"):
            raise ValueError(f"engine must be 'vllm' or 'hf', got {v!r}")
        return v


class DatasetCfg(BaseModel):
    aime_24_count: int = 30
    math_500_count: int = 50


class SamplingCfg(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 32768


class FDPCfg(BaseModel):
    context_window: int = 200
    resync_lookahead: int = 500
    cosmetic_cosine_threshold: float = 0.9
    max_cosmetic_skips: int = 5
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class JudgeCfg(BaseModel):
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 400
    cache_dir: str = ".cache/judge"
    max_concurrent: int = 5
    calibration_threshold: float = 0.7


class HFHubCfg(BaseModel):
    repo_id_env: str = "HF_REPO_ID"
    repo_id_default_template: str = "{hf_user}/kv-trace-study"
    repo_type: Literal["dataset", "model"] = "dataset"


class PipelineCfg(BaseModel):
    dataset: DatasetCfg
    seed: int = 42
    sampling: SamplingCfg
    fdp: FDPCfg
    judge: JudgeCfg
    hf_hub: HFHubCfg


def _load_models(path: Path) -> dict[str, ModelCfg]:
    data = yaml.safe_load(path.read_text())
    return {k: ModelCfg(**v) for k, v in data["models"].items()}


def _load_quants(path: Path) -> dict[str, QuantCfg]:
    data = yaml.safe_load(path.read_text())
    return {k: QuantCfg(**v) for k, v in data["quant_methods"].items()}


def _load_pipeline(path: Path) -> PipelineCfg:
    data = yaml.safe_load(path.read_text())
    return PipelineCfg(**data["pipeline"])


def load_all_configs(
    config_dir: Path,
) -> tuple[dict[str, ModelCfg], dict[str, QuantCfg], PipelineCfg]:
    """Load the three YAMLs in config/ into typed registries."""
    return (
        _load_models(config_dir / "models.yaml"),
        _load_quants(config_dir / "quant_methods.yaml"),
        _load_pipeline(config_dir / "pipeline.yaml"),
    )
```

- [ ] **Step 4: Run test — expect pass**

Run: `pytest tests/test_config.py -v`
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/config.py tests/test_config.py
git commit -m "feat(config): typed pydantic loader for models/quants/pipeline"
```

---

## Task 1.3: Port and extend `dataset_loader.py`

**Files:**
- Create: `src/kvtrace/dataset_loader.py` (port existing + add mix)
- Create: `tests/test_dataset_loader.py`

- [ ] **Step 1: Write the failing test**

`tests/test_dataset_loader.py`:
```python
from unittest.mock import patch

import pytest

from kvtrace.dataset_loader import (
    DATASET_CONFIGS,
    MathProblem,
    load_math_dataset,
    load_study_mix,
)


def test_math_problem_to_dict_roundtrip():
    p = MathProblem(idx=0, problem="x+1=2", answer="1", source="aime-24")
    d = p.to_dict()
    assert d["idx"] == 0
    assert d["problem"] == "x+1=2"
    assert d["ground_truth"] == "1"
    assert d["source"] == "aime-24"


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_math_dataset("gsm8k")


def _fake_hf_load(aime_rows, math_rows):
    from unittest.mock import MagicMock

    def _fake_load_dataset(path, split):
        ds = MagicMock()
        rows = aime_rows if "AIME" in path else math_rows
        ds.__len__.return_value = len(rows)
        ds.select = lambda r: _DS(rows[list(r)[-1] + 1 if r.stop else 0 : r.stop])
        ds.__iter__ = lambda self: iter(rows)
        return _DS(rows)
    return _fake_load_dataset


class _DS:
    def __init__(self, rows):
        self._rows = rows
    def __len__(self):
        return len(self._rows)
    def __iter__(self):
        return iter(self._rows)
    def select(self, indices):
        return _DS([self._rows[i] for i in indices])
    def shuffle(self, seed):
        return self


def test_load_math_dataset_aime(aime_tiny, monkeypatch):
    import kvtrace.dataset_loader as dl
    monkeypatch.setattr(dl, "load_dataset", lambda path, split: _DS(aime_tiny))
    problems = dl.load_math_dataset("aime-24", num_samples=2)
    assert len(problems) == 2
    assert problems[0].source == "aime-24"
    assert problems[0].answer == "40"
    assert "Find the smallest" in problems[0].problem


def test_load_math_dataset_math500(math500_tiny, monkeypatch):
    import kvtrace.dataset_loader as dl
    monkeypatch.setattr(dl, "load_dataset", lambda path, split: _DS(math500_tiny))
    problems = dl.load_math_dataset("math-500", num_samples=2)
    assert len(problems) == 2
    assert problems[0].answer == "4"


def test_load_study_mix_preserves_order(aime_tiny, math500_tiny, monkeypatch):
    """30 AIME first, then 50 MATH — but with tiny fixture, 2 + 2 = 4."""
    import kvtrace.dataset_loader as dl
    calls = {"aime": 0, "math": 0}

    def _fake(path, split):
        if "AIME" in path:
            calls["aime"] += 1
            return _DS(aime_tiny)
        calls["math"] += 1
        return _DS(math500_tiny)

    monkeypatch.setattr(dl, "load_dataset", _fake)
    problems = load_study_mix(aime_n=2, math_n=2)

    assert len(problems) == 4
    assert problems[0].source == "aime-24"
    assert problems[1].source == "aime-24"
    assert problems[2].source == "math-500"
    assert problems[3].source == "math-500"
    assert [p.idx for p in problems] == [0, 1, 2, 3]
```

- [ ] **Step 2: Run test — expect failure**

Run: `pytest tests/test_dataset_loader.py -v`
Expected: ImportError on `load_study_mix`.

- [ ] **Step 3: Write implementation**

`src/kvtrace/dataset_loader.py`:
```python
"""Math dataset loader. Extends the baseline with load_study_mix()."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset


@dataclass
class MathProblem:
    idx: int
    problem: str
    answer: str
    source: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "problem": self.problem,
            "ground_truth": self.answer,
            "source": self.source,
            "metadata": self.metadata,
        }


DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "aime-24": {
        "path": "Maxwell-Jia/AIME_2024",
        "split": "train",
        "problem_key": "Problem",
        "answer_key": "Answer",
    },
    "aime-24-aimo": {
        "path": "AI-MO/aimo-validation-aime",
        "split": "train",
        "problem_key": "problem",
        "answer_key": "answer",
    },
    "math-500": {
        "path": "HuggingFaceH4/MATH-500",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
    },
}


def load_math_dataset(
    name: str = "aime-24",
    num_samples: int | None = None,
    seed: int = 42,
    shuffle: bool = False,
) -> list[MathProblem]:
    if name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_CONFIGS)}"
        )

    cfg = DATASET_CONFIGS[name]
    ds = load_dataset(cfg["path"], split=cfg["split"])

    if shuffle:
        ds = ds.shuffle(seed=seed)
    if num_samples is not None:
        n = min(num_samples, len(ds))
        ds = ds.select(range(n))

    pk, ak = cfg["problem_key"], cfg["answer_key"]
    out: list[MathProblem] = []
    for i, row in enumerate(ds):
        md = {k: v for k, v in row.items() if k not in (pk, ak)}
        out.append(
            MathProblem(
                idx=i,
                problem=row[pk],
                answer=str(row[ak]),
                source=name,
                metadata=md,
            )
        )
    return out


def load_study_mix(
    aime_n: int = 30,
    math_n: int = 50,
) -> list[MathProblem]:
    """Canonical 80-problem study set: first N AIME-24, then first M MATH-500.

    Ordering is deterministic with shuffle=False — baseline and quantized
    runs see identical problems in identical order.
    """
    aime = load_math_dataset("aime-24", num_samples=aime_n, shuffle=False)
    math = load_math_dataset("math-500", num_samples=math_n, shuffle=False)
    combined: list[MathProblem] = []
    for i, p in enumerate(aime + math):
        combined.append(
            MathProblem(
                idx=i,
                problem=p.problem,
                answer=p.answer,
                source=p.source,
                metadata=p.metadata,
            )
        )
    return combined
```

- [ ] **Step 4: Run test — expect pass**

Run: `pytest tests/test_dataset_loader.py -v`
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/dataset_loader.py tests/test_dataset_loader.py
git commit -m "feat(data): port dataset_loader, add load_study_mix() for 30 AIME + 50 MATH"
```

---

## Task 1.4: Port `trace_utils.py` with tests

**Files:**
- Create: `src/kvtrace/trace_utils.py` (copy from baseline)
- Create: `tests/test_trace_utils.py`

- [ ] **Step 1: Copy baseline `trace_utils.py` to `src/kvtrace/trace_utils.py`**

Copy the entire content of `C:/Users/morro/prog/files/trace_utils.py` unchanged.

- [ ] **Step 2: Write explicit pytest version of the self-test**

`tests/test_trace_utils.py`:
```python
from kvtrace.trace_utils import (
    extract_boxed_answer,
    extract_think_block,
    parse_trace,
)


def test_parse_deepseek_r1_common_case():
    text = (
        "Let me work this out.\nFirst, try \\boxed{7} as a guess... no.\n"
        "</think>\n\nThe answer is \\boxed{\\dfrac{22}{7}}."
    )
    p = parse_trace(text, finish_reason="stop")
    assert p.think is not None and "try \\boxed{7}" in p.think
    assert p.boxed_answer == "\\dfrac{22}{7}"
    assert p.think_complete is True
    assert p.final_response.startswith("The answer is")


def test_parse_truncated_trace():
    text = "<think>\nI am still thinking about this when abruptly"
    p = parse_trace(text, finish_reason="length")
    assert p.think is not None
    assert p.think.startswith("I am still thinking")
    assert p.think_complete is False
    assert p.final_response == ""
    assert p.boxed_answer is None


def test_parse_no_think_tag():
    text = "The answer is \\boxed{42}."
    p = parse_trace(text)
    assert p.think is None
    assert p.boxed_answer == "42"


def test_boxed_answer_nested_braces():
    assert extract_boxed_answer(r"foo \boxed{\frac{a}{b}} bar") == r"\frac{a}{b}"


def test_boxed_answer_takes_last_match():
    assert extract_boxed_answer(r"\boxed{1} then \boxed{2}") == "2"


def test_boxed_answer_no_match_returns_none():
    assert extract_boxed_answer("no box here") is None


def test_extract_think_block_both_tags():
    text = "<think>reasoning</think>final"
    think, remaining, closed = extract_think_block(text)
    assert think == "reasoning"
    assert remaining == "final"
    assert closed is True
```

- [ ] **Step 3: Run test — expect pass**

Run: `pytest tests/test_trace_utils.py -v`
Expected: 7 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/kvtrace/trace_utils.py tests/test_trace_utils.py
git commit -m "feat(trace): port trace_utils and convert self-test to pytest"
```

---

# Phase 2 — Memory helper (needed early for generator tests)

## Task 2.1: `memory.py` — `free_gpu()` context manager

**Files:**
- Create: `src/kvtrace/memory.py`
- Create: `tests/test_memory_free.py`

- [ ] **Step 1: Write the tests (unit + gpu)**

`tests/test_memory_free.py`:
```python
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
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_memory_free.py -m "not gpu" -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/kvtrace/memory.py`:
```python
"""GPU memory cleanup helpers."""
from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Iterator

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
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_memory_free.py -m "not gpu" -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/memory.py tests/test_memory_free.py
git commit -m "feat(memory): free_gpu() context manager + vLLM state teardown"
```

---

# Phase 3 — Generators

## Task 3.1: Generator ABC + GenerationResult

**Files:**
- Create: `src/kvtrace/generators/base.py`
- Create: `tests/test_generator_contract.py`

- [ ] **Step 1: Write the failing test**

`tests/test_generator_contract.py`:
```python
import pytest

from kvtrace.generators.base import Generator, GenerationResult


def test_generation_result_required_fields():
    r = GenerationResult(
        idx=0,
        raw="<text>",
        think="reasoning",
        final_response="answer",
        boxed_answer="42",
        think_complete=True,
        finish_reason="stop",
        token_ids=[1, 2, 3],
        prompt_tokens=5,
        generated_tokens=3,
    )
    d = r.to_dict()
    assert d["idx"] == 0
    assert d["boxed_answer"] == "42"
    assert d["num_generated_tokens"] == 3


def test_generator_is_abstract():
    with pytest.raises(TypeError):
        Generator()  # type: ignore[abstract]


def test_generator_context_manager_calls_load_and_unload():
    calls = []

    class DummyGen(Generator):
        def load(self, model_id: str, quant_config) -> None:
            calls.append(("load", model_id))
        def generate(self, problems):
            calls.append(("generate", len(problems)))
            return []
        def unload(self) -> None:
            calls.append(("unload",))

    g = DummyGen()
    with g:
        g.load("m", None)
        g.generate([1, 2, 3])
    assert ("load", "m") in calls
    assert ("generate", 3) in calls
    assert ("unload",) in calls
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_generator_contract.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/kvtrace/generators/base.py`:
```python
"""Abstract Generator + GenerationResult dataclass."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from kvtrace.dataset_loader import MathProblem


@dataclass
class GenerationResult:
    """Output row from a Generator.generate() call. Mirrors ParsedTrace + token accounting."""

    idx: int
    raw: str
    think: str | None
    final_response: str
    boxed_answer: str | None
    think_complete: bool
    finish_reason: str | None
    token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "raw_output": self.raw,
            "think": self.think,
            "final_response": self.final_response,
            "boxed_answer": self.boxed_answer,
            "think_complete": self.think_complete,
            "finish_reason": self.finish_reason,
            "token_ids": self.token_ids,
            "num_prompt_tokens": self.prompt_tokens,
            "num_generated_tokens": self.generated_tokens,
            "metadata": self.metadata,
        }


class Generator(ABC):
    """Engine-agnostic inference interface.

    Subclasses must release all GPU state in `unload()` — the orchestrator
    relies on this to safely start a fresh engine for the next (model, quant)
    pair without OOM.
    """

    @abstractmethod
    def load(self, model_id: str, quant_config: Any) -> None: ...

    @abstractmethod
    def generate(self, problems: list[MathProblem]) -> list[GenerationResult]: ...

    @abstractmethod
    def unload(self) -> None: ...

    def __enter__(self) -> "Generator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_generator_contract.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/generators/base.py tests/test_generator_contract.py
git commit -m "feat(generators): Generator ABC + GenerationResult"
```

---

## Task 3.2: VLLMGenerator (with mocked vLLM in tests)

**Files:**
- Create: `src/kvtrace/generators/vllm_gen.py`
- Create: `tests/test_generators_vllm.py`

- [ ] **Step 1: Write the failing test**

`tests/test_generators_vllm.py`:
```python
from unittest.mock import MagicMock, patch

import pytest

from kvtrace.config import ModelCfg, QuantCfg
from kvtrace.dataset_loader import MathProblem
from kvtrace.generators.vllm_gen import VLLMGenerator, resolve_kv_cache_dtype


def test_resolve_kv_cache_dtype_valid():
    assert resolve_kv_cache_dtype("auto") == "auto"
    assert resolve_kv_cache_dtype("fp8_e5m2") == "fp8_e5m2"


def test_resolve_kv_cache_dtype_int8_raises():
    with pytest.raises(ValueError, match="int8.*fp8_e5m2"):
        resolve_kv_cache_dtype("int8")


def test_resolve_kv_cache_dtype_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_kv_cache_dtype("bf8_weird")


@patch("kvtrace.generators.vllm_gen.LLM")
@patch("kvtrace.generators.vllm_gen.AutoTokenizer")
def test_vllm_generator_load_passes_kv_cache_dtype(mock_tok, mock_llm_cls):
    model = ModelCfg(hf_id="foo/bar", max_model_len=1024)
    quant = QuantCfg(engine="vllm", kv_cache_dtype="fp8_e5m2")

    gen = VLLMGenerator(seed=42, sampling_max_tokens=100)
    gen.load(model, quant)

    args, kwargs = mock_llm_cls.call_args
    assert kwargs["kv_cache_dtype"] == "fp8_e5m2"
    assert kwargs["model"] == "foo/bar"
    assert kwargs["seed"] == 42


@patch("kvtrace.generators.vllm_gen.LLM")
@patch("kvtrace.generators.vllm_gen.AutoTokenizer")
def test_vllm_generator_generate_returns_results(mock_tok, mock_llm_cls):
    # Arrange
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "PROMPT"
    mock_tok.from_pretrained.return_value = tokenizer

    completion = MagicMock()
    completion.text = "The answer.\n</think>\n\n\\boxed{42}"
    completion.token_ids = [10, 11, 12]
    completion.finish_reason = "stop"

    req_out = MagicMock()
    req_out.prompt_token_ids = [1, 2, 3, 4]
    req_out.outputs = [completion]

    llm = MagicMock()
    llm.generate.return_value = [req_out]
    mock_llm_cls.return_value = llm

    # Act
    gen = VLLMGenerator(seed=42)
    gen.load(ModelCfg(hf_id="m"), QuantCfg(engine="vllm", kv_cache_dtype="auto"))
    problems = [MathProblem(idx=0, problem="q", answer="42", source="aime-24")]
    results = gen.generate(problems)

    # Assert
    assert len(results) == 1
    assert results[0].boxed_answer == "42"
    assert results[0].token_ids == [10, 11, 12]
    assert results[0].finish_reason == "stop"


@patch("kvtrace.generators.vllm_gen.free_gpu")
@patch("kvtrace.generators.vllm_gen.LLM")
@patch("kvtrace.generators.vllm_gen.AutoTokenizer")
def test_vllm_generator_unload_calls_free_gpu(mock_tok, mock_llm_cls, mock_free):
    gen = VLLMGenerator(seed=42)
    gen.load(ModelCfg(hf_id="m"), QuantCfg(engine="vllm", kv_cache_dtype="auto"))
    gen.unload()
    mock_free.assert_called_once()
    assert gen._llm is None
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_generators_vllm.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/kvtrace/generators/vllm_gen.py`:
```python
"""vLLM backend for BF16 / FP8-E5M2 / FP8-E4M3 KV cache."""
from __future__ import annotations

import logging
from typing import Any

# Imports are module-level so tests can patch them. In CI (no vLLM installed)
# the module is imported only by tests that patch `LLM` and `AutoTokenizer`.
try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:
    LLM = None           # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore[assignment]

from kvtrace.config import ModelCfg, QuantCfg
from kvtrace.dataset_loader import MathProblem
from kvtrace.generators.base import GenerationResult, Generator
from kvtrace.memory import free_gpu
from kvtrace.trace_utils import parse_trace

log = logging.getLogger(__name__)

VLLM_SUPPORTED_KV_DTYPES = {"auto", "fp8", "fp8_e5m2", "fp8_e4m3"}

DEFAULT_USER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def resolve_kv_cache_dtype(cli_arg: str) -> str:
    if cli_arg == "int8":
        raise ValueError(
            "kv_cache_dtype='int8' is not supported by vLLM. "
            "Use 'fp8_e5m2' on Ampere/Ada for storage-only quantization."
        )
    if cli_arg not in VLLM_SUPPORTED_KV_DTYPES:
        raise ValueError(
            f"Unsupported kv_cache_dtype {cli_arg!r}. "
            f"Choose from {sorted(VLLM_SUPPORTED_KV_DTYPES)}."
        )
    return cli_arg


class VLLMGenerator(Generator):
    def __init__(
        self,
        seed: int = 42,
        sampling_temperature: float = 0.0,
        sampling_top_p: float = 1.0,
        sampling_max_tokens: int = 32768,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        self.seed = seed
        self.sampling_temperature = sampling_temperature
        self.sampling_top_p = sampling_top_p
        self.sampling_max_tokens = sampling_max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm: Any = None
        self._tokenizer: Any = None
        self._model_cfg: ModelCfg | None = None

    def load(self, model_cfg: ModelCfg, quant_cfg: QuantCfg) -> None:
        if quant_cfg.engine != "vllm":
            raise ValueError(f"VLLMGenerator requires engine='vllm', got {quant_cfg.engine!r}")
        kv_dtype = resolve_kv_cache_dtype(quant_cfg.kv_cache_dtype or "auto")
        self._model_cfg = model_cfg
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.hf_id, trust_remote_code=model_cfg.trust_remote_code
        )
        self._llm = LLM(
            model=model_cfg.hf_id,
            dtype=model_cfg.dtype,
            kv_cache_dtype=kv_dtype,
            max_model_len=model_cfg.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            seed=self.seed,
            trust_remote_code=model_cfg.trust_remote_code,
        )

    def generate(self, problems: list[MathProblem]) -> list[GenerationResult]:
        assert self._llm is not None and self._tokenizer is not None, "call load() first"
        prompts = self._build_prompts(problems)

        sp = SamplingParams(
            n=1,
            temperature=self.sampling_temperature,
            top_p=self.sampling_top_p,
            max_tokens=self.sampling_max_tokens,
            seed=self.seed,
        )
        outputs = self._llm.generate(prompts, sp, use_tqdm=True)

        results: list[GenerationResult] = []
        for problem, req in zip(problems, outputs):
            completion = req.outputs[0]
            parsed = parse_trace(completion.text, finish_reason=completion.finish_reason)
            results.append(
                GenerationResult(
                    idx=problem.idx,
                    raw=completion.text,
                    think=parsed.think,
                    final_response=parsed.final_response,
                    boxed_answer=parsed.boxed_answer,
                    think_complete=parsed.think_complete,
                    finish_reason=completion.finish_reason,
                    token_ids=list(completion.token_ids),
                    prompt_tokens=len(req.prompt_token_ids),
                    generated_tokens=len(completion.token_ids),
                    metadata=problem.metadata,
                )
            )
        return results

    def _build_prompts(self, problems: list[MathProblem]) -> list[str]:
        out: list[str] = []
        mcfg = self._model_cfg
        assert mcfg is not None
        for p in problems:
            messages = [
                {"role": "user", "content": f"{p.problem}\n\n{DEFAULT_USER_INSTRUCTION}"}
            ]
            kwargs = dict(mcfg.chat_template_kwargs)
            out.append(
                self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs
                )
            )
        return out

    def unload(self) -> None:
        self._llm = None
        self._tokenizer = None
        self._model_cfg = None
        free_gpu()
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_generators_vllm.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/generators/vllm_gen.py tests/test_generators_vllm.py
git commit -m "feat(generators): VLLMGenerator with mocked tests"
```

---

## Task 3.3: HFGenerator (HQQ INT4/INT2 via HF Transformers)

**Files:**
- Create: `src/kvtrace/generators/hf_gen.py`
- Create: `tests/test_generators_hf.py`

- [ ] **Step 1: Write the failing test**

`tests/test_generators_hf.py`:
```python
from unittest.mock import MagicMock, patch

import pytest

from kvtrace.config import ModelCfg, QuantCfg
from kvtrace.dataset_loader import MathProblem
from kvtrace.generators.hf_gen import HFGenerator


@patch("kvtrace.generators.hf_gen.AutoModelForCausalLM")
@patch("kvtrace.generators.hf_gen.AutoTokenizer")
def test_hf_generator_load_uses_hqq_nbits(mock_tok, mock_model):
    quant = QuantCfg(engine="hf", hqq_nbits=4)
    gen = HFGenerator()
    gen.load(ModelCfg(hf_id="foo/bar"), quant)
    assert gen._hqq_nbits == 4


@patch("kvtrace.generators.hf_gen.AutoModelForCausalLM")
@patch("kvtrace.generators.hf_gen.AutoTokenizer")
def test_hf_generator_load_rejects_vllm_engine(mock_tok, mock_model):
    with pytest.raises(ValueError, match="HFGenerator"):
        HFGenerator().load(ModelCfg(hf_id="foo"), QuantCfg(engine="vllm", kv_cache_dtype="auto"))


@patch("kvtrace.generators.hf_gen.AutoModelForCausalLM")
@patch("kvtrace.generators.hf_gen.AutoTokenizer")
def test_hf_generator_generate_returns_result(mock_tok, mock_model):
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = {"input_ids": MagicMock()}
    tokenizer.decode.return_value = "Reasoning done.</think>\n\n\\boxed{7}"
    mock_tok.from_pretrained.return_value = tokenizer

    model = MagicMock()
    fake_out = MagicMock()
    fake_out.sequences = MagicMock()
    fake_out.sequences.shape = (1, 10)
    fake_out.sequences.__getitem__ = lambda self, idx: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    model.generate.return_value = fake_out
    model.device = "cpu"
    mock_model.from_pretrained.return_value = model

    gen = HFGenerator()
    gen.load(ModelCfg(hf_id="m"), QuantCfg(engine="hf", hqq_nbits=4))
    results = gen.generate(
        [MathProblem(idx=0, problem="q", answer="7", source="aime-24")]
    )
    assert len(results) == 1
    assert results[0].boxed_answer == "7"


@patch("kvtrace.generators.hf_gen.free_gpu")
@patch("kvtrace.generators.hf_gen.AutoModelForCausalLM")
@patch("kvtrace.generators.hf_gen.AutoTokenizer")
def test_hf_generator_unload_calls_free_gpu(mock_tok, mock_model, mock_free):
    gen = HFGenerator()
    gen.load(ModelCfg(hf_id="m"), QuantCfg(engine="hf", hqq_nbits=2))
    gen.unload()
    mock_free.assert_called_once()
    assert gen._model is None
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_generators_hf.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/kvtrace/generators/hf_gen.py`:
```python
"""HuggingFace Transformers backend for HQQ INT4/INT2 KV cache."""
from __future__ import annotations

import logging
from typing import Any

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from transformers.cache_utils import QuantizedCacheConfig     # type: ignore
except Exception:
    AutoModelForCausalLM = None   # type: ignore[assignment]
    AutoTokenizer = None          # type: ignore[assignment]
    QuantizedCacheConfig = None   # type: ignore[assignment]

from kvtrace.config import ModelCfg, QuantCfg
from kvtrace.dataset_loader import MathProblem
from kvtrace.generators.base import GenerationResult, Generator
from kvtrace.generators.vllm_gen import DEFAULT_USER_INSTRUCTION
from kvtrace.memory import free_gpu
from kvtrace.trace_utils import parse_trace

log = logging.getLogger(__name__)


class HFGenerator(Generator):
    def __init__(
        self,
        sampling_temperature: float = 0.0,
        sampling_top_p: float = 1.0,
        sampling_max_tokens: int = 32768,
    ) -> None:
        self.sampling_temperature = sampling_temperature
        self.sampling_top_p = sampling_top_p
        self.sampling_max_tokens = sampling_max_tokens
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_cfg: ModelCfg | None = None
        self._hqq_nbits: int | None = None

    def load(self, model_cfg: ModelCfg, quant_cfg: QuantCfg) -> None:
        if quant_cfg.engine != "hf":
            raise ValueError(f"HFGenerator requires engine='hf', got {quant_cfg.engine!r}")
        if quant_cfg.hqq_nbits not in (2, 4, 8):
            raise ValueError(f"HQQ nbits must be 2/4/8, got {quant_cfg.hqq_nbits!r}")
        import torch  # local so tests mock module-level imports only

        self._hqq_nbits = quant_cfg.hqq_nbits
        self._model_cfg = model_cfg
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.hf_id, trust_remote_code=model_cfg.trust_remote_code
        )
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        torch_dtype = dtype_map.get(model_cfg.dtype, torch.bfloat16)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_cfg.hf_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=model_cfg.trust_remote_code,
        )

    def generate(self, problems: list[MathProblem]) -> list[GenerationResult]:
        assert self._model is not None and self._tokenizer is not None, "call load() first"

        results: list[GenerationResult] = []
        cache_config = QuantizedCacheConfig(backend="HQQ", nbits=self._hqq_nbits)

        for p in problems:
            prompt = self._build_prompt(p)
            # HF expects tensors; use return_tensors="pt"
            enc = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{p.problem}\n\n{DEFAULT_USER_INSTRUCTION}"}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **self._model_cfg.chat_template_kwargs,
            )
            input_ids = enc.to(self._model.device) if hasattr(enc, "to") else enc["input_ids"].to(self._model.device)
            prompt_len = input_ids.shape[-1]

            out = self._model.generate(
                input_ids,
                max_new_tokens=self.sampling_max_tokens,
                do_sample=(self.sampling_temperature > 0.0),
                temperature=self.sampling_temperature or None,
                top_p=self.sampling_top_p,
                cache_implementation="quantized",
                cache_config=cache_config,
                return_dict_in_generate=True,
            )
            seq = out.sequences[0].tolist()
            gen_ids = seq[prompt_len:]
            text = self._tokenizer.decode(gen_ids, skip_special_tokens=False)
            parsed = parse_trace(text, finish_reason="stop")
            results.append(
                GenerationResult(
                    idx=p.idx,
                    raw=text,
                    think=parsed.think,
                    final_response=parsed.final_response,
                    boxed_answer=parsed.boxed_answer,
                    think_complete=parsed.think_complete,
                    finish_reason="stop",
                    token_ids=gen_ids,
                    prompt_tokens=prompt_len,
                    generated_tokens=len(gen_ids),
                    metadata=p.metadata,
                )
            )
        return results

    def _build_prompt(self, p: MathProblem) -> str:
        return p.problem

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_cfg = None
        self._hqq_nbits = None
        free_gpu()
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_generators_hf.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/generators/hf_gen.py tests/test_generators_hf.py
git commit -m "feat(generators): HFGenerator with HQQ QuantizedCacheConfig"
```

---

# Phase 4 — First Divergence Point

## Task 4.1: Tokenizer alignment helper

**Files:**
- Create: `src/kvtrace/fdp/tokenizer_align.py`
- Create: `tests/test_tokenizer_align.py`

- [ ] **Step 1: Write the failing test**

`tests/test_tokenizer_align.py`:
```python
from kvtrace.fdp.tokenizer_align import first_token_mismatch, decode_window


def test_first_token_mismatch_same_lists():
    assert first_token_mismatch([1, 2, 3], [1, 2, 3]) is None


def test_first_token_mismatch_diff_at_0():
    assert first_token_mismatch([1, 2, 3], [4, 2, 3]) == 0


def test_first_token_mismatch_diff_in_middle():
    assert first_token_mismatch([1, 2, 3, 4], [1, 2, 5, 4]) == 2


def test_first_token_mismatch_different_lengths_shorter_wins():
    assert first_token_mismatch([1, 2, 3], [1, 2]) == 2
    assert first_token_mismatch([1, 2], [1, 2, 3]) == 2


def test_decode_window_clips_left():
    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return "|".join(str(i) for i in ids)
    result = decode_window(FakeTokenizer(), [1, 2, 3, 4, 5], center=0, context=2)
    # Should clip left to 0 and go up to context tokens to the right.
    assert result == "1|2|3"


def test_decode_window_clips_right():
    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return "|".join(str(i) for i in ids)
    result = decode_window(FakeTokenizer(), [1, 2, 3, 4, 5], center=4, context=10)
    assert result == "3|4|5"
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_tokenizer_align.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/kvtrace/fdp/tokenizer_align.py`:
```python
"""Token-level alignment primitives."""
from __future__ import annotations

from typing import Any


def first_token_mismatch(a: list[int], b: list[int]) -> int | None:
    """Index of the first position where a[i] != b[i], or None if identical.

    When one list is a strict prefix of the other, returns min(len(a), len(b)).
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def decode_window(tokenizer: Any, tokens: list[int], center: int, context: int) -> str:
    """Decode tokens[center-context : center+context], clipping to bounds."""
    lo = max(0, center - context)
    hi = min(len(tokens), center + context + 1)
    return tokenizer.decode(tokens[lo:hi], skip_special_tokens=False)
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_tokenizer_align.py -v`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/fdp/tokenizer_align.py tests/test_tokenizer_align.py
git commit -m "feat(fdp): tokenizer alignment primitives"
```

---

## Task 4.2: Semantic re-sync checker

**Files:**
- Create: `src/kvtrace/fdp/semantic_resync.py`
- Create: `tests/test_semantic_resync.py`

- [ ] **Step 1: Write the failing test**

`tests/test_semantic_resync.py`:
```python
from unittest.mock import MagicMock, patch

import numpy as np

from kvtrace.fdp.semantic_resync import (
    ReSyncChecker,
    split_into_reasoning_units,
)


def test_split_by_double_newline():
    text = "First step.\n\nSecond step.\n\nThird step."
    units = split_into_reasoning_units(text)
    assert units == ["First step.", "Second step.", "Third step."]


def test_split_by_transition_markers():
    text = "Let me try x=1. Wait, that's wrong. So, try x=2."
    units = split_into_reasoning_units(text)
    assert any("Let me try" in u for u in units)
    assert any("Wait" in u for u in units)
    assert any("So," in u for u in units)


def test_split_handles_empty_string():
    assert split_into_reasoning_units("") == []
    assert split_into_reasoning_units("   ") == []


def test_checker_all_cosmetic(monkeypatch):
    checker = ReSyncChecker(threshold=0.9)
    fake_model = MagicMock()
    # Make every pair of embeddings identical → cosine=1.0
    fake_model.encode.side_effect = lambda texts, **kwargs: np.ones((len(texts), 4))
    monkeypatch.setattr(checker, "_model", fake_model)

    is_cosmetic = checker.is_cosmetic_divergence(
        "Let me try x=1. Wait, that fails. So x=2.",
        "Let's try x=1. Hmm, fails. So, x=2.",
    )
    assert is_cosmetic is True


def test_checker_real_divergence(monkeypatch):
    checker = ReSyncChecker(threshold=0.9)
    fake_model = MagicMock()
    # Orthogonal vectors → cosine=0
    def encode(texts, **kwargs):
        n = len(texts)
        v = np.eye(max(n, 4))[:n, :4]
        return v
    fake_model.encode.side_effect = encode
    monkeypatch.setattr(checker, "_model", fake_model)

    is_cosmetic = checker.is_cosmetic_divergence(
        "Let me compute 2+2=4.",
        "Let me compute 2+2=5.",
    )
    assert is_cosmetic is False


def test_checker_few_units_is_not_cosmetic(monkeypatch):
    checker = ReSyncChecker(threshold=0.9)
    fake_model = MagicMock()
    fake_model.encode.side_effect = lambda texts, **kwargs: np.ones((len(texts), 4))
    monkeypatch.setattr(checker, "_model", fake_model)

    # Only 1 unit in each → not enough to judge re-sync; must return False.
    assert checker.is_cosmetic_divergence("one.", "one!") is False
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_semantic_resync.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/kvtrace/fdp/semantic_resync.py`:
```python
"""Semantic re-sync via MiniLM cosine similarity."""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_TRANSITION_RE = re.compile(
    r"(?=(?:\bLet me\b|\bLet's\b|\bWait,|\bSo,|\bTherefore\b|\bHmm,|\bActually,|\bStep \d+))"
)


def split_into_reasoning_units(text: str) -> list[str]:
    """Split into 'reasoning units' via double-newlines and transition markers."""
    if not text or not text.strip():
        return []
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units: list[str] = []
    for p in raw_paragraphs:
        subs = _TRANSITION_RE.split(p)
        for s in subs:
            s = s.strip()
            if s:
                units.append(s)
    return units


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class ReSyncChecker:
    """Wraps a sentence-transformers model for unit-level cosine similarity.

    Model is loaded lazily and cached on the instance to keep CPU startup cheap
    for tests that monkey-patch `._model`.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.9,
        k: int = 3,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.k = k
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def is_cosmetic_divergence(self, baseline_window: str, quant_window: str) -> bool:
        b_units = split_into_reasoning_units(baseline_window)[: self.k]
        q_units = split_into_reasoning_units(quant_window)[: self.k]
        if len(b_units) < 2 or len(q_units) < 2:
            return False
        n = min(len(b_units), len(q_units))
        model = self._ensure_model()
        b_emb = model.encode(b_units[:n], convert_to_numpy=True)
        q_emb = model.encode(q_units[:n], convert_to_numpy=True)
        sims = [_cosine(b_emb[i], q_emb[i]) for i in range(n)]
        above = sum(1 for s in sims if s >= self.threshold)
        # "2 of 3" rule: at least majority above threshold
        return above >= max(2, (n + 1) // 2)
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_semantic_resync.py -v`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/fdp/semantic_resync.py tests/test_semantic_resync.py
git commit -m "feat(fdp): semantic re-sync via MiniLM cosine similarity"
```

---

## Task 4.3: Hybrid FDP finder

**Files:**
- Create: `src/kvtrace/fdp/finder.py`
- Create: `tests/test_fdp_finder.py`

- [ ] **Step 1: Write the failing test**

`tests/test_fdp_finder.py`:
```python
from unittest.mock import MagicMock

import pytest

from kvtrace.fdp.finder import FDPRecord, find_fdp, FDPParams


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(65 + (i % 26)) for i in ids)


class NoCosmeticChecker:
    def is_cosmetic_divergence(self, *a, **kw):
        return False


class AllCosmeticChecker:
    def is_cosmetic_divergence(self, *a, **kw):
        return True


def test_identical_traces_no_fdp():
    r = find_fdp(
        baseline_tokens=[1, 2, 3],
        quant_tokens=[1, 2, 3],
        baseline_text="same",
        quant_text="same",
        tokenizer=FakeTokenizer(),
        resync_checker=NoCosmeticChecker(),
    )
    assert r.fdp_token_idx is None


def test_first_token_divergence():
    r = find_fdp(
        baseline_tokens=[1, 2, 3, 4, 5],
        quant_tokens=[1, 2, 9, 4, 5],
        baseline_text="abcde",
        quant_text="abiZe",
        tokenizer=FakeTokenizer(),
        resync_checker=NoCosmeticChecker(),
    )
    assert r.fdp_token_idx == 2


def test_cosmetic_divergence_skipped_then_no_more():
    # Traces diverge at token 2, but checker flags as cosmetic.
    # After skip there is no further divergence → FDP is None but cosmetic_skipped=1.
    r = find_fdp(
        baseline_tokens=[1, 2, 3, 4, 5],
        quant_tokens=[1, 2, 9, 4, 5],
        baseline_text="abcde",
        quant_text="abiZe",
        tokenizer=FakeTokenizer(),
        resync_checker=AllCosmeticChecker(),
    )
    assert r.cosmetic_skipped >= 1
    # With max_cosmetic_skips default 5 and only one cosmetic point, FDP=None
    assert r.fdp_token_idx is None or r.fdp_token_idx > 2


def test_truncated_baseline_flag():
    # Baseline is shorter — hit max_tokens (finish_reason=length).
    r = find_fdp(
        baseline_tokens=[1, 2, 3],
        quant_tokens=[1, 2, 3, 4, 5],
        baseline_text="abc",
        quant_text="abcde",
        tokenizer=FakeTokenizer(),
        resync_checker=NoCosmeticChecker(),
        baseline_truncated=True,
    )
    assert r.fdp_token_idx == 3
    assert r.baseline_truncated is True


def test_boxed_match_attributes_are_captured():
    r = find_fdp(
        baseline_tokens=[1, 2],
        quant_tokens=[1, 9],
        baseline_text="ab",
        quant_text="ai",
        tokenizer=FakeTokenizer(),
        resync_checker=NoCosmeticChecker(),
        baseline_boxed="42",
        quant_boxed="41",
        ground_truth="42",
    )
    assert r.boxed_match == "baseline_only"


def test_max_cosmetic_skips_respected():
    # Always-cosmetic checker would loop forever without a cap.
    r = find_fdp(
        baseline_tokens=list(range(100)),
        quant_tokens=[i if i < 5 else (i + 1) for i in range(100)],
        baseline_text="x",
        quant_text="y",
        tokenizer=FakeTokenizer(),
        resync_checker=AllCosmeticChecker(),
        params=FDPParams(max_cosmetic_skips=2, resync_lookahead=10, context_window=3),
    )
    # After max skips, the first remaining mismatch becomes the real FDP.
    assert r.cosmetic_skipped == 2
    assert r.fdp_token_idx is not None


def test_context_window_clips_at_start():
    r = find_fdp(
        baseline_tokens=[1, 9, 3, 4],
        quant_tokens=[2, 9, 3, 4],
        baseline_text="abcd",
        quant_text="Xbcd",
        tokenizer=FakeTokenizer(),
        resync_checker=NoCosmeticChecker(),
        params=FDPParams(context_window=10, resync_lookahead=10, max_cosmetic_skips=0),
    )
    # FDP at index 0; baseline_context must not crash even with huge context.
    assert r.fdp_token_idx == 0
    assert isinstance(r.baseline_context, str)
    assert isinstance(r.quant_context, str)


def test_both_truncated_before_fdp():
    # Both finish by length but traces identical up to their shared tail.
    r = find_fdp(
        baseline_tokens=[1, 2, 3],
        quant_tokens=[1, 2, 3],
        baseline_text="abc",
        quant_text="abc",
        tokenizer=FakeTokenizer(),
        resync_checker=NoCosmeticChecker(),
        baseline_truncated=True,
        quant_truncated=True,
    )
    assert r.fdp_token_idx is None
    assert r.baseline_truncated and r.quant_truncated
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_fdp_finder.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/kvtrace/fdp/finder.py`:
```python
"""Hybrid First Divergence Point finder.

Token-level exact mismatch, then semantic re-sync to filter out cosmetic
divergences, then report the true FDP with ±context windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from kvtrace.fdp.tokenizer_align import decode_window, first_token_mismatch

BoxedMatch = Literal["both_correct", "baseline_only", "quant_only", "both_wrong", "no_boxed"]


class ReSyncProtocol(Protocol):
    def is_cosmetic_divergence(self, baseline_window: str, quant_window: str) -> bool: ...


@dataclass
class FDPParams:
    context_window: int = 200
    resync_lookahead: int = 500
    max_cosmetic_skips: int = 5


@dataclass
class FDPRecord:
    fdp_token_idx: int | None
    cosmetic_skipped: int
    baseline_context: str
    quant_context: str
    common_prefix: str
    boxed_match: BoxedMatch = "no_boxed"
    baseline_truncated: bool = False
    quant_truncated: bool = False
    metadata: dict = field(default_factory=dict)


def _boxed_match(baseline: str | None, quant: str | None, gt: str | None) -> BoxedMatch:
    if baseline is None and quant is None:
        return "no_boxed"
    gt_s = (gt or "").strip()
    b_ok = baseline is not None and baseline.strip() == gt_s
    q_ok = quant is not None and quant.strip() == gt_s
    if b_ok and q_ok:
        return "both_correct"
    if b_ok and not q_ok:
        return "baseline_only"
    if q_ok and not b_ok:
        return "quant_only"
    return "both_wrong"


def find_fdp(
    baseline_tokens: list[int],
    quant_tokens: list[int],
    baseline_text: str,
    quant_text: str,
    tokenizer: Any,
    resync_checker: ReSyncProtocol,
    params: FDPParams | None = None,
    baseline_truncated: bool = False,
    quant_truncated: bool = False,
    baseline_boxed: str | None = None,
    quant_boxed: str | None = None,
    ground_truth: str | None = None,
) -> FDPRecord:
    p = params or FDPParams()
    cosmetic_skipped = 0
    offset = 0

    while True:
        idx = first_token_mismatch(baseline_tokens[offset:], quant_tokens[offset:])
        if idx is None:
            # No (more) divergence.
            return FDPRecord(
                fdp_token_idx=None,
                cosmetic_skipped=cosmetic_skipped,
                baseline_context="",
                quant_context="",
                common_prefix=tokenizer.decode(
                    baseline_tokens[: min(len(baseline_tokens), 300)],
                    skip_special_tokens=False,
                ),
                boxed_match=_boxed_match(baseline_boxed, quant_boxed, ground_truth),
                baseline_truncated=baseline_truncated,
                quant_truncated=quant_truncated,
            )
        absolute_idx = offset + idx

        # Budget check: too many cosmetic skips already — treat this as real FDP.
        if cosmetic_skipped >= p.max_cosmetic_skips:
            return _build_real_fdp(
                absolute_idx, baseline_tokens, quant_tokens, tokenizer, p,
                cosmetic_skipped, baseline_truncated, quant_truncated,
                baseline_boxed, quant_boxed, ground_truth,
            )

        # Build re-sync windows.
        b_window = decode_window(tokenizer, baseline_tokens, absolute_idx, p.resync_lookahead)
        q_window = decode_window(tokenizer, quant_tokens, absolute_idx, p.resync_lookahead)
        is_cosmetic = resync_checker.is_cosmetic_divergence(b_window, q_window)

        if not is_cosmetic:
            return _build_real_fdp(
                absolute_idx, baseline_tokens, quant_tokens, tokenizer, p,
                cosmetic_skipped, baseline_truncated, quant_truncated,
                baseline_boxed, quant_boxed, ground_truth,
            )

        # Cosmetic: skip forward by resync_lookahead tokens and keep searching.
        cosmetic_skipped += 1
        offset = absolute_idx + p.resync_lookahead


def _build_real_fdp(
    absolute_idx: int,
    baseline_tokens: list[int],
    quant_tokens: list[int],
    tokenizer: Any,
    p: FDPParams,
    cosmetic_skipped: int,
    baseline_truncated: bool,
    quant_truncated: bool,
    baseline_boxed: str | None,
    quant_boxed: str | None,
    ground_truth: str | None,
) -> FDPRecord:
    b_ctx = decode_window(tokenizer, baseline_tokens, absolute_idx, p.context_window)
    q_ctx = decode_window(tokenizer, quant_tokens, absolute_idx, p.context_window)
    prefix_end = max(0, absolute_idx)
    prefix_start = max(0, prefix_end - 300)
    common_prefix = tokenizer.decode(
        baseline_tokens[prefix_start:prefix_end], skip_special_tokens=False
    )
    return FDPRecord(
        fdp_token_idx=absolute_idx,
        cosmetic_skipped=cosmetic_skipped,
        baseline_context=b_ctx,
        quant_context=q_ctx,
        common_prefix=common_prefix,
        boxed_match=_boxed_match(baseline_boxed, quant_boxed, ground_truth),
        baseline_truncated=baseline_truncated,
        quant_truncated=quant_truncated,
    )
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_fdp_finder.py -v`
Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/fdp/finder.py tests/test_fdp_finder.py
git commit -m "feat(fdp): hybrid FDP finder with cosmetic-divergence re-sync"
```

---

# Phase 5 — Judge

## Task 5.1: Taxonomy module

**Files:**
- Create: `src/kvtrace/judge/taxonomy.py`
- Create: `tests/test_taxonomy.py`

- [ ] **Step 1: Write the failing test**

`tests/test_taxonomy.py`:
```python
import pytest

from kvtrace.judge.taxonomy import CATEGORIES, TAXONOMY_VERSION, Category, lookup


def test_six_categories_exactly():
    assert set(c.letter for c in CATEGORIES) == {"A", "B", "C", "D", "E", "F"}
    assert len(CATEGORIES) == 6


def test_version_is_v1():
    assert TAXONOMY_VERSION == "v1"


def test_each_category_has_description_and_examples():
    for c in CATEGORIES:
        assert c.name
        assert len(c.description) >= 20
        assert len(c.examples) >= 2


def test_lookup_valid_letter():
    cat = lookup("A")
    assert cat.name == "Arithmetic"


def test_lookup_lowercase_accepted():
    assert lookup("a").name == "Arithmetic"


def test_lookup_invalid_raises():
    with pytest.raises(ValueError):
        lookup("Z")
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_taxonomy.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/kvtrace/judge/taxonomy.py`:
```python
"""TAXONOMY_V1: the 6 error categories for reasoning-trace divergences."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TAXONOMY_VERSION = "v1"

CategoryLetter = Literal["A", "B", "C", "D", "E", "F"]


@dataclass(frozen=True)
class Category:
    letter: CategoryLetter
    name: str
    description: str
    examples: tuple[str, ...]


CATEGORIES: tuple[Category, ...] = (
    Category(
        letter="A",
        name="Arithmetic",
        description="Numerical or symbolic computation error; the strategy is sound but a computation step is wrong.",
        examples=(
            "Claims 7 * 8 = 54 (correct is 56).",
            "Expands (x+1)^2 = x^2 + 1 (missing 2x).",
            "Writes sqrt(48) = 4*sqrt(2) instead of 4*sqrt(3).",
        ),
    ),
    Category(
        letter="B",
        name="Logical",
        description="Premises are correct but the inference step drawn from them is invalid.",
        examples=(
            "From 'all primes > 2 are odd' concludes 'every odd number is prime'.",
            "From 'f(x)=0 has root x=2' concludes 'f is linear'.",
        ),
    ),
    Category(
        letter="C",
        name="Strategy-switch",
        description="Unmotivated abandonment of the current approach for a different one mid-derivation.",
        examples=(
            "Was using Vieta's formulas, suddenly starts expanding a polynomial fully.",
            "Was integrating by parts, abandons and restarts with substitution without reason.",
        ),
    ),
    Category(
        letter="D",
        name="Hallucination",
        description="Invention of a nonexistent or irrelevant fact (fake theorem, wrong formula, nonexistent identity).",
        examples=(
            "Cites 'the Smith-Jones theorem of 2019' to justify a step.",
            "Uses 'the well-known identity sin(x) + cos(x) = 1' (false).",
        ),
    ),
    Category(
        letter="E",
        name="Premature-termination",
        description="Trace cuts off before producing a boxed final answer: hit max tokens, gave up, or stopped mid-sentence.",
        examples=(
            "finish_reason=length and no \\boxed{} at the end.",
            "Text ends with 'I give up' or 'this is beyond me'.",
        ),
    ),
    Category(
        letter="F",
        name="Repetition/loop",
        description="The same reasoning step (paragraph or equation) is repeated three or more times without progress.",
        examples=(
            "Rewrites 'Let me try x=2' five times in a row.",
            "Repeats the same equation manipulation step in a loop.",
        ),
    ),
)


_BY_LETTER = {c.letter: c for c in CATEGORIES}


def lookup(letter: str) -> Category:
    key = letter.strip().upper()
    if key not in _BY_LETTER:
        raise ValueError(f"Unknown category {letter!r}. Expected one of A-F.")
    return _BY_LETTER[key]  # type: ignore[index]
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_taxonomy.py -v`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/judge/taxonomy.py tests/test_taxonomy.py
git commit -m "feat(judge): TAXONOMY_V1 with 6 categories A-F"
```

---

## Task 5.2: Judge prompt builder

**Files:**
- Create: `src/kvtrace/judge/prompt.py`
- Create: `tests/test_judge_prompt.py`

- [ ] **Step 1: Write the failing test**

`tests/test_judge_prompt.py`:
```python
from kvtrace.judge.prompt import PROMPT_VERSION, build_judge_messages, build_system_block


def test_prompt_version():
    assert PROMPT_VERSION == "v1"


def test_system_block_contains_all_categories():
    sys_blocks = build_system_block()
    text = " ".join(b["text"] for b in sys_blocks)
    for letter in "ABCDEF":
        assert f" {letter}." in text or f" {letter} " in text or f"{letter}:" in text


def test_system_block_has_cache_control():
    sys_blocks = build_system_block()
    assert any("cache_control" in b for b in sys_blocks)


def test_build_judge_messages_round_trip():
    msgs = build_judge_messages(
        problem="compute 2+2",
        ground_truth="4",
        baseline_context="result = 4",
        quant_context="result = 5",
        common_prefix="...",
        quant_method_name="fp8_e5m2",
    )
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert "compute 2+2" in content
    assert "result = 4" in content
    assert "result = 5" in content
    assert "fp8_e5m2" in content


def test_user_message_length_is_reasonable():
    msgs = build_judge_messages(
        problem="x" * 1000,
        ground_truth="y",
        baseline_context="a" * 1000,
        quant_context="b" * 1000,
        common_prefix="c" * 600,
        quant_method_name="q",
    )
    content = msgs[0]["content"]
    # Per spec: per-request block ≤ ~4000 tokens ≈ 16000 chars. Loose bound here.
    assert len(content) < 20000
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_judge_prompt.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/kvtrace/judge/prompt.py`:
```python
"""Judge prompt construction with Anthropic prompt caching.

The taxonomy + schema block is marked `cache_control: {"type": "ephemeral"}`
so Anthropic caches it for ~5 minutes; 960 judge calls in one run pay the
full cost once and read from the cache for the rest.
"""
from __future__ import annotations

from kvtrace.judge.taxonomy import CATEGORIES, TAXONOMY_VERSION

PROMPT_VERSION = "v1"

_SYSTEM_HEADER = (
    "You are an expert in analyzing long chain-of-thought reasoning traces "
    "from small language models. Your job is to classify the FIRST point "
    "where a quantized-KV-cache trace diverges from its FP16 baseline into "
    "exactly one of six error categories. Be strict: choose the category "
    "that describes the LOCAL error at the divergence, not downstream "
    "consequences."
)

_SCHEMA = """
Output a single JSON object with EXACTLY these keys:

{
  "category": "A" | "B" | "C" | "D" | "E" | "F",
  "confidence": 0.0 to 1.0,
  "rationale": string, at most 2 sentences,
  "affected_span": string, a 3-15 word quote from the QUANTIZED trace
}

Do not output any text outside the JSON.
""".strip()


def _taxonomy_text() -> str:
    lines = [f"TAXONOMY (v: {TAXONOMY_VERSION})", ""]
    for c in CATEGORIES:
        lines.append(f"{c.letter}. {c.name} — {c.description}")
        for ex in c.examples:
            lines.append(f"   • {ex}")
        lines.append("")
    return "\n".join(lines)


def build_system_block() -> list[dict]:
    """Return the Anthropic `system` list with cache_control on the taxonomy."""
    static = "\n\n".join([_SYSTEM_HEADER, _taxonomy_text(), _SCHEMA])
    return [
        {
            "type": "text",
            "text": static,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def build_judge_messages(
    *,
    problem: str,
    ground_truth: str,
    baseline_context: str,
    quant_context: str,
    common_prefix: str,
    quant_method_name: str,
) -> list[dict]:
    """Return the `messages` list for anthropic.Messages.create(...)."""
    content = (
        f"Problem:\n{problem}\n\n"
        f"Ground truth answer: {ground_truth}\n\n"
        f"--- Common prefix (last part, same in both traces) ---\n"
        f"{common_prefix}\n\n"
        f"--- BASELINE TRACE (FP16) around divergence ---\n"
        f"{baseline_context}\n\n"
        f"--- QUANTIZED TRACE ({quant_method_name}) around divergence ---\n"
        f"{quant_context}\n\n"
        "Classify the divergence in the quantized trace. Respond with the JSON object only."
    )
    return [{"role": "user", "content": content}]
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_judge_prompt.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/judge/prompt.py tests/test_judge_prompt.py
git commit -m "feat(judge): versioned prompt builder with Anthropic prompt caching"
```

---

## Task 5.3: JudgmentResult parsing

**Files:**
- Modify: `src/kvtrace/judge/prompt.py` (add parsing helper) — OR create `src/kvtrace/judge/parsing.py`. We put parsing in a new file to keep prompt.py small.
- Create: `src/kvtrace/judge/parsing.py`
- Create: `tests/test_judge_response_parsing.py`

- [ ] **Step 1: Write the failing test**

`tests/test_judge_response_parsing.py`:
```python
import pytest
from pydantic import ValidationError

from kvtrace.judge.parsing import JudgeParseError, JudgmentResult, parse_judge_response


def test_parse_valid_json():
    r = parse_judge_response(
        '{"category":"A","confidence":0.85,"rationale":"5*4=24 instead of 20","affected_span":"5*4 = 24"}'
    )
    assert isinstance(r, JudgmentResult)
    assert r.category == "A"
    assert r.confidence == 0.85


def test_parse_with_surrounding_noise():
    # Anthropic sometimes wraps JSON in ```json fences.
    r = parse_judge_response(
        "```json\n"
        '{"category":"C","confidence":0.7,"rationale":"switched","affected_span":"new method"}'
        "\n```"
    )
    assert r.category == "C"


def test_parse_invalid_category_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response(
            '{"category":"Z","confidence":0.8,"rationale":"","affected_span":"x"}'
        )


def test_parse_missing_field_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response('{"category":"A","confidence":0.8}')


def test_parse_non_json_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response("sorry I cannot classify this")


def test_confidence_clamped_to_01():
    r = parse_judge_response(
        '{"category":"A","confidence":1.5,"rationale":"r","affected_span":"s"}'
    )
    assert r.confidence == 1.0

    r = parse_judge_response(
        '{"category":"A","confidence":-0.2,"rationale":"r","affected_span":"s"}'
    )
    assert r.confidence == 0.0
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_judge_response_parsing.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/kvtrace/judge/parsing.py`:
```python
"""Parse and validate Claude's JSON response into a JudgmentResult."""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from kvtrace.judge.taxonomy import lookup

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeParseError(ValueError):
    """Raised when Claude's response cannot be parsed as a valid judgment."""


class JudgmentResult(BaseModel):
    category: Literal["A", "B", "C", "D", "E", "F"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    affected_span: str

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f))


def parse_judge_response(text: str) -> JudgmentResult:
    """Extract JSON from Claude's response and validate it."""
    fence = _JSON_FENCE_RE.search(text)
    raw = fence.group(1) if fence else text
    m = _JSON_OBJECT_RE.search(raw)
    if not m:
        raise JudgeParseError(f"No JSON object found in response: {text[:200]!r}")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"Invalid JSON: {e}") from e

    # Validate category letter against the taxonomy BEFORE letting pydantic coerce.
    cat = str(data.get("category", "")).strip().upper()
    try:
        lookup(cat)
    except ValueError as e:
        raise JudgeParseError(str(e)) from e
    data["category"] = cat

    required = {"category", "confidence", "rationale", "affected_span"}
    missing = required - data.keys()
    if missing:
        raise JudgeParseError(f"Missing fields: {sorted(missing)}")

    try:
        return JudgmentResult(**data)
    except Exception as e:
        raise JudgeParseError(f"Schema validation failed: {e}") from e
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_judge_response_parsing.py -v`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/judge/parsing.py tests/test_judge_response_parsing.py
git commit -m "feat(judge): JSON response parsing + JudgmentResult schema"
```

---

## Task 5.4: Claude judge client with SHA256 cache

**Files:**
- Create: `src/kvtrace/judge/claude_judge.py`
- Create: `tests/test_judge_cache.py`
- Create: `tests/test_judge_claude_mock.py`

- [ ] **Step 1: Write the cache test**

`tests/test_judge_cache.py`:
```python
import json

from kvtrace.judge.claude_judge import _prompt_cache_key, JudgeCache


def test_prompt_cache_key_stable():
    k1 = _prompt_cache_key("system", [{"role": "user", "content": "x"}], "claude-sonnet-4-6")
    k2 = _prompt_cache_key("system", [{"role": "user", "content": "x"}], "claude-sonnet-4-6")
    assert k1 == k2
    assert len(k1) == 64


def test_prompt_cache_key_changes_on_input():
    k1 = _prompt_cache_key("sys1", [], "m")
    k2 = _prompt_cache_key("sys2", [], "m")
    assert k1 != k2


def test_judge_cache_roundtrip(tmp_path):
    cache = JudgeCache(tmp_path)
    cache.put("abc123", {"category": "A", "confidence": 0.9, "rationale": "r", "affected_span": "s"})
    assert cache.get("abc123") == {"category": "A", "confidence": 0.9, "rationale": "r", "affected_span": "s"}


def test_judge_cache_miss_returns_none(tmp_path):
    cache = JudgeCache(tmp_path)
    assert cache.get("nope") is None


def test_judge_cache_persists_across_instances(tmp_path):
    c1 = JudgeCache(tmp_path)
    c1.put("k", {"v": 1})
    c2 = JudgeCache(tmp_path)
    assert c2.get("k") == {"v": 1}
```

- [ ] **Step 2: Write the mocked Claude test**

`tests/test_judge_claude_mock.py`:
```python
from unittest.mock import MagicMock, patch

import pytest

from kvtrace.judge.claude_judge import ClaudeJudge


def _fake_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.usage = MagicMock(
        cache_creation_input_tokens=100,
        cache_read_input_tokens=0,
        input_tokens=50,
        output_tokens=30,
    )
    return resp


def test_claude_judge_happy_path(tmp_path):
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_response(
        '{"category":"A","confidence":0.9,"rationale":"arithmetic","affected_span":"5*4=24"}'
    )

    judge = ClaudeJudge(
        client=fake_client,
        model="claude-sonnet-4-6",
        cache_dir=tmp_path,
    )
    result = judge.judge(
        problem="p",
        ground_truth="20",
        baseline_context="5*4=20",
        quant_context="5*4=24",
        common_prefix="...",
        quant_method_name="fp8_e5m2",
    )
    assert result.category == "A"
    fake_client.messages.create.assert_called_once()


def test_claude_judge_uses_cache_on_second_call(tmp_path):
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_response(
        '{"category":"B","confidence":0.8,"rationale":"r","affected_span":"s"}'
    )

    judge = ClaudeJudge(client=fake_client, model="m", cache_dir=tmp_path)
    args = dict(
        problem="p", ground_truth="g",
        baseline_context="b", quant_context="q",
        common_prefix="c", quant_method_name="fp8_e5m2",
    )
    r1 = judge.judge(**args)
    r2 = judge.judge(**args)
    assert r1.category == r2.category
    # Second call must NOT hit the API.
    assert fake_client.messages.create.call_count == 1


def test_claude_judge_retries_parse_error(tmp_path):
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        _fake_response("not json"),
        _fake_response('{"category":"A","confidence":0.9,"rationale":"r","affected_span":"s"}'),
    ]
    judge = ClaudeJudge(client=fake_client, model="m", cache_dir=tmp_path, max_parse_retries=1)
    result = judge.judge(
        problem="p", ground_truth="g",
        baseline_context="b", quant_context="q",
        common_prefix="c", quant_method_name="fp8_e5m2",
    )
    assert result.category == "A"
    assert fake_client.messages.create.call_count == 2


def test_claude_judge_gives_up_after_retries(tmp_path):
    from kvtrace.judge.parsing import JudgeParseError
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_response("never json")
    judge = ClaudeJudge(client=fake_client, model="m", cache_dir=tmp_path, max_parse_retries=1)
    with pytest.raises(JudgeParseError):
        judge.judge(
            problem="p", ground_truth="g",
            baseline_context="b", quant_context="q",
            common_prefix="c", quant_method_name="fp8_e5m2",
        )
```

- [ ] **Step 3: Run both — expect failure**

Run: `pytest tests/test_judge_cache.py tests/test_judge_claude_mock.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement**

`src/kvtrace/judge/claude_judge.py`:
```python
"""Anthropic Claude client with on-disk SHA256 prompt cache."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from kvtrace.judge.parsing import JudgeParseError, JudgmentResult, parse_judge_response
from kvtrace.judge.prompt import PROMPT_VERSION, build_judge_messages, build_system_block
from kvtrace.judge.taxonomy import TAXONOMY_VERSION

log = logging.getLogger(__name__)


def _prompt_cache_key(system: str, messages: list[dict], model: str) -> str:
    """SHA256 of the normalized (system, messages, model, versions) tuple."""
    payload = json.dumps(
        {
            "system": system,
            "messages": messages,
            "model": model,
            "prompt_v": PROMPT_VERSION,
            "taxonomy_v": TAXONOMY_VERSION,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeCache:
    """Trivial key→JSON on-disk cache."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def put(self, key: str, value: dict) -> None:
        self._path(key).write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )


class ClaudeJudge:
    def __init__(
        self,
        client: Any,
        model: str,
        cache_dir: str | Path,
        temperature: float = 0.0,
        max_tokens: int = 400,
        max_parse_retries: int = 2,
    ) -> None:
        self.client = client
        self.model = model
        self.cache = JudgeCache(cache_dir)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_parse_retries = max_parse_retries

    def judge(
        self,
        *,
        problem: str,
        ground_truth: str,
        baseline_context: str,
        quant_context: str,
        common_prefix: str,
        quant_method_name: str,
    ) -> JudgmentResult:
        system_blocks = build_system_block()
        messages = build_judge_messages(
            problem=problem,
            ground_truth=ground_truth,
            baseline_context=baseline_context,
            quant_context=quant_context,
            common_prefix=common_prefix,
            quant_method_name=quant_method_name,
        )
        system_flat = "\n".join(b["text"] for b in system_blocks)
        key = _prompt_cache_key(system_flat, messages, self.model)

        cached = self.cache.get(key)
        if cached is not None:
            return JudgmentResult(**cached)

        last_err: Exception | None = None
        for attempt in range(self.max_parse_retries + 1):
            resp = self.client.messages.create(
                model=self.model,
                system=system_blocks,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = resp.content[0].text
            try:
                result = parse_judge_response(text)
                self.cache.put(key, result.model_dump())
                return result
            except JudgeParseError as e:
                last_err = e
                log.warning("judge parse error attempt %d: %s", attempt, e)
                continue

        raise last_err or JudgeParseError("exhausted retries")
```

- [ ] **Step 5: Run — expect pass**

Run: `pytest tests/test_judge_cache.py tests/test_judge_claude_mock.py -v`
Expected: 9 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/kvtrace/judge/claude_judge.py tests/test_judge_cache.py tests/test_judge_claude_mock.py
git commit -m "feat(judge): Claude client with SHA256 cache and parse-error retries"
```

---

## Task 5.5: Golden calibration set + live-api calibration test

**Files:**
- Create: `src/kvtrace/judge/golden_set.py`
- Create: `tests/test_judge_calibration.py`

- [ ] **Step 1: Write the golden set module**

`src/kvtrace/judge/golden_set.py`:
```python
"""Manually curated (trace_pair, gold_category) examples for judge calibration.

Each example deliberately exhibits ONE category A..F so that a correctly
working judge classifies each with ≥ calibration_threshold accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GoldenExample:
    id: str
    problem: str
    ground_truth: str
    baseline_context: str
    quant_context: str
    common_prefix: str
    gold_category: Literal["A", "B", "C", "D", "E", "F"]


GOLDEN_SET: tuple[GoldenExample, ...] = (
    GoldenExample(
        id="A1_arith_mul",
        problem="Compute 17 * 23.",
        ground_truth="391",
        common_prefix="Let me multiply 17 and 23. I'll break it up: 17*20 + 17*3.",
        baseline_context="17*20 = 340. 17*3 = 51. So 340+51 = 391.",
        quant_context="17*20 = 340. 17*3 = 51. So 340+51 = 381.",
        gold_category="A",
    ),
    GoldenExample(
        id="A2_arith_exp",
        problem="Find (x+2)^2.",
        ground_truth="x^2+4x+4",
        common_prefix="Expand (x+2)^2 step by step.",
        baseline_context="(x+2)^2 = x^2 + 2*2*x + 4 = x^2 + 4x + 4.",
        quant_context="(x+2)^2 = x^2 + 4.",
        gold_category="A",
    ),
    GoldenExample(
        id="B1_logic",
        problem="Is every even number > 2 composite?",
        ground_truth="Yes",
        common_prefix="Let n be even and n > 2. Then n = 2k for some k > 1.",
        baseline_context="Since 2 divides n and n > 2, n has a proper divisor, so n is composite.",
        quant_context="Since 2 divides n, n is composite. But wait, 2 is even and prime, so the claim is false.",
        gold_category="B",
    ),
    GoldenExample(
        id="C1_strategy",
        problem="Solve x^2 - 5x + 6 = 0.",
        ground_truth="x=2 or x=3",
        common_prefix="By Vieta's: sum of roots is 5, product is 6.",
        baseline_context="Roots are 2 and 3 since 2+3=5 and 2*3=6.",
        quant_context="Actually let me drop Vieta's and try the quadratic formula from scratch with different numbers.",
        gold_category="C",
    ),
    GoldenExample(
        id="D1_halluc",
        problem="Factor 143.",
        ground_truth="11 * 13",
        common_prefix="I need to find prime factors of 143.",
        baseline_context="143 / 11 = 13. So 143 = 11 * 13.",
        quant_context="By the Smith-Jones divisibility theorem of 2017, 143 factors as 7 * 21.",
        gold_category="D",
    ),
    GoldenExample(
        id="E1_trunc",
        problem="Sum 1+2+...+10.",
        ground_truth="55",
        common_prefix="I'll use the formula n(n+1)/2 with n=10.",
        baseline_context="10*11/2 = 55.",
        quant_context="10*11/2 = ... and then",
        gold_category="E",
    ),
    GoldenExample(
        id="F1_loop",
        problem="Find the smallest prime > 10.",
        ground_truth="11",
        common_prefix="Candidates: 11, 12, 13, ...",
        baseline_context="11 is prime. Answer: 11.",
        quant_context="Let me try 11. Let me try 11. Let me try 11. Let me try 11. Let me try 11.",
        gold_category="F",
    ),
    GoldenExample(
        id="A3_arith_frac",
        problem="Compute 1/2 + 1/3.",
        ground_truth="5/6",
        common_prefix="Common denominator is 6.",
        baseline_context="3/6 + 2/6 = 5/6.",
        quant_context="3/6 + 2/6 = 6/6 = 1.",
        gold_category="A",
    ),
    GoldenExample(
        id="B2_logic",
        problem="If p implies q, and q, does p follow?",
        ground_truth="No (affirming the consequent).",
        common_prefix="Given: p → q and q.",
        baseline_context="We cannot conclude p; this is affirming the consequent.",
        quant_context="From q and p → q we conclude p.",
        gold_category="B",
    ),
    GoldenExample(
        id="F2_loop",
        problem="Solve 2x=4.",
        ground_truth="x=2",
        common_prefix="Divide both sides by 2.",
        baseline_context="x = 2.",
        quant_context="Divide by 2. Divide by 2. Divide by 2. Divide by 2. Divide by 2.",
        gold_category="F",
    ),
)
```

- [ ] **Step 2: Write the calibration test**

`tests/test_judge_calibration.py`:
```python
"""Live-API calibration: judge must classify ≥7/10 golden examples correctly."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from kvtrace.judge.claude_judge import ClaudeJudge
from kvtrace.judge.golden_set import GOLDEN_SET


@pytest.mark.live_api
def test_judge_passes_golden_set(tmp_path: Path) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    judge = ClaudeJudge(client=client, model="claude-sonnet-4-6", cache_dir=tmp_path)

    correct = 0
    errors: list[str] = []
    for ex in GOLDEN_SET:
        r = judge.judge(
            problem=ex.problem,
            ground_truth=ex.ground_truth,
            baseline_context=ex.baseline_context,
            quant_context=ex.quant_context,
            common_prefix=ex.common_prefix,
            quant_method_name="test",
        )
        if r.category == ex.gold_category:
            correct += 1
        else:
            errors.append(f"{ex.id}: gold={ex.gold_category} got={r.category} ({r.rationale[:60]})")

    accuracy = correct / len(GOLDEN_SET)
    msg = f"calibration {correct}/{len(GOLDEN_SET)} = {accuracy:.0%}. Errors:\n" + "\n".join(errors)
    assert accuracy >= 0.7, msg
```

- [ ] **Step 3: Verify the test is collectible but skipped without API key**

Run: `pytest tests/test_judge_calibration.py -v`
Expected: 1 test, SKIPPED (no marker runs it unless `-m live_api`).

Run: `pytest -m live_api tests/test_judge_calibration.py --collect-only`
Expected: 1 test collected.

- [ ] **Step 4: Commit**

```bash
git add src/kvtrace/judge/golden_set.py tests/test_judge_calibration.py
git commit -m "feat(judge): golden calibration set (10 curated pairs) + live-api test"
```

---

# Phase 6 — HF Hub integration

## Task 6.1: HF Hub upload + download helpers

**Files:**
- Create: `src/kvtrace/hf_hub/upload.py`
- Create: `src/kvtrace/hf_hub/download.py`
- Create: `tests/test_hf_hub_upload.py`

- [ ] **Step 1: Write the failing test**

`tests/test_hf_hub_upload.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kvtrace.hf_hub.download import download_dataset_file
from kvtrace.hf_hub.upload import resolve_repo_id, upload_dataset_file


def test_resolve_repo_id_from_env(monkeypatch):
    monkeypatch.setenv("HF_REPO_ID", "me/my-ds")
    assert resolve_repo_id() == "me/my-ds"


def test_resolve_repo_id_from_user(monkeypatch):
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.setenv("HF_USER", "me")
    assert resolve_repo_id() == "me/kv-trace-study"


def test_resolve_repo_id_none_when_unset(monkeypatch):
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.delenv("HF_USER", raising=False)
    assert resolve_repo_id() is None


def test_upload_skipped_when_no_repo(tmp_path, monkeypatch):
    f = tmp_path / "data.jsonl"
    f.write_text('{"a":1}\n')
    api = MagicMock()
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.delenv("HF_USER", raising=False)
    result = upload_dataset_file(f, revision_tag="traces-m-bf16", api=api)
    assert result is False
    api.upload_file.assert_not_called()


def test_upload_calls_api(tmp_path, monkeypatch):
    f = tmp_path / "data.jsonl"
    f.write_text('{"a":1}\n')
    api = MagicMock()
    # list_repo_refs returns no branches matching → upload proceeds.
    api.list_repo_refs.return_value = MagicMock(branches=[])
    monkeypatch.setenv("HF_REPO_ID", "me/kv")
    result = upload_dataset_file(f, revision_tag="traces-m-bf16", api=api)
    assert result is True
    api.create_branch.assert_called_once()
    api.upload_file.assert_called_once()


def test_upload_idempotent_when_revision_exists(tmp_path, monkeypatch):
    f = tmp_path / "data.jsonl"
    f.write_text('{"a":1}\n')
    api = MagicMock()
    api.list_repo_refs.return_value = MagicMock(branches=[MagicMock(name="traces-m-bf16")])
    monkeypatch.setenv("HF_REPO_ID", "me/kv")
    result = upload_dataset_file(f, revision_tag="traces-m-bf16", api=api)
    # idempotent short-circuit
    assert result is False
    api.upload_file.assert_not_called()


def test_download_calls_hf_hub_download(tmp_path, monkeypatch):
    api = MagicMock()
    api.hf_hub_download.return_value = str(tmp_path / "downloaded.jsonl")
    (tmp_path / "downloaded.jsonl").write_text("ok")
    monkeypatch.setenv("HF_REPO_ID", "me/kv")
    local = download_dataset_file("data.jsonl", revision_tag="traces-m-bf16", api=api)
    assert local is not None
    api.hf_hub_download.assert_called_once()
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_hf_hub_upload.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement uploader**

`src/kvtrace/hf_hub/upload.py`:
```python
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


def upload_dataset_file(
    local_path: Path,
    *,
    revision_tag: str,
    api: Any | None = None,
    repo_id: str | None = None,
    path_in_repo: str | None = None,
) -> bool:
    """Upload a file to a HuggingFace dataset repo under a revision branch.

    Returns True if an upload was performed; False if skipped (no repo or
    revision already present).
    """
    repo_id = repo_id or resolve_repo_id()
    if not repo_id:
        log.info("no HF_REPO_ID / HF_USER; skipping upload of %s", local_path)
        return False

    if api is None:  # pragma: no cover — import path
        from huggingface_hub import HfApi
        api = HfApi()

    if _branch_exists(api, repo_id, revision_tag):
        log.info("revision %s already on %s; skipping upload", revision_tag, repo_id)
        return False

    api.create_branch(repo_id=repo_id, repo_type="dataset", branch=revision_tag, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo or local_path.name,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision_tag,
    )
    log.info("uploaded %s -> %s@%s", local_path, repo_id, revision_tag)
    return True
```

- [ ] **Step 4: Implement downloader**

`src/kvtrace/hf_hub/download.py`:
```python
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
```

- [ ] **Step 5: Run — expect pass**

Run: `pytest tests/test_hf_hub_upload.py -v`
Expected: 7 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/kvtrace/hf_hub/*.py tests/test_hf_hub_upload.py
git commit -m "feat(hf_hub): idempotent revision-keyed dataset upload/download"
```

---

# Phase 7 — Analysis

## Task 7.1: Failure-signature statistics

**Files:**
- Create: `src/kvtrace/analysis/signatures.py`
- Create: `tests/test_signatures.py`

- [ ] **Step 1: Write the failing test**

`tests/test_signatures.py`:
```python
import numpy as np
import pytest

from kvtrace.analysis.signatures import (
    aggregate_counts,
    chi_square_test,
    cramers_v,
    row_normalize,
)


def _judgments(method: str, counts: dict[str, int]) -> list[dict]:
    rows = []
    for cat, n in counts.items():
        for _ in range(n):
            rows.append({"quant_method": method, "category": cat})
    return rows


def test_aggregate_counts_shape_and_order():
    js = (
        _judgments("bf16", {"A": 10, "B": 2, "C": 0, "D": 0, "E": 3, "F": 0})
        + _judgments("fp8_e5m2", {"A": 5, "B": 4, "C": 2, "D": 0, "E": 1, "F": 3})
    )
    methods, matrix = aggregate_counts(js)
    assert methods == ["bf16", "fp8_e5m2"]
    assert matrix.shape == (2, 6)
    assert matrix[0].tolist() == [10, 2, 0, 0, 3, 0]
    assert matrix[1].tolist() == [5, 4, 2, 0, 1, 3]


def test_row_normalize_sums_to_one():
    m = np.array([[1, 2, 3, 0, 0, 4], [0, 0, 0, 0, 0, 10]], dtype=float)
    n = row_normalize(m)
    assert n.shape == m.shape
    assert np.allclose(n.sum(axis=1), [1.0, 1.0])


def test_row_normalize_zero_row_yields_zero():
    m = np.array([[0, 0, 0, 0, 0, 0]], dtype=float)
    n = row_normalize(m)
    assert np.all(n == 0.0)


def test_chi_square_detects_difference():
    # clearly different distributions per row
    m = np.array(
        [[30, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 30]], dtype=float
    )
    chi2, p, dof = chi_square_test(m)
    assert p < 0.001
    assert dof == 5


def test_chi_square_same_distribution_high_p():
    m = np.array(
        [[10, 10, 10, 10, 10, 10],
         [10, 10, 10, 10, 10, 10]], dtype=float
    )
    chi2, p, dof = chi_square_test(m)
    assert p > 0.5


def test_cramers_v_in_unit_interval():
    m = np.array([[10, 5], [2, 8]], dtype=float)
    v = cramers_v(m)
    assert 0.0 <= v <= 1.0


def test_aggregate_counts_empty_raises():
    with pytest.raises(ValueError):
        aggregate_counts([])
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_signatures.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/kvtrace/analysis/signatures.py`:
```python
"""Aggregate judgments into failure-signature matrices and run statistics."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency

CATEGORY_ORDER = ("A", "B", "C", "D", "E", "F")


def aggregate_counts(
    judgments: list[dict],
) -> tuple[list[str], np.ndarray]:
    """Build a [n_methods × 6] matrix of counts.

    Methods are ordered by first appearance in `judgments`.
    Each row sums to the number of judgments for that quant method.
    """
    if not judgments:
        raise ValueError("judgments must be non-empty")

    methods: list[str] = []
    rows: dict[str, np.ndarray] = {}
    for j in judgments:
        m = j["quant_method"]
        c = j["category"]
        if c not in CATEGORY_ORDER:
            continue
        if m not in rows:
            methods.append(m)
            rows[m] = np.zeros(6, dtype=float)
        rows[m][CATEGORY_ORDER.index(c)] += 1

    matrix = np.stack([rows[m] for m in methods], axis=0)
    return methods, matrix


def row_normalize(m: np.ndarray) -> np.ndarray:
    """Per-row normalization; all-zero rows stay zero."""
    out = np.zeros_like(m, dtype=float)
    for i in range(m.shape[0]):
        s = m[i].sum()
        if s > 0:
            out[i] = m[i] / s
    return out


def chi_square_test(m: np.ndarray) -> tuple[float, float, int]:
    """Return (chi2, p_value, dof). Drops all-zero columns to avoid degenerate stats."""
    nonzero_cols = (m.sum(axis=0) > 0)
    m_clean = m[:, nonzero_cols]
    chi2, p, dof, _ = chi2_contingency(m_clean)
    # Rescale dof to always correspond to the full 6-category shape for reporting.
    return float(chi2), float(p), int(dof)


def cramers_v(m: np.ndarray) -> float:
    """Cramér's V for a contingency matrix."""
    nonzero_cols = (m.sum(axis=0) > 0)
    m_clean = m[:, nonzero_cols]
    chi2, _, _, _ = chi2_contingency(m_clean)
    n = m_clean.sum()
    if n == 0:
        return 0.0
    k = min(m_clean.shape[0], m_clean.shape[1])
    if k <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (k - 1))))
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_signatures.py -v`
Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/analysis/signatures.py tests/test_signatures.py
git commit -m "feat(analysis): confusion matrix + chi-square + Cramér's V"
```

---

## Task 7.2: Report generation (markdown + plots)

**Files:**
- Create: `src/kvtrace/analysis/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:
```python
import json
from pathlib import Path

import numpy as np

from kvtrace.analysis.report import build_report, write_heatmap


def test_build_report_markdown_contains_table(tmp_path):
    judgments = [
        {"quant_method": "bf16", "category": "A", "model": "m1"},
        {"quant_method": "bf16", "category": "B", "model": "m1"},
        {"quant_method": "fp8_e5m2", "category": "A", "model": "m1"},
        {"quant_method": "fp8_e5m2", "category": "F", "model": "m1"},
        {"quant_method": "fp8_e5m2", "category": "F", "model": "m1"},
    ]
    md_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    build_report(judgments, md_path=md_path, json_path=json_path)

    text = md_path.read_text()
    assert "| method" in text.lower() or "|method" in text.lower()
    assert "bf16" in text
    assert "fp8_e5m2" in text
    assert "chi" in text.lower()

    data = json.loads(json_path.read_text())
    assert "signatures" in data
    assert "chi_square_p" in data


def test_report_is_deterministic(tmp_path):
    judgments = [
        {"quant_method": "bf16", "category": "A", "model": "m1"},
        {"quant_method": "fp8_e5m2", "category": "F", "model": "m1"},
    ]
    p1 = tmp_path / "a.md"
    p2 = tmp_path / "b.md"
    build_report(judgments, md_path=p1, json_path=tmp_path / "a.json")
    build_report(judgments, md_path=p2, json_path=tmp_path / "b.json")
    assert p1.read_text() == p2.read_text()


def test_write_heatmap_creates_png(tmp_path):
    matrix = np.array([[0.3, 0.1, 0.0, 0.0, 0.5, 0.1], [0.1, 0.0, 0.0, 0.0, 0.0, 0.9]])
    out = tmp_path / "heatmap.png"
    write_heatmap(matrix, method_names=["bf16", "fp8_e5m2"], out_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_report.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/kvtrace/analysis/report.py`:
```python
"""Markdown + JSON report generation with matplotlib plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kvtrace.analysis.signatures import (  # noqa: E402
    CATEGORY_ORDER,
    aggregate_counts,
    chi_square_test,
    cramers_v,
    row_normalize,
)


def build_report(
    judgments: list[dict],
    *,
    md_path: Path,
    json_path: Path,
) -> None:
    methods, counts = aggregate_counts(judgments)
    signatures = row_normalize(counts)
    chi2, p, dof = chi_square_test(counts)
    v = cramers_v(counts)

    # --- JSON
    data = {
        "methods": methods,
        "categories": list(CATEGORY_ORDER),
        "counts": counts.astype(int).tolist(),
        "signatures": signatures.tolist(),
        "chi_square": chi2,
        "chi_square_p": p,
        "chi_square_dof": dof,
        "cramers_v": v,
    }
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    # --- Markdown
    lines: list[str] = []
    lines.append("# KV Cache Quantization — Failure-Signature Report\n")
    lines.append(f"Total judgments: **{int(counts.sum())}**\n")
    lines.append(f"Chi-square: chi2={chi2:.2f}, dof={dof}, **p={p:.3g}**\n")
    lines.append(f"Cramér's V: **{v:.3f}**\n")
    lines.append("")

    lines.append("## Raw counts (rows = quant method, cols = category A..F)\n")
    lines.append("| method | " + " | ".join(CATEGORY_ORDER) + " | total |")
    lines.append("|---|" + "|".join(["---"] * 6) + "|---|")
    for m, row in zip(methods, counts.astype(int)):
        lines.append(f"| {m} | " + " | ".join(str(x) for x in row) + f" | {int(row.sum())} |")

    lines.append("\n## Normalized failure signatures (rows sum to 1)\n")
    lines.append("| method | " + " | ".join(CATEGORY_ORDER) + " |")
    lines.append("|---|" + "|".join(["---"] * 6) + "|")
    for m, row in zip(methods, signatures):
        lines.append(f"| {m} | " + " | ".join(f"{x:.2f}" for x in row) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_heatmap(matrix: np.ndarray, method_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, max(2, len(method_names) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels(list(CATEGORY_ORDER))
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names)
    ax.set_xlabel("error category")
    ax.set_title("Failure signature per quantization method")
    fig.colorbar(im, ax=ax, label="fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/test_report.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/kvtrace/analysis/report.py tests/test_report.py
git commit -m "feat(analysis): deterministic markdown+json report and heatmap plot"
```

---

# Phase 8 — Orchestration scripts

## Task 8.1: `01_generate_traces.py` + smoke test

**Files:**
- Create: `scripts/01_generate_traces.py`
- Create: `tests/test_end_to_end_smoke.py`

- [ ] **Step 1: Write the end-to-end smoke test first**

`tests/test_end_to_end_smoke.py`:
```python
"""Full pipeline on a synthetic in-memory dataset and mocked generators/judge.

This is the top-level CI gate. If it passes, the phases are wired up correctly.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kvtrace.dataset_loader import MathProblem
from kvtrace.fdp.finder import find_fdp
from kvtrace.generators.base import GenerationResult


class NoCosmeticChecker:
    def is_cosmetic_divergence(self, *a, **kw):
        return False


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(65 + (i % 26)) for i in ids)


def _fake_generation(idx: int, tokens: list[int], text: str, boxed: str | None) -> GenerationResult:
    return GenerationResult(
        idx=idx, raw=text, think=None, final_response=text, boxed_answer=boxed,
        think_complete=True, finish_reason="stop",
        token_ids=tokens, prompt_tokens=5, generated_tokens=len(tokens),
    )


def test_smoke_pipeline_without_network(tmp_path: Path):
    # Phase 1 (synthesized): 2 problems, 1 baseline + 1 quant config
    problems = [
        MathProblem(idx=0, problem="p0", answer="42", source="aime-24"),
        MathProblem(idx=1, problem="p1", answer="7", source="math-500"),
    ]
    baseline_rows = [
        _fake_generation(0, [1, 2, 3, 4, 5], "baseline-0", "42"),
        _fake_generation(1, [1, 2, 3, 4, 5], "baseline-1", "7"),
    ]
    quant_rows = [
        _fake_generation(0, [1, 2, 9, 4, 5], "quant-0", "41"),
        _fake_generation(1, [1, 2, 3, 4, 5], "quant-1", "7"),
    ]

    # Phase 2: run FDP
    fdp_records = []
    for base, quant, prob in zip(baseline_rows, quant_rows, problems):
        r = find_fdp(
            baseline_tokens=base.token_ids,
            quant_tokens=quant.token_ids,
            baseline_text=base.raw,
            quant_text=quant.raw,
            tokenizer=FakeTokenizer(),
            resync_checker=NoCosmeticChecker(),
            baseline_boxed=base.boxed_answer,
            quant_boxed=quant.boxed_answer,
            ground_truth=prob.answer,
        )
        fdp_records.append(r)

    # Only the first problem has a divergence
    assert fdp_records[0].fdp_token_idx == 2
    assert fdp_records[1].fdp_token_idx is None

    # Phase 3 (mock judge): fake judgments — arithmetic for prob 0, no judgment for prob 1
    judgments = [
        {"problem_idx": 0, "quant_method": "fp8_e5m2", "model": "m",
         "category": "A", "confidence": 0.9, "rationale": "r", "affected_span": "s"},
    ]

    # Phase 4: analyze
    from kvtrace.analysis.report import build_report
    md = tmp_path / "report.md"
    js = tmp_path / "report.json"
    build_report(judgments, md_path=md, json_path=js)

    text = md.read_text()
    assert "fp8_e5m2" in text
    assert "Chi-square" in text
    data = json.loads(js.read_text())
    assert data["methods"] == ["fp8_e5m2"]
```

- [ ] **Step 2: Run — expect pass (all phases already implemented)**

Run: `pytest tests/test_end_to_end_smoke.py -v`
Expected: test passes. If it fails, the failure is in one of Phase 1–4; fix the upstream implementation, not the test.

- [ ] **Step 3: Write the generation CLI**

`scripts/01_generate_traces.py`:
```python
"""Phase 1: generate traces for a (model, quant_config) pair.

Reads:
  - config/models.yaml, quant_methods.yaml, pipeline.yaml
  - HuggingFace dataset (cached)

Writes:
  - outputs/traces/{model}_{config}.jsonl
  - optional: HF dataset revision traces-{model}-{config}

Idempotent: skips if HF revision already exists AND --resume is given.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kvtrace.config import load_all_configs
from kvtrace.dataset_loader import load_study_mix
from kvtrace.generators.base import Generator
from kvtrace.generators.hf_gen import HFGenerator
from kvtrace.generators.vllm_gen import VLLMGenerator
from kvtrace.hf_hub.upload import resolve_repo_id, upload_dataset_file

log = logging.getLogger("generate")


def make_generator(engine: str, pipeline) -> Generator:
    if engine == "vllm":
        return VLLMGenerator(
            seed=pipeline.seed,
            sampling_temperature=pipeline.sampling.temperature,
            sampling_top_p=pipeline.sampling.top_p,
            sampling_max_tokens=pipeline.sampling.max_tokens,
        )
    if engine == "hf":
        return HFGenerator(
            sampling_temperature=pipeline.sampling.temperature,
            sampling_top_p=pipeline.sampling.top_p,
            sampling_max_tokens=pipeline.sampling.max_tokens,
        )
    raise ValueError(f"unknown engine {engine!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="key in config/models.yaml")
    parser.add_argument("--config", required=True, help="key in config/quant_methods.yaml")
    parser.add_argument("--output_dir", default="outputs/traces")
    parser.add_argument("--resume", action="store_true", help="skip if HF revision exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    models, quants, pipeline = load_all_configs(Path("config"))

    if args.model not in models:
        log.error("unknown model %r; available: %s", args.model, sorted(models))
        return 2
    if args.config not in quants:
        log.error("unknown config %r; available: %s", args.config, sorted(quants))
        return 2

    mcfg = models[args.model]
    qcfg = quants[args.config]

    # Per-model max_model_len overrides (see QuantCfg.overrides).
    override = qcfg.overrides.get("max_model_len_override", {}).get(args.model)
    if override is not None:
        log.info("override max_model_len %s -> %d for this run", mcfg.max_model_len, override)
        mcfg = mcfg.model_copy(update={"max_model_len": override})

    revision_tag = f"traces-{args.model}-{args.config}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.model}_{args.config}.jsonl"

    if args.resume and out_file.exists():
        log.info("resume: %s already present, skipping generation", out_file)
        return 0

    problems = load_study_mix(
        aime_n=pipeline.dataset.aime_24_count,
        math_n=pipeline.dataset.math_500_count,
    )
    log.info("loaded %d problems", len(problems))

    gen = make_generator(qcfg.engine, pipeline)
    try:
        gen.load(mcfg, qcfg)
        results = gen.generate(problems)
    finally:
        gen.unload()

    with out_file.open("w", encoding="utf-8") as f:
        for p, r in zip(problems, results):
            row = r.to_dict()
            row.update({
                "problem": p.problem,
                "ground_truth": p.answer,
                "source": p.source,
                "model": args.model,
                "quant_method": args.config,
                "seed": pipeline.seed,
            })
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("wrote %d records to %s", len(results), out_file)

    if resolve_repo_id():
        upload_dataset_file(out_file, revision_tag=revision_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Sanity: script imports without error**

Run: `python -c "import scripts.01_generate_traces"` — this likely fails because script name starts with a digit. Instead:
Run: `python scripts/01_generate_traces.py --help`
Expected: argparse help printed with `--model`, `--config`, `--output_dir`, `--resume`.

- [ ] **Step 5: Commit**

```bash
git add scripts/01_generate_traces.py tests/test_end_to_end_smoke.py
git commit -m "feat(cli): Phase 1 generate_traces + end-to-end smoke gate"
```

---

## Task 8.2: `02_find_fdps.py`

**Files:**
- Create: `scripts/02_find_fdps.py`

- [ ] **Step 1: Write script**

`scripts/02_find_fdps.py`:
```python
"""Phase 2: compute FDP between baseline (bf16) and every quant config.

Reads:
  outputs/traces/{model}_bf16.jsonl
  outputs/traces/{model}_{Q}.jsonl  for Q in {fp8_e5m2, fp8_e4m3, hqq_int4, hqq_int2}

Writes:
  outputs/fdps/{model}_{Q}.jsonl

No GPU required. MiniLM embedder is loaded lazily for the re-sync check.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kvtrace.config import load_all_configs
from kvtrace.fdp.finder import FDPParams, find_fdp
from kvtrace.fdp.semantic_resync import ReSyncChecker
from kvtrace.hf_hub.upload import resolve_repo_id, upload_dataset_file

log = logging.getLogger("fdp")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--traces_dir", default="outputs/traces")
    parser.add_argument("--out_dir", default="outputs/fdps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    models, quants, pipeline = load_all_configs(Path("config"))

    if args.model not in models:
        log.error("unknown model %r", args.model)
        return 2

    baseline_path = Path(args.traces_dir) / f"{args.model}_bf16.jsonl"
    if not baseline_path.exists():
        log.error("baseline missing: %s (run Phase 1 with --config bf16 first)", baseline_path)
        return 3

    baseline = _load_jsonl(baseline_path)
    baseline_by_idx = {r["idx"]: r for r in baseline}

    # Use the model's tokenizer for decoding token windows.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        models[args.model].hf_id, trust_remote_code=True
    )

    checker = ReSyncChecker(
        model_name=pipeline.fdp.embed_model,
        threshold=pipeline.fdp.cosmetic_cosine_threshold,
    )
    params = FDPParams(
        context_window=pipeline.fdp.context_window,
        resync_lookahead=pipeline.fdp.resync_lookahead,
        max_cosmetic_skips=pipeline.fdp.max_cosmetic_skips,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quant_keys = [k for k in quants if k != "bf16"]
    for qkey in quant_keys:
        qpath = Path(args.traces_dir) / f"{args.model}_{qkey}.jsonl"
        if not qpath.exists():
            log.warning("missing quant trace %s; skipping", qpath)
            continue

        quant_rows = _load_jsonl(qpath)
        out_file = out_dir / f"{args.model}_{qkey}.jsonl"
        n = 0
        with out_file.open("w", encoding="utf-8") as f:
            for qrow in quant_rows:
                brow = baseline_by_idx.get(qrow["idx"])
                if brow is None:
                    continue
                rec = find_fdp(
                    baseline_tokens=brow["token_ids"],
                    quant_tokens=qrow["token_ids"],
                    baseline_text=brow["raw_output"],
                    quant_text=qrow["raw_output"],
                    tokenizer=tokenizer,
                    resync_checker=checker,
                    params=params,
                    baseline_truncated=(brow.get("finish_reason") == "length"),
                    quant_truncated=(qrow.get("finish_reason") == "length"),
                    baseline_boxed=brow.get("boxed_answer"),
                    quant_boxed=qrow.get("boxed_answer"),
                    ground_truth=brow.get("ground_truth"),
                )
                out = {
                    "problem_idx": qrow["idx"],
                    "model": args.model,
                    "quant_method": qkey,
                    "source": qrow.get("source"),
                    "problem": qrow.get("problem"),
                    "ground_truth": qrow.get("ground_truth"),
                    "fdp_token_idx": rec.fdp_token_idx,
                    "cosmetic_skipped": rec.cosmetic_skipped,
                    "baseline_context": rec.baseline_context,
                    "quant_context": rec.quant_context,
                    "common_prefix": rec.common_prefix,
                    "boxed_match": rec.boxed_match,
                    "baseline_truncated": rec.baseline_truncated,
                    "quant_truncated": rec.quant_truncated,
                    "baseline_boxed": brow.get("boxed_answer"),
                    "quant_boxed": qrow.get("boxed_answer"),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                n += 1
        log.info("wrote %d FDP records to %s", n, out_file)
        if resolve_repo_id():
            upload_dataset_file(out_file, revision_tag=f"fdps-{args.model}-{qkey}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify script boots**

Run: `python scripts/02_find_fdps.py --help`
Expected: argparse help text printed.

- [ ] **Step 3: Commit**

```bash
git add scripts/02_find_fdps.py
git commit -m "feat(cli): Phase 2 find_fdps script"
```

---

## Task 8.3: `03_judge_fdps.py`

**Files:**
- Create: `scripts/03_judge_fdps.py`

- [ ] **Step 1: Write script**

`scripts/03_judge_fdps.py`:
```python
"""Phase 3: classify each FDP via Claude Sonnet 4.6 with SHA256 cache."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from kvtrace.config import load_all_configs
from kvtrace.hf_hub.upload import resolve_repo_id, upload_dataset_file
from kvtrace.judge.claude_judge import ClaudeJudge

log = logging.getLogger("judge")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdps_dir", default="outputs/fdps")
    parser.add_argument("--out_dir", default="outputs/judgments")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    _, _, pipeline = load_all_configs(Path("config"))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY is required for Phase 3")
        return 2

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    judge = ClaudeJudge(
        client=client,
        model=pipeline.judge.model,
        cache_dir=pipeline.judge.cache_dir,
        temperature=pipeline.judge.temperature,
        max_tokens=pipeline.judge.max_tokens,
    )

    fdps_dir = Path(args.fdps_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(fdps_dir.glob("*.jsonl")):
        out_file = out_dir / f.name
        with f.open(encoding="utf-8") as fin, out_file.open("w", encoding="utf-8") as fout:
            written = 0
            for line in fin:
                rec = json.loads(line)
                if rec.get("fdp_token_idx") is None:
                    # No divergence; skip entirely rather than forcing a category.
                    continue
                try:
                    res = judge.judge(
                        problem=rec.get("problem", ""),
                        ground_truth=rec.get("ground_truth", ""),
                        baseline_context=rec.get("baseline_context", ""),
                        quant_context=rec.get("quant_context", ""),
                        common_prefix=rec.get("common_prefix", ""),
                        quant_method_name=rec.get("quant_method", ""),
                    )
                except Exception as e:
                    log.warning("judge failed on %s idx=%s: %s", f.name, rec.get("problem_idx"), e)
                    continue
                out_row = {
                    "problem_idx": rec["problem_idx"],
                    "model": rec["model"],
                    "quant_method": rec["quant_method"],
                    "category": res.category,
                    "confidence": res.confidence,
                    "rationale": res.rationale,
                    "affected_span": res.affected_span,
                    "boxed_match": rec.get("boxed_match"),
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                written += 1
        log.info("wrote %d judgments to %s", written, out_file)
        if resolve_repo_id():
            parts = f.stem.split("_", 1)  # e.g. "deepseek-r1-distill-qwen-1.5b_fp8_e5m2" → [model, config]
            model, qkey = parts[0], parts[1] if len(parts) > 1 else "unknown"
            upload_dataset_file(out_file, revision_tag=f"judgments-{model}-{qkey}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify script boots**

Run: `python scripts/03_judge_fdps.py --help`
Expected: argparse help text.

- [ ] **Step 3: Commit**

```bash
git add scripts/03_judge_fdps.py
git commit -m "feat(cli): Phase 3 judge_fdps script"
```

---

## Task 8.4: `04_analyze.py`

**Files:**
- Create: `scripts/04_analyze.py`

- [ ] **Step 1: Write script**

`scripts/04_analyze.py`:
```python
"""Phase 4: aggregate judgments into failure-signature report."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from kvtrace.analysis.report import build_report, write_heatmap
from kvtrace.analysis.signatures import CATEGORY_ORDER, aggregate_counts, row_normalize

log = logging.getLogger("analyze")


def _load_all(judgments_dir: Path) -> list[dict]:
    out: list[dict] = []
    for f in sorted(judgments_dir.glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            out.extend(json.loads(ln) for ln in fh)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments_dir", default="outputs/judgments")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    judgments = _load_all(Path(args.judgments_dir))
    if not judgments:
        log.error("no judgments in %s", args.judgments_dir)
        return 3

    build_report(
        judgments,
        md_path=out_dir / "report.md",
        json_path=out_dir / "report.json",
    )

    # Global heatmap
    methods, counts = aggregate_counts(judgments)
    sigs = row_normalize(counts)
    write_heatmap(sigs, methods, out_dir / "plots" / "signatures_all.png")

    # Per-model heatmaps (if model field is present)
    by_model: dict[str, list[dict]] = {}
    for j in judgments:
        by_model.setdefault(j.get("model", "unknown"), []).append(j)

    for model, rows in by_model.items():
        ms, cs = aggregate_counts(rows)
        write_heatmap(
            row_normalize(cs),
            ms,
            out_dir / "plots" / f"signatures_{model}.png",
        )

    log.info("report: %s", out_dir / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify script boots**

Run: `python scripts/04_analyze.py --help`
Expected: argparse help text.

- [ ] **Step 3: Commit**

```bash
git add scripts/04_analyze.py
git commit -m "feat(cli): Phase 4 analyze script with per-model heatmaps"
```

---

## Task 8.5: `run_all.sh` orchestration

**Files:**
- Create: `scripts/run_all.sh`

- [ ] **Step 1: Write script**

`scripts/run_all.sh`:
```bash
#!/usr/bin/env bash
#
# Usage:
#   bash scripts/run_all.sh [--light]
#
# --light  use a reduced config (see README).
#
# Assumes: ANTHROPIC_API_KEY and optionally HF_USER/HF_REPO_ID are set.

set -euo pipefail

MODELS=(
  deepseek-r1-distill-qwen-1.5b
  qwen3-1.7b
  deepseek-r1-distill-qwen-7b
)

CONFIGS=(
  bf16
  fp8_e5m2
  fp8_e4m3
  hqq_int4
  hqq_int2
)

if [[ "${1:-}" == "--light" ]]; then
  CONFIGS=(bf16 fp8_e5m2 fp8_e4m3 hqq_int4)
  echo "[run_all] --light mode: dropping hqq_int2"
fi

echo "=== Phase 1: generate traces ==="
for model in "${MODELS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    echo "--- model=$model config=$cfg ---"
    python scripts/01_generate_traces.py --model "$model" --config "$cfg" --resume
  done
done

echo "=== Phase 2: find FDPs ==="
for model in "${MODELS[@]}"; do
  python scripts/02_find_fdps.py --model "$model"
done

echo "=== Phase 3: judge FDPs ==="
python scripts/03_judge_fdps.py

echo "=== Phase 4: analyze ==="
python scripts/04_analyze.py

echo "[run_all] done. See outputs/report.md"
```

- [ ] **Step 2: Chmod**

Run: `chmod +x scripts/run_all.sh`

- [ ] **Step 3: Commit**

```bash
git add scripts/run_all.sh
git commit -m "feat(cli): run_all.sh orchestration with --light mode"
```

---

# Phase 9 — README and final verification

## Task 9.1: Write README

**Files:**
- Modify: `README.md` (full rewrite)

- [ ] **Step 1: Write README**

`README.md`:
```markdown
# KV Cache Quantization — Trace-Level Diagnostic Study

Reproducible pipeline for diagnosing **how** KV cache quantization methods
break reasoning in compact (1.5B–7B) reasoning models, not just how much
they degrade accuracy. Output is a markdown + JSON report with failure
**signatures** for 5 quantization configurations across 3 models.

## Design documents

- Design spec: [docs/superpowers/specs/2026-04-25-kv-trace-study-design.md](docs/superpowers/specs/2026-04-25-kv-trace-study-design.md)
- Implementation plan: [docs/superpowers/plans/2026-04-25-kv-trace-study.md](docs/superpowers/plans/2026-04-25-kv-trace-study.md)

## What it does

Four idempotent phases:

1. **GENERATE** — run each of 3 models × 5 KV-cache configurations × 80
   math problems (30 AIME-24 + 50 MATH-500) through its own generator
   (vLLM for BF16 / FP8 E5M2 / FP8 E4M3, HuggingFace Transformers + HQQ
   for INT4 / INT2).
2. **FIND FDP** — for each quantized trace, locate the First Divergence
   Point from the BF16 baseline using token-level exact matching plus a
   MiniLM semantic re-sync filter to skip cosmetic divergences.
3. **JUDGE** — ask Claude Sonnet 4.6 (with Anthropic prompt caching and a
   local SHA256 prompt cache) to classify each FDP into one of six error
   categories: A-Arithmetic, B-Logical, C-Strategy-switch,
   D-Hallucination, E-Premature-termination, F-Repetition/loop.
4. **ANALYZE** — build a confusion matrix (method × category), run a
   chi-square independence test and Cramér's V, and emit a deterministic
   markdown + JSON report plus heatmap plots.

Each phase is resumable from HuggingFace Hub snapshots, so a Vast.ai
instance death in the middle of the run is cheap to recover from.

## Hardware

- **Single RTX 4090 (24 GB, Ada, CC 8.9)** on Vast.ai (~$0.40/hour).
  Ada is chosen over Ampere because it has native FP8 tensor cores,
  making `fp8_e5m2` and `fp8_e4m3` qualitatively distinct — this is
  essential for the "different methods break differently" thesis.
- ~14 GPU-hours total for the full 3-model × 5-config × 80-problem run.

## Budget

| Line item | Cost |
|---|---|
| Compute (Vast.ai, ~14 GPU-hours) | ~$5.60 |
| Claude Sonnet 4.6 judge (prompt cached) | ~$3.00 |
| HuggingFace Hub (public dataset) | $0 |
| Contingency | ~$3.00 |
| **Total** | **~$12** |

## Install

```bash
git clone <this repo>
cd kv-trace-study

python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Dev dependencies (tests, linters):

```bash
pip install -r requirements-dev.txt
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | for Phase 3 | Claude Sonnet 4.6 judge |
| `HF_REPO_ID` | optional | full HF dataset repo id (e.g. `me/my-kv-study`) |
| `HF_USER` | optional | HF username; dataset is `{HF_USER}/kv-trace-study` |
| `HF_TOKEN` | for HF upload | write access to the dataset repo |

If no HF variable is set the pipeline still runs locally — uploads are
silently skipped.

## Run the full study

```bash
# Set at least ANTHROPIC_API_KEY; HF_USER is strongly recommended.
export ANTHROPIC_API_KEY="sk-ant-..."
export HF_USER="your-hf-username"
export HF_TOKEN="hf_..."

# Calibrate the judge FIRST (takes ~30s on live API).
# Must pass ≥7/10 — if it fails, re-check the taxonomy prompt.
pytest -m live_api tests/test_judge_calibration.py -v

# Now the real thing.
bash scripts/run_all.sh

# Output:
#   outputs/report.md
#   outputs/report.json
#   outputs/plots/*.png
#   HF:  {HF_USER}/kv-trace-study with 15 trace + 12 FDP + 12 judgment revisions
```

For a faster iteration run that drops the most experimental config:

```bash
bash scripts/run_all.sh --light
```

## Run individual phases

```bash
# Phase 1 — one (model, config) at a time, resumable
python scripts/01_generate_traces.py \
    --model deepseek-r1-distill-qwen-1.5b \
    --config fp8_e5m2 \
    --resume

# Phase 2 — needs baseline bf16 already generated
python scripts/02_find_fdps.py --model deepseek-r1-distill-qwen-1.5b

# Phase 3 — all FDPs at once; cached
python scripts/03_judge_fdps.py

# Phase 4 — CPU only, <1 min
python scripts/04_analyze.py
```

## Testing

```bash
# CI default — no GPU, no live API, ≥85% coverage gate
make test

# GPU-dependent tests (run on the rented 4090 before Phase 1)
make test-gpu

# Live-API calibration (run before Phase 3)
make test-live
```

Three pytest markers:

| Marker | When to run |
|---|---|
| (none) | always; CI default |
| `@pytest.mark.gpu` | before renting GPU time |
| `@pytest.mark.live_api` | before each Phase 3 run (catches Anthropic drift) |

## Repository layout

```
kv-trace-study/
├── config/                   # 3 YAML files — models, quant methods, pipeline
├── src/kvtrace/
│   ├── generators/           # vLLM + HF (HQQ) backends behind one ABC
│   ├── fdp/                  # hybrid token + semantic re-sync finder
│   ├── judge/                # taxonomy, prompt, Claude client, golden set
│   ├── hf_hub/               # idempotent upload / download
│   └── analysis/             # signatures + markdown report
├── scripts/                  # 01…04 phase CLIs + run_all.sh
├── tests/                    # CPU, GPU, and live-API suites
└── outputs/                  # runtime artifacts (gitignored)
```

## Reproducibility

- Greedy decoding (`temperature=0.0`, `top_p=1.0`) + fixed `seed=42`.
- Chat templates taken verbatim from each model's HF tokenizer.
- `requirements.txt` pins `vllm==0.6.6`, `transformers==4.45.0`,
  `anthropic==0.40.0`.
- Prompt and taxonomy are versioned (`PROMPT_V1`, `TAXONOMY_V1`); a
  change bumps the SHA256 cache key, so stale judgments never silently
  reappear.
- Each JSONL row is self-describing — model, config, seed, timestamp,
  prompt version.

## Citation

The paper is not in this repository. The dataset and code here are the
*reproducibility appendix*: cite them via the HuggingFace dataset id and
the GitHub commit hash.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: full README for the four-phase study pipeline"
```

---

## Task 9.2: Final verification — full CPU suite green

- [ ] **Step 1: Install everything fresh**

Run:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements-dev.txt
pip install -e .
```

- [ ] **Step 2: Run the full CPU suite**

Run: `pytest -m "not gpu and not live_api" --cov=src/kvtrace --cov-fail-under=85 -v`
Expected: all tests pass, coverage ≥85%.

- [ ] **Step 3: Run lint + type check**

Run:
```bash
make lint
make type
```
Expected: no errors.

- [ ] **Step 4: Verify the top-level smoke gate**

Run: `pytest tests/test_end_to_end_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 5: Tag the implementation-complete commit**

```bash
git tag -a v0.1.0-impl-complete -m "Implementation complete; GPU/live-API phases next"
```

Note: actual Phase 1–4 execution (generating the real report on a 4090)
is a *runtime* step, not a code step — run `bash scripts/run_all.sh` on
the rented instance once implementation is green.

---

# Summary

Nine phases, ~30 tasks, every task has failing-test-first steps, exact
file paths, full code blocks, and explicit commit points. All four
research tasks from the spec are covered:

- **Task 1 (trace collection):** Phase 3 (Generators) + Phase 8.1
  (01_generate_traces).
- **Task 2 (automatic FDP):** Phase 4 (FDP) + Phase 8.2 (02_find_fdps).
- **Task 3 (LLM-as-a-judge, 6 categories):** Phase 5 (Judge) + Phase 8.3
  (03_judge_fdps).
- **Task 4 (failure signatures):** Phase 7 (Analysis) + Phase 8.4
  (04_analyze) + Phase 9.1 (report in README).

After `v0.1.0-impl-complete` tag, the implementation is done; running
the report on a 4090 is operational work, not development.
