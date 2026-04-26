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
    # Per DeepSeek R1-Distill guidance: greedy decoding (T=0) frequently
    # triggers degenerate repetition loops on hard reasoning problems. A small
    # repetition_penalty (>1.0, e.g. 1.05) breaks loops while remaining
    # deterministic — preserves the divergence-study property that two runs of
    # the same (model, quant) pair produce identical token streams.
    repetition_penalty: float = 1.0
    # HF Transformers backend (HQQ configs) processes prompts in batches of
    # this size per `model.generate()` call. vLLM ignores this — it has its
    # own continuous batching. Higher values raise GPU utilization on the
    # otherwise sequential HQQ path; 4 is comfortable for 1.5B/1.7B/7B at
    # 16K context on a 24-48 GB GPU.
    hf_batch_size: int = 1


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
