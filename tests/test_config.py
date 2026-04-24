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
