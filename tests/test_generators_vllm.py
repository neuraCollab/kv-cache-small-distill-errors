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
