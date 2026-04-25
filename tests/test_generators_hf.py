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
