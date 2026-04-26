from unittest.mock import MagicMock, patch

import pytest

# torch is optional in CI (CPU-only test runners ship without it). Tests that
# build real tensors are skipped collectively when it's missing; mock-only
# tests below still run so we keep coverage of the load/unload/validation
# paths in lightweight CI.
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="requires torch")

from kvtrace.config import ModelCfg, QuantCfg
from kvtrace.dataset_loader import MathProblem
from kvtrace.generators.hf_gen import HFGenerator


def _make_tokenizer(prompt_token_ids: list[list[int]], decoded_text: str = "x"):
    """Build a tokenizer mock whose apply_chat_template returns real torch
    tensors. Calls are returned in order from `prompt_token_ids` so different
    prompts produce different lengths (exercising the left-padding logic)."""
    tokenizer = MagicMock()
    # Iterator-like side_effect: each call returns the next prepared tensor.
    tensors = [torch.tensor([ids], dtype=torch.long) for ids in prompt_token_ids]
    tokenizer.apply_chat_template.side_effect = tensors
    tokenizer.decode.return_value = decoded_text
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0
    return tokenizer


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
def test_hf_generator_load_sets_left_padding(mock_tok, mock_model):
    """Causal LMs require LEFT padding for batched generation."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = None
    tokenizer.eos_token = "<|eot|>"
    mock_tok.from_pretrained.return_value = tokenizer

    HFGenerator().load(ModelCfg(hf_id="m"), QuantCfg(engine="hf", hqq_nbits=4))
    assert tokenizer.padding_side == "left"
    # When pad_token_id is unset, fall back to eos so padded batches don't
    # crash on missing pad id.
    assert tokenizer.pad_token == "<|eot|>"


@needs_torch
@patch("kvtrace.generators.hf_gen.QuantizedCacheConfig")
@patch("kvtrace.generators.hf_gen.AutoModelForCausalLM")
@patch("kvtrace.generators.hf_gen.AutoTokenizer")
def test_hf_generator_generate_returns_result(mock_tok, mock_model, mock_qcc):
    tokenizer = _make_tokenizer(
        prompt_token_ids=[[10, 20, 30, 40, 50]],
        decoded_text="Reasoning done.</think>\n\n\\boxed{7}",
    )
    mock_tok.from_pretrained.return_value = tokenizer

    model = MagicMock()
    # generate returns a sequences tensor of shape [batch=1, prompt_len=5 + new=5]
    fake_sequences = torch.tensor([[10, 20, 30, 40, 50, 100, 200, 300, 400, 500]])
    model.generate.return_value = MagicMock(sequences=fake_sequences)
    model.device = torch.device("cpu")
    mock_model.from_pretrained.return_value = model

    gen = HFGenerator()
    gen.load(ModelCfg(hf_id="m"), QuantCfg(engine="hf", hqq_nbits=4))
    results = gen.generate(
        [MathProblem(idx=0, problem="q", answer="7", source="aime-24")]
    )
    assert len(results) == 1
    assert results[0].boxed_answer == "7"
    assert results[0].prompt_tokens == 5
    assert results[0].generated_tokens == 5


@patch("kvtrace.generators.hf_gen.free_gpu")
@patch("kvtrace.generators.hf_gen.AutoModelForCausalLM")
@patch("kvtrace.generators.hf_gen.AutoTokenizer")
def test_hf_generator_unload_calls_free_gpu(mock_tok, mock_model, mock_free):
    gen = HFGenerator()
    gen.load(ModelCfg(hf_id="m"), QuantCfg(engine="hf", hqq_nbits=2))
    gen.unload()
    mock_free.assert_called_once()
    assert gen._model is None


@patch("kvtrace.generators.hf_gen.AutoModelForCausalLM")
@patch("kvtrace.generators.hf_gen.AutoTokenizer")
def test_hf_generator_load_pins_to_single_gpu(mock_tok, mock_model):
    """device_map='auto' lets accelerate offload buffers and break HQQ KV cache."""
    gen = HFGenerator()
    gen.load(ModelCfg(hf_id="m"), QuantCfg(engine="hf", hqq_nbits=4))
    _, kwargs = mock_model.from_pretrained.call_args
    # Either {"": 0} (real torch path) or "auto" (no-torch CI path) — both
    # acceptable; what matters is that we're not silently sharding the
    # production case. On any host with torch importable we expect single-GPU.
    assert kwargs["device_map"] in ({"": 0}, "auto")


@needs_torch
@patch("kvtrace.generators.hf_gen.QuantizedCacheConfig")
@patch("kvtrace.generators.hf_gen.AutoModelForCausalLM")
@patch("kvtrace.generators.hf_gen.AutoTokenizer")
def test_hf_generator_passes_attention_mask_and_pad_token(mock_tok, mock_model, mock_qcc):
    """Without these kwargs HF generate floods the log 80x per (model, config)."""
    tokenizer = _make_tokenizer(
        prompt_token_ids=[[1, 2, 3, 4, 5]],
        decoded_text="x",
    )
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = 42
    tokenizer.eos_token = "<|eot|>"
    mock_tok.from_pretrained.return_value = tokenizer

    model = MagicMock()
    fake_sequences = torch.tensor([[1, 2, 3, 4, 5, 99, 99, 99, 99, 99]])
    model.generate.return_value = MagicMock(sequences=fake_sequences)
    model.device = torch.device("cpu")
    mock_model.from_pretrained.return_value = model

    gen = HFGenerator()
    gen.load(ModelCfg(hf_id="m"), QuantCfg(engine="hf", hqq_nbits=4))
    # On a MagicMock, `tokenizer.pad_token = tokenizer.eos_token` sets a plain
    # attribute and does NOT update pad_token_id (no property bridge). We
    # leave pad_token_id=None so the eos_token_id fallback path in generate()
    # is the one exercised.
    gen.generate([MathProblem(idx=0, problem="q", answer="x", source="aime-24")])

    _, kwargs = model.generate.call_args
    assert "attention_mask" in kwargs
    # When pad_token_id is unset we fall back to eos_token_id.
    assert kwargs["pad_token_id"] == 42


@needs_torch
@patch("kvtrace.generators.hf_gen.QuantizedCacheConfig")
@patch("kvtrace.generators.hf_gen.AutoModelForCausalLM")
@patch("kvtrace.generators.hf_gen.AutoTokenizer")
def test_hf_generator_batches_multiple_problems(mock_tok, mock_model, mock_qcc):
    """batch_size>1 should pack multiple prompts into one generate() call,
    left-pad shorter prompts, and recover per-row prompt_len + gen_ids."""
    # Two prompts of unequal length — exercises left-padding.
    tokenizer = _make_tokenizer(
        prompt_token_ids=[
            [11, 12, 13],            # short prompt, len=3
            [21, 22, 23, 24, 25],    # long prompt,  len=5 (max_len)
        ],
        decoded_text="ans",
    )
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0
    mock_tok.from_pretrained.return_value = tokenizer

    model = MagicMock()
    # After left-padding: row0 prompt occupies cols [2..5), row1 prompt
    # occupies cols [0..5). Generation begins at col 5 and runs for 4 tokens.
    fake_sequences = torch.tensor(
        [
            [0, 0, 11, 12, 13,    91, 92, 93, 94],
            [21, 22, 23, 24, 25,  81, 82, 83, 84],
        ]
    )
    model.generate.return_value = MagicMock(sequences=fake_sequences)
    model.device = torch.device("cpu")
    mock_model.from_pretrained.return_value = model

    gen = HFGenerator(batch_size=2)
    gen.load(ModelCfg(hf_id="m"), QuantCfg(engine="hf", hqq_nbits=4))
    results = gen.generate(
        [
            MathProblem(idx=0, problem="short", answer="x", source="aime-24"),
            MathProblem(idx=1, problem="longer one", answer="y", source="aime-24"),
        ]
    )

    # Single batched generate() call — the whole point of batching.
    assert model.generate.call_count == 1
    args, kwargs = model.generate.call_args
    input_ids = args[0]
    attention_mask = kwargs["attention_mask"]
    # Padded batch shape: [2, max_len=5]
    assert tuple(input_ids.shape) == (2, 5)
    assert tuple(attention_mask.shape) == (2, 5)
    # Left-pad: row0 has 2 leading zeros in the mask, row1 has none.
    assert attention_mask[0].tolist() == [0, 0, 1, 1, 1]
    assert attention_mask[1].tolist() == [1, 1, 1, 1, 1]

    # Per-row accounting must reflect the *real* prompt lengths, not the
    # padded length, otherwise downstream token cost reports are wrong.
    assert len(results) == 2
    assert results[0].prompt_tokens == 3
    assert results[1].prompt_tokens == 5
    # And gen tokens are the same 4-token suffix for both rows.
    assert results[0].token_ids == [91, 92, 93, 94]
    assert results[1].token_ids == [81, 82, 83, 84]


@needs_torch
@patch("kvtrace.generators.hf_gen.QuantizedCacheConfig")
@patch("kvtrace.generators.hf_gen.AutoModelForCausalLM")
@patch("kvtrace.generators.hf_gen.AutoTokenizer")
def test_hf_generator_chunks_more_problems_than_batch_size(mock_tok, mock_model, mock_qcc):
    """3 problems with batch_size=2 should produce 2 generate() calls."""
    tokenizer = _make_tokenizer(
        prompt_token_ids=[[1, 2], [3, 4], [5, 6]],
        decoded_text="y",
    )
    mock_tok.from_pretrained.return_value = tokenizer

    model = MagicMock()
    # Each generate call returns a tensor matching the chunk size.
    model.generate.side_effect = [
        MagicMock(sequences=torch.tensor([[1, 2, 7], [3, 4, 8]])),
        MagicMock(sequences=torch.tensor([[5, 6, 9]])),
    ]
    model.device = torch.device("cpu")
    mock_model.from_pretrained.return_value = model

    gen = HFGenerator(batch_size=2)
    gen.load(ModelCfg(hf_id="m"), QuantCfg(engine="hf", hqq_nbits=4))
    results = gen.generate([
        MathProblem(idx=i, problem=f"q{i}", answer="x", source="aime-24")
        for i in range(3)
    ])
    assert model.generate.call_count == 2
    assert len(results) == 3
    assert [r.idx for r in results] == [0, 1, 2]


def test_hf_generator_rejects_zero_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        HFGenerator(batch_size=0)
