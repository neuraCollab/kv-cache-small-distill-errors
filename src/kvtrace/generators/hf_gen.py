"""HuggingFace Transformers backend for HQQ INT4/INT2 KV cache."""
from __future__ import annotations

import logging
from typing import Any

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:
    AutoModelForCausalLM = None   # type: ignore[assignment]
    AutoTokenizer = None          # type: ignore[assignment]

# QuantizedCacheConfig is only present in newer transformers (≥4.43); keep the
# import isolated so a missing symbol on older installs doesn't also null
# AutoModelForCausalLM / AutoTokenizer above. Tests that exercise generate()
# patch this symbol explicitly.
try:
    from transformers.cache_utils import QuantizedCacheConfig  # type: ignore
except Exception:
    QuantizedCacheConfig = None   # type: ignore[assignment]

# torch is optional at import time so CI (CPU-only, no torch) can still import
# this module to test the contract via mocks. The actual GPU path requires it.
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore[assignment]

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
        sampling_repetition_penalty: float = 1.0,
        batch_size: int = 1,
    ) -> None:
        self.sampling_temperature = sampling_temperature
        self.sampling_top_p = sampling_top_p
        self.sampling_max_tokens = sampling_max_tokens
        self.sampling_repetition_penalty = sampling_repetition_penalty
        if batch_size < 1:
            raise ValueError(f"batch_size must be ≥1, got {batch_size}")
        self.batch_size = batch_size
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_cfg: ModelCfg | None = None
        self._hqq_nbits: int | None = None

    def load(self, model_cfg: ModelCfg, quant_cfg: QuantCfg) -> None:
        if quant_cfg.engine != "hf":
            raise ValueError(f"HFGenerator requires engine='hf', got {quant_cfg.engine!r}")
        if quant_cfg.hqq_nbits not in (2, 4, 8):
            raise ValueError(f"HQQ nbits must be 2/4/8, got {quant_cfg.hqq_nbits!r}")

        self._hqq_nbits = quant_cfg.hqq_nbits
        self._model_cfg = model_cfg
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.hf_id, trust_remote_code=model_cfg.trust_remote_code
        )
        # Causal LMs require LEFT padding for batched generation: the model
        # autoregresses from the rightmost prompt token, and right-padding
        # would put pad tokens in that position and produce garbage. Also
        # ensure pad_token_id is set — many R1/Qwen tokenizers leave it None
        # which makes the padded batch path crash.
        self._tokenizer.padding_side = "left"
        if getattr(self._tokenizer, "pad_token_id", None) is None:
            # Fall back to eos. Safe because attention_mask zeros out these
            # positions, so pad_token_id == eos_token_id never confuses the
            # model into early-stopping the prompt.
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if torch is not None:
            dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
            torch_dtype = dtype_map.get(model_cfg.dtype, torch.bfloat16)
        else:
            # CI (mocked AutoModelForCausalLM) doesn't need a real torch dtype.
            torch_dtype = None
        # Force single-GPU placement. `device_map="auto"` lets accelerate
        # offload buffers to CPU when it thinks memory is tight, which
        # silently breaks HQQ's quantized KV cache (it does in-place dequant
        # against `meta["zero"] / meta["scale"]` and can't follow tensors
        # across devices). On the target GPUs (≥24 GB) the 1.5B/1.7B/7B
        # models all fit comfortably on cuda:0, so single-device placement
        # is correct and avoids the cuda:0/cpu mismatch crash.
        device_map: Any = {"": 0} if torch is not None else "auto"
        self._model = AutoModelForCausalLM.from_pretrained(
            model_cfg.hf_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=model_cfg.trust_remote_code,
        )

    def _tokenize_prompt(self, problem: MathProblem) -> Any:
        """Tokenize a single problem into a 1-D LongTensor (no padding)."""
        enc = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{problem.problem}\n\n{DEFAULT_USER_INSTRUCTION}"}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **(self._model_cfg.chat_template_kwargs if self._model_cfg else {}),
        )
        # apply_chat_template returns either a Tensor of shape [1, L] or a
        # BatchEncoding-like dict with "input_ids" of shape [1, L].
        if hasattr(enc, "shape"):
            ids = enc
        else:
            ids = enc["input_ids"]
        # Squeeze the leading batch dim so we get a 1-D tensor [L].
        if hasattr(ids, "dim") and ids.dim() == 2:
            ids = ids.squeeze(0)
        return ids

    def generate(self, problems: list[MathProblem]) -> list[GenerationResult]:
        assert (
            self._model is not None
            and self._tokenizer is not None
            and self._model_cfg is not None
        ), "call load() first"

        # `QuantizedCacheConfig.device` defaults to "cpu" in transformers 4.45.
        # If we leave it default, HQQ allocates the quantized KV buffers on
        # CPU while the model runs on GPU and dequant fails with
        # "found at least two devices, cuda:0 and cpu!". Pin the cache to
        # the model's actual device.
        if torch is not None and torch.cuda.is_available():
            cache_device = "cuda"
            compute_dtype = torch.bfloat16
        else:
            cache_device = "cpu"
            compute_dtype = None

        pad_token_id = (
            self._tokenizer.pad_token_id
            if getattr(self._tokenizer, "pad_token_id", None) is not None
            else self._tokenizer.eos_token_id
        )

        # Pre-tokenize every prompt as a 1-D tensor; we need individual
        # prompt lengths to slice the right span out of each generated row.
        encoded: list[Any] = [self._tokenize_prompt(p) for p in problems]

        results: list[GenerationResult] = []
        for chunk_start in range(0, len(problems), self.batch_size):
            chunk_problems = problems[chunk_start : chunk_start + self.batch_size]
            chunk_ids = encoded[chunk_start : chunk_start + self.batch_size]

            prompt_lens = [int(t.shape[-1]) for t in chunk_ids]
            max_len = max(prompt_lens)
            n = len(chunk_ids)

            # Allocate padded input_ids + attention_mask using a representative
            # tensor's `.new_*` so dtype/device match the tokenizer output.
            ref = chunk_ids[0]
            input_ids = ref.new_full((n, max_len), pad_token_id)
            attention_mask = ref.new_zeros((n, max_len))
            for i, ids in enumerate(chunk_ids):
                L = int(ids.shape[-1])
                # LEFT pad: real tokens go to the right edge.
                input_ids[i, max_len - L :] = ids
                attention_mask[i, max_len - L :] = 1

            # Move to model device once per chunk.
            input_ids = input_ids.to(self._model.device)
            attention_mask = attention_mask.to(self._model.device)

            # QuantizedCacheConfig is stateful — give each chunk its own so
            # leftover KV buffers from the previous batch can't bleed in.
            cache_config = QuantizedCacheConfig(
                backend="HQQ",
                nbits=self._hqq_nbits,
                device=cache_device,
                compute_dtype=compute_dtype,
            )

            out = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                max_new_tokens=self.sampling_max_tokens,
                do_sample=(self.sampling_temperature > 0.0),
                temperature=self.sampling_temperature or None,
                top_p=self.sampling_top_p,
                repetition_penalty=self.sampling_repetition_penalty,
                cache_implementation="quantized",
                cache_config=cache_config,
                return_dict_in_generate=True,
            )

            # out.sequences shape: [n, max_len + new_tokens]
            # The first `max_len` columns are the (left-padded) prompt; the
            # generated tokens for every row start at column `max_len`.
            for i, p in enumerate(chunk_problems):
                seq = out.sequences[i].tolist()
                gen_ids = seq[max_len:]
                # Rows that hit EOS early are right-padded by `generate` with
                # pad_token_id. Strip the trailing pad run so we don't decode
                # garbage <pad>… on short rows. Use eos as a secondary stop —
                # when pad falls back to eos they're the same id, but on
                # tokenizers with both, eos belongs in the trace and pad does
                # not.
                eos_id = self._tokenizer.eos_token_id
                while gen_ids and gen_ids[-1] == pad_token_id and pad_token_id != eos_id:
                    gen_ids.pop()
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
                        prompt_tokens=prompt_lens[i],
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
