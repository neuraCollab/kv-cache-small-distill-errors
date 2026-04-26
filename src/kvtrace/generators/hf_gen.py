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
    ) -> None:
        self.sampling_temperature = sampling_temperature
        self.sampling_top_p = sampling_top_p
        self.sampling_max_tokens = sampling_max_tokens
        self.sampling_repetition_penalty = sampling_repetition_penalty
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

    def generate(self, problems: list[MathProblem]) -> list[GenerationResult]:
        assert (
            self._model is not None
            and self._tokenizer is not None
            and self._model_cfg is not None
        ), "call load() first"

        results: list[GenerationResult] = []
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
        cache_config = QuantizedCacheConfig(
            backend="HQQ",
            nbits=self._hqq_nbits,
            device=cache_device,
            compute_dtype=compute_dtype,
        )

        for p in problems:
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
            # Without an explicit attention_mask + pad_token_id, HF generate
            # emits a per-call warning ("you may observe unexpected behavior")
            # that floods the log 80x per (model, config) and risks subtly
            # incorrect attention on tokenizers where pad_token_id is unset.
            # batch=1 with no padding => mask is all 1s.
            attention_mask = input_ids.new_ones(input_ids.shape)
            pad_token_id = (
                self._tokenizer.pad_token_id
                if getattr(self._tokenizer, "pad_token_id", None) is not None
                else self._tokenizer.eos_token_id
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
