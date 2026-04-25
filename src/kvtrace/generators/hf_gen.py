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
        self._model = AutoModelForCausalLM.from_pretrained(
            model_cfg.hf_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=model_cfg.trust_remote_code,
        )

    def generate(self, problems: list[MathProblem]) -> list[GenerationResult]:
        assert (
            self._model is not None
            and self._tokenizer is not None
            and self._model_cfg is not None
        ), "call load() first"

        results: list[GenerationResult] = []
        cache_config = QuantizedCacheConfig(backend="HQQ", nbits=self._hqq_nbits)

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
