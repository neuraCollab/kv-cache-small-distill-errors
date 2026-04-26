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
        sampling_repetition_penalty: float = 1.0,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        self.seed = seed
        self.sampling_temperature = sampling_temperature
        self.sampling_top_p = sampling_top_p
        self.sampling_max_tokens = sampling_max_tokens
        self.sampling_repetition_penalty = sampling_repetition_penalty
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
            repetition_penalty=self.sampling_repetition_penalty,
            seed=self.seed,
        )
        outputs = self._llm.generate(prompts, sp, use_tqdm=True)

        results: list[GenerationResult] = []
        for problem, req in zip(problems, outputs, strict=False):
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
