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
