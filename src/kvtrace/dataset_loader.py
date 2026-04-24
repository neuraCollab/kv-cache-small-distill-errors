"""Math dataset loader. Extends the baseline with load_study_mix()."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset


@dataclass
class MathProblem:
    idx: int
    problem: str
    answer: str
    source: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "problem": self.problem,
            "ground_truth": self.answer,
            "source": self.source,
            "metadata": self.metadata,
        }


DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "aime-24": {
        "path": "Maxwell-Jia/AIME_2024",
        "split": "train",
        "problem_key": "Problem",
        "answer_key": "Answer",
    },
    "aime-24-aimo": {
        "path": "AI-MO/aimo-validation-aime",
        "split": "train",
        "problem_key": "problem",
        "answer_key": "answer",
    },
    "math-500": {
        "path": "HuggingFaceH4/MATH-500",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
    },
}


def load_math_dataset(
    name: str = "aime-24",
    num_samples: int | None = None,
    seed: int = 42,
    shuffle: bool = False,
) -> list[MathProblem]:
    if name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_CONFIGS)}"
        )

    cfg = DATASET_CONFIGS[name]
    ds = load_dataset(cfg["path"], split=cfg["split"])

    if shuffle:
        ds = ds.shuffle(seed=seed)
    if num_samples is not None:
        n = min(num_samples, len(ds))
        ds = ds.select(range(n))

    pk, ak = cfg["problem_key"], cfg["answer_key"]
    out: list[MathProblem] = []
    for i, row in enumerate(ds):
        md = {k: v for k, v in row.items() if k not in (pk, ak)}
        out.append(
            MathProblem(
                idx=i,
                problem=row[pk],
                answer=str(row[ak]),
                source=name,
                metadata=md,
            )
        )
    return out


def load_study_mix(
    aime_n: int = 30,
    math_n: int = 50,
) -> list[MathProblem]:
    """Canonical 80-problem study set: first N AIME-24, then first M MATH-500.

    Ordering is deterministic with shuffle=False — baseline and quantized
    runs see identical problems in identical order.
    """
    aime = load_math_dataset("aime-24", num_samples=aime_n, shuffle=False)
    math = load_math_dataset("math-500", num_samples=math_n, shuffle=False)
    combined: list[MathProblem] = []
    for i, p in enumerate(aime + math):
        combined.append(
            MathProblem(
                idx=i,
                problem=p.problem,
                answer=p.answer,
                source=p.source,
                metadata=p.metadata,
            )
        )
    return combined
