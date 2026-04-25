"""
dataset_loader.py
-----------------
Load mathematical reasoning datasets for the DeepSeek-R1 KV cache trace study.

Supports:
  * "aime-24"       -> Maxwell-Jia/AIME_2024   (30 problems, AIME 2024 I + II)
  * "aime-24-aimo"  -> AI-MO/aimo-validation-aime  (backup mirror)
  * "math-500"      -> HuggingFaceH4/MATH-500  (500 problems, mixed difficulty)

Each problem is normalized into a MathProblem dataclass so the pipeline
is dataset-agnostic downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset


@dataclass
class MathProblem:
    """Canonical problem record emitted by this loader."""

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


# Dataset schema registry. Add new sources here and the rest of the
# pipeline picks them up automatically.
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
    """Load a math dataset and normalize it into MathProblem records.

    Args:
        name: One of the keys in DATASET_CONFIGS.
        num_samples: If set, take the first N records (after optional shuffle).
        seed: Shuffle seed (only used if shuffle=True).
        shuffle: If True, shuffle the dataset before slicing.
            For a *diagnostic* trace study you almost always want shuffle=False
            so that the baseline run and the quantized run see the exact same
            problems in the exact same order.

    Returns:
        A list of MathProblem dataclasses.
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {sorted(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[name]
    ds = load_dataset(cfg["path"], split=cfg["split"])

    if shuffle:
        ds = ds.shuffle(seed=seed)

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    problem_key = cfg["problem_key"]
    answer_key = cfg["answer_key"]

    problems: list[MathProblem] = []
    for i, row in enumerate(ds):
        problem_text = row[problem_key]
        # Some datasets store answers as ints; coerce to str for a uniform schema.
        answer_text = str(row[answer_key])
        metadata = {k: v for k, v in row.items() if k not in (problem_key, answer_key)}
        problems.append(
            MathProblem(
                idx=i,
                problem=problem_text,
                answer=answer_text,
                source=name,
                metadata=metadata,
            )
        )
    return problems


if __name__ == "__main__":
    # Quick smoke test: python dataset_loader.py
    import json

    probs = load_math_dataset("aime-24", num_samples=2)
    for p in probs:
        print(json.dumps(p.to_dict(), indent=2, ensure_ascii=False)[:500])
        print("---")
