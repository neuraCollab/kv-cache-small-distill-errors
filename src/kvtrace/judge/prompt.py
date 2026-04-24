"""Judge prompt construction with Anthropic prompt caching.

The taxonomy + schema block is marked `cache_control: {"type": "ephemeral"}`
so Anthropic caches it for ~5 minutes; 960 judge calls in one run pay the
full cost once and read from the cache for the rest.
"""
from __future__ import annotations

from kvtrace.judge.taxonomy import CATEGORIES, TAXONOMY_VERSION

PROMPT_VERSION = "v1"

_SYSTEM_HEADER = (
    "You are an expert in analyzing long chain-of-thought reasoning traces "
    "from small language models. Your job is to classify the FIRST point "
    "where a quantized-KV-cache trace diverges from its FP16 baseline into "
    "exactly one of six error categories. Be strict: choose the category "
    "that describes the LOCAL error at the divergence, not downstream "
    "consequences."
)

_SCHEMA = """
Output a single JSON object with EXACTLY these keys:

{
  "category": "A" | "B" | "C" | "D" | "E" | "F",
  "confidence": 0.0 to 1.0,
  "rationale": string, at most 2 sentences,
  "affected_span": string, a 3-15 word quote from the QUANTIZED trace
}

Do not output any text outside the JSON.
""".strip()


def _taxonomy_text() -> str:
    lines = [f"TAXONOMY (v: {TAXONOMY_VERSION})", ""]
    for c in CATEGORIES:
        lines.append(f"{c.letter}. {c.name} — {c.description}")
        for ex in c.examples:
            lines.append(f"   • {ex}")
        lines.append("")
    return "\n".join(lines)


def build_system_block() -> list[dict]:
    """Return the Anthropic `system` list with cache_control on the taxonomy."""
    static = "\n\n".join([_SYSTEM_HEADER, _taxonomy_text(), _SCHEMA])
    return [
        {
            "type": "text",
            "text": static,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def build_judge_messages(
    *,
    problem: str,
    ground_truth: str,
    baseline_context: str,
    quant_context: str,
    common_prefix: str,
    quant_method_name: str,
) -> list[dict]:
    """Return the `messages` list for anthropic.Messages.create(...)."""
    content = (
        f"Problem:\n{problem}\n\n"
        f"Ground truth answer: {ground_truth}\n\n"
        f"--- Common prefix (last part, same in both traces) ---\n"
        f"{common_prefix}\n\n"
        f"--- BASELINE TRACE (FP16) around divergence ---\n"
        f"{baseline_context}\n\n"
        f"--- QUANTIZED TRACE ({quant_method_name}) around divergence ---\n"
        f"{quant_context}\n\n"
        "Classify the divergence in the quantized trace. Respond with the JSON object only."
    )
    return [{"role": "user", "content": content}]
