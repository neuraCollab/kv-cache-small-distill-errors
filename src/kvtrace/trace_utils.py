"""
trace_utils.py
--------------
Parse DeepSeek-R1 distill outputs into (think trace, final response, boxed answer).

The DeepSeek-R1 distill chat template *pre-seeds* the assistant turn with
``<think>\\n`` before generation starts. That means the generated text usually
looks like this:

    <reasoning content, starts mid-thought>
    ...
    </think>
    <final response, often ending in \\boxed{...}>

so the generated string typically has a *closing* </think> but no *opening*
<think>. This module handles all four observed cases:

  (1) Has both <think> and </think>  -> standard block
  (2) Has </think> only              -> DeepSeek-R1 common case
  (3) Has <think> only               -> truncated (hit max_tokens mid-trace)
  (4) Has neither                    -> model produced no think tag at all

We also extract the *last* ``\\boxed{...}`` in the post-think region as the
canonical final answer, since R1 sometimes shows intermediate boxed work.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

# --- Regexes ----------------------------------------------------------------

THINK_OPEN_RE = re.compile(r"<think>", flags=re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)


def _find_matching_brace(text: str, open_idx: int) -> int:
    """Return the index of the '}' that matches the '{' at open_idx.

    Handles nested braces (LaTeX like \\boxed{\\frac{a}{b}}). Returns -1 if unmatched.
    """
    assert text[open_idx] == "{"
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_boxed_answer(text: str) -> Optional[str]:
    """Return the content of the *last* \\boxed{...} in text, or None.

    Uses a brace-matching scan rather than a flat regex so it correctly
    handles nested braces like \\boxed{\\dfrac{22}{7}}.
    """
    results = []
    i = 0
    needle = r"\boxed"
    while True:
        j = text.find(needle, i)
        if j == -1:
            break
        k = j + len(needle)
        # Skip optional whitespace between \boxed and {
        while k < len(text) and text[k].isspace():
            k += 1
        if k < len(text) and text[k] == "{":
            end = _find_matching_brace(text, k)
            if end != -1:
                results.append(text[k + 1 : end].strip())
                i = end + 1
                continue
        i = j + 1
    if not results:
        return None
    return results[-1]


def extract_think_block(text: str) -> Tuple[Optional[str], str, bool]:
    """Split text into (think_content, remaining_text, think_was_closed).

    Strategy:
      - If both <think> and </think> are present, extract the enclosed block.
      - If only </think> is present (standard DeepSeek-R1 distill output),
        everything before it is the trace and everything after is the response.
      - If only <think> is present (output truncated), everything after it is
        the trace and remaining is empty; think_was_closed=False.
      - If neither is present, there is no think block; treat the entire text
        as the final response.
    """
    has_open = bool(THINK_OPEN_RE.search(text))
    close_match = THINK_CLOSE_RE.search(text)

    if close_match:
        if has_open:
            m = THINK_BLOCK_RE.search(text)
            if m:
                think = m.group(1).strip()
                remaining = (text[: m.start()] + text[m.end():]).strip()
                return think, remaining, True
        # Close tag only (common DeepSeek-R1 case, or open tag was eaten by prompt)
        think = text[: close_match.start()].strip()
        remaining = text[close_match.end():].strip()
        return think, remaining, True

    # No closing tag
    if has_open:
        open_match = THINK_OPEN_RE.search(text)
        assert open_match is not None
        think = text[open_match.end():].strip()
        return think, "", False

    # No think markers at all
    return None, text.strip(), False


@dataclass
class ParsedTrace:
    raw: str
    think: Optional[str]
    final_response: str
    boxed_answer: Optional[str]
    think_complete: bool
    finish_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "think": self.think,
            "final_response": self.final_response,
            "boxed_answer": self.boxed_answer,
            "think_complete": self.think_complete,
            "finish_reason": self.finish_reason,
        }


def parse_trace(text: str, finish_reason: Optional[str] = None) -> ParsedTrace:
    """Parse a raw DeepSeek-R1 generation into its structured components."""
    think, remaining, closed = extract_think_block(text)
    # Prefer a boxed answer found in the post-think region; fall back to
    # scanning the full text (rare - only if </think> was never emitted).
    boxed = extract_boxed_answer(remaining) if remaining else None
    if boxed is None:
        boxed = extract_boxed_answer(text)
    return ParsedTrace(
        raw=text,
        think=think,
        final_response=remaining,
        boxed_answer=boxed,
        think_complete=closed,
        finish_reason=finish_reason,
    )


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # DeepSeek-R1 common case: close tag only, nested \boxed{}
    sample_1 = (
        "Let me work this out.\nFirst, try \\boxed{7} as a guess... no.\n"
        "</think>\n\nThe answer is \\boxed{\\dfrac{22}{7}}."
    )
    p = parse_trace(sample_1, finish_reason="stop")
    assert p.think is not None and "try \\boxed{7}" in p.think
    assert p.boxed_answer == "\\dfrac{22}{7}", p.boxed_answer
    assert p.think_complete is True

    # Truncated trace: open tag only
    sample_2 = "<think>\nI am still thinking about this when abruptly"
    p = parse_trace(sample_2, finish_reason="length")
    assert p.think is not None and p.think.startswith("I am still thinking")
    assert p.think_complete is False
    assert p.final_response == ""
    assert p.boxed_answer is None

    # No think tag at all
    sample_3 = "The answer is \\boxed{42}."
    p = parse_trace(sample_3)
    assert p.think is None
    assert p.boxed_answer == "42"

    print("trace_utils self-test OK")
