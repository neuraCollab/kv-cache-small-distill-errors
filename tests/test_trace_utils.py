from kvtrace.trace_utils import (
    extract_boxed_answer,
    extract_think_block,
    parse_trace,
)


def test_parse_deepseek_r1_common_case():
    text = (
        "Let me work this out.\nFirst, try \\boxed{7} as a guess... no.\n"
        "</think>\n\nThe answer is \\boxed{\\dfrac{22}{7}}."
    )
    p = parse_trace(text, finish_reason="stop")
    assert p.think is not None and "try \\boxed{7}" in p.think
    assert p.boxed_answer == "\\dfrac{22}{7}"
    assert p.think_complete is True
    assert p.final_response.startswith("The answer is")


def test_parse_truncated_trace():
    text = "<think>\nI am still thinking about this when abruptly"
    p = parse_trace(text, finish_reason="length")
    assert p.think is not None
    assert p.think.startswith("I am still thinking")
    assert p.think_complete is False
    assert p.final_response == ""
    assert p.boxed_answer is None


def test_parse_no_think_tag():
    text = "The answer is \\boxed{42}."
    p = parse_trace(text)
    assert p.think is None
    assert p.boxed_answer == "42"


def test_boxed_answer_nested_braces():
    assert extract_boxed_answer(r"foo \boxed{\frac{a}{b}} bar") == r"\frac{a}{b}"


def test_boxed_answer_takes_last_match():
    assert extract_boxed_answer(r"\boxed{1} then \boxed{2}") == "2"


def test_boxed_answer_no_match_returns_none():
    assert extract_boxed_answer("no box here") is None


def test_extract_think_block_both_tags():
    text = "<think>reasoning</think>final"
    think, remaining, closed = extract_think_block(text)
    assert think == "reasoning"
    assert remaining == "final"
    assert closed is True
