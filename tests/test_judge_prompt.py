from kvtrace.judge.prompt import PROMPT_VERSION, build_judge_messages, build_system_block


def test_prompt_version():
    assert PROMPT_VERSION == "v1"


def test_system_block_contains_all_categories():
    sys_blocks = build_system_block()
    text = " ".join(b["text"] for b in sys_blocks)
    for letter in "ABCDEF":
        assert f" {letter}." in text or f" {letter} " in text or f"{letter}:" in text


def test_system_block_has_cache_control():
    sys_blocks = build_system_block()
    assert any("cache_control" in b for b in sys_blocks)


def test_build_judge_messages_round_trip():
    msgs = build_judge_messages(
        problem="compute 2+2",
        ground_truth="4",
        baseline_context="result = 4",
        quant_context="result = 5",
        common_prefix="...",
        quant_method_name="fp8_e5m2",
    )
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert "compute 2+2" in content
    assert "result = 4" in content
    assert "result = 5" in content
    assert "fp8_e5m2" in content


def test_user_message_length_is_reasonable():
    msgs = build_judge_messages(
        problem="x" * 1000,
        ground_truth="y",
        baseline_context="a" * 1000,
        quant_context="b" * 1000,
        common_prefix="c" * 600,
        quant_method_name="q",
    )
    content = msgs[0]["content"]
    # Per spec: per-request block ≤ ~4000 tokens ≈ 16000 chars. Loose bound here.
    assert len(content) < 20000
