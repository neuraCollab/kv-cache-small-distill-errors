from unittest.mock import MagicMock

import pytest

from kvtrace.judge.claude_judge import ClaudeJudge


def _fake_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.usage = MagicMock(
        cache_creation_input_tokens=100,
        cache_read_input_tokens=0,
        input_tokens=50,
        output_tokens=30,
    )
    return resp


def test_claude_judge_happy_path(tmp_path):
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_response(
        '{"category":"A","confidence":0.9,"rationale":"arithmetic","affected_span":"5*4=24"}'
    )

    judge = ClaudeJudge(
        client=fake_client,
        model="claude-sonnet-4-6",
        cache_dir=tmp_path,
    )
    result = judge.judge(
        problem="p",
        ground_truth="20",
        baseline_context="5*4=20",
        quant_context="5*4=24",
        common_prefix="...",
        quant_method_name="fp8_e5m2",
    )
    assert result.category == "A"
    fake_client.messages.create.assert_called_once()


def test_claude_judge_uses_cache_on_second_call(tmp_path):
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_response(
        '{"category":"B","confidence":0.8,"rationale":"r","affected_span":"s"}'
    )

    judge = ClaudeJudge(client=fake_client, model="m", cache_dir=tmp_path)
    args = dict(
        problem="p", ground_truth="g",
        baseline_context="b", quant_context="q",
        common_prefix="c", quant_method_name="fp8_e5m2",
    )
    r1 = judge.judge(**args)
    r2 = judge.judge(**args)
    assert r1.category == r2.category
    # Second call must NOT hit the API.
    assert fake_client.messages.create.call_count == 1


def test_claude_judge_retries_parse_error(tmp_path):
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        _fake_response("not json"),
        _fake_response('{"category":"A","confidence":0.9,"rationale":"r","affected_span":"s"}'),
    ]
    judge = ClaudeJudge(client=fake_client, model="m", cache_dir=tmp_path, max_parse_retries=1)
    result = judge.judge(
        problem="p", ground_truth="g",
        baseline_context="b", quant_context="q",
        common_prefix="c", quant_method_name="fp8_e5m2",
    )
    assert result.category == "A"
    assert fake_client.messages.create.call_count == 2


def test_claude_judge_gives_up_after_retries(tmp_path):
    from kvtrace.judge.parsing import JudgeParseError
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_response("never json")
    judge = ClaudeJudge(client=fake_client, model="m", cache_dir=tmp_path, max_parse_retries=1)
    with pytest.raises(JudgeParseError):
        judge.judge(
            problem="p", ground_truth="g",
            baseline_context="b", quant_context="q",
            common_prefix="c", quant_method_name="fp8_e5m2",
        )
