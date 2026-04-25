import pytest

from kvtrace.judge.parsing import JudgeParseError, JudgmentResult, parse_judge_response


def test_parse_valid_json():
    r = parse_judge_response(
        '{"category":"A","confidence":0.85,"rationale":"5*4=24 instead of 20","affected_span":"5*4 = 24"}'
    )
    assert isinstance(r, JudgmentResult)
    assert r.category == "A"
    assert r.confidence == 0.85


def test_parse_with_surrounding_noise():
    # Anthropic sometimes wraps JSON in ```json fences.
    r = parse_judge_response(
        "```json\n"
        '{"category":"C","confidence":0.7,"rationale":"switched","affected_span":"new method"}'
        "\n```"
    )
    assert r.category == "C"


def test_parse_invalid_category_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response(
            '{"category":"Z","confidence":0.8,"rationale":"","affected_span":"x"}'
        )


def test_parse_missing_field_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response('{"category":"A","confidence":0.8}')


def test_parse_non_json_raises():
    with pytest.raises(JudgeParseError):
        parse_judge_response("sorry I cannot classify this")


def test_confidence_clamped_to_01():
    r = parse_judge_response(
        '{"category":"A","confidence":1.5,"rationale":"r","affected_span":"s"}'
    )
    assert r.confidence == 1.0

    r = parse_judge_response(
        '{"category":"A","confidence":-0.2,"rationale":"r","affected_span":"s"}'
    )
    assert r.confidence == 0.0
