import json

from kvtrace.judge.claude_judge import _prompt_cache_key, JudgeCache


def test_prompt_cache_key_stable():
    k1 = _prompt_cache_key("system", [{"role": "user", "content": "x"}], "claude-sonnet-4-6")
    k2 = _prompt_cache_key("system", [{"role": "user", "content": "x"}], "claude-sonnet-4-6")
    assert k1 == k2
    assert len(k1) == 64


def test_prompt_cache_key_changes_on_input():
    k1 = _prompt_cache_key("sys1", [], "m")
    k2 = _prompt_cache_key("sys2", [], "m")
    assert k1 != k2


def test_judge_cache_roundtrip(tmp_path):
    cache = JudgeCache(tmp_path)
    cache.put("abc123", {"category": "A", "confidence": 0.9, "rationale": "r", "affected_span": "s"})
    assert cache.get("abc123") == {"category": "A", "confidence": 0.9, "rationale": "r", "affected_span": "s"}


def test_judge_cache_miss_returns_none(tmp_path):
    cache = JudgeCache(tmp_path)
    assert cache.get("nope") is None


def test_judge_cache_persists_across_instances(tmp_path):
    c1 = JudgeCache(tmp_path)
    c1.put("k", {"v": 1})
    c2 = JudgeCache(tmp_path)
    assert c2.get("k") == {"v": 1}
