"""Anthropic Claude client with on-disk SHA256 prompt cache."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from kvtrace.judge.parsing import JudgeParseError, JudgmentResult, parse_judge_response
from kvtrace.judge.prompt import PROMPT_VERSION, build_judge_messages, build_system_block
from kvtrace.judge.taxonomy import TAXONOMY_VERSION

log = logging.getLogger(__name__)


def _prompt_cache_key(system: str, messages: list[dict], model: str) -> str:
    """SHA256 of the normalized (system, messages, model, versions) tuple."""
    payload = json.dumps(
        {
            "system": system,
            "messages": messages,
            "model": model,
            "prompt_v": PROMPT_VERSION,
            "taxonomy_v": TAXONOMY_VERSION,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeCache:
    """Trivial key→JSON on-disk cache."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def put(self, key: str, value: dict) -> None:
        self._path(key).write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )


class ClaudeJudge:
    def __init__(
        self,
        client: Any,
        model: str,
        cache_dir: str | Path,
        temperature: float = 0.0,
        max_tokens: int = 400,
        max_parse_retries: int = 2,
    ) -> None:
        self.client = client
        self.model = model
        self.cache = JudgeCache(cache_dir)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_parse_retries = max_parse_retries

    def judge(
        self,
        *,
        problem: str,
        ground_truth: str,
        baseline_context: str,
        quant_context: str,
        common_prefix: str,
        quant_method_name: str,
    ) -> JudgmentResult:
        system_blocks = build_system_block()
        messages = build_judge_messages(
            problem=problem,
            ground_truth=ground_truth,
            baseline_context=baseline_context,
            quant_context=quant_context,
            common_prefix=common_prefix,
            quant_method_name=quant_method_name,
        )
        system_flat = "\n".join(b["text"] for b in system_blocks)
        key = _prompt_cache_key(system_flat, messages, self.model)

        cached = self.cache.get(key)
        if cached is not None:
            return JudgmentResult(**cached)

        last_err: Exception | None = None
        for attempt in range(self.max_parse_retries + 1):
            resp = self.client.messages.create(
                model=self.model,
                system=system_blocks,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = resp.content[0].text
            try:
                result = parse_judge_response(text)
                self.cache.put(key, result.model_dump())
                return result
            except JudgeParseError as e:
                last_err = e
                log.warning("judge parse error attempt %d: %s", attempt, e)
                continue

        raise last_err or JudgeParseError("exhausted retries")
