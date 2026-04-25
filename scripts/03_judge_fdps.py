"""Phase 3: classify each FDP via Claude Sonnet 4.6 with SHA256 cache."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from kvtrace.config import load_all_configs
from kvtrace.hf_hub.upload import resolve_repo_id, upload_dataset_file
from kvtrace.judge.claude_judge import ClaudeJudge

log = logging.getLogger("judge")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdps_dir", default="outputs/fdps")
    parser.add_argument("--out_dir", default="outputs/judgments")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    _, _, pipeline = load_all_configs(Path("config"))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY is required for Phase 3")
        return 2

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    judge = ClaudeJudge(
        client=client,
        model=pipeline.judge.model,
        cache_dir=pipeline.judge.cache_dir,
        temperature=pipeline.judge.temperature,
        max_tokens=pipeline.judge.max_tokens,
    )

    fdps_dir = Path(args.fdps_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(fdps_dir.glob("*.jsonl")):
        out_file = out_dir / f.name
        with f.open(encoding="utf-8") as fin, out_file.open("w", encoding="utf-8") as fout:
            written = 0
            for line in fin:
                rec = json.loads(line)
                if rec.get("fdp_token_idx") is None:
                    # No divergence; skip entirely rather than forcing a category.
                    continue
                try:
                    res = judge.judge(
                        problem=rec.get("problem", ""),
                        ground_truth=rec.get("ground_truth", ""),
                        baseline_context=rec.get("baseline_context", ""),
                        quant_context=rec.get("quant_context", ""),
                        common_prefix=rec.get("common_prefix", ""),
                        quant_method_name=rec.get("quant_method", ""),
                    )
                except Exception as e:
                    log.warning("judge failed on %s idx=%s: %s", f.name, rec.get("problem_idx"), e)
                    continue
                out_row = {
                    "problem_idx": rec["problem_idx"],
                    "model": rec["model"],
                    "quant_method": rec["quant_method"],
                    "category": res.category,
                    "confidence": res.confidence,
                    "rationale": res.rationale,
                    "affected_span": res.affected_span,
                    "boxed_match": rec.get("boxed_match"),
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                written += 1
        log.info("wrote %d judgments to %s", written, out_file)
        if resolve_repo_id():
            parts = f.stem.split("_", 1)  # e.g. "deepseek-r1-distill-qwen-1.5b_fp8_e5m2" → [model, config]
            model, qkey = parts[0], parts[1] if len(parts) > 1 else "unknown"
            upload_dataset_file(out_file, revision_tag=f"judgments-{model}-{qkey}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
