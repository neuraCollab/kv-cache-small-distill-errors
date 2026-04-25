"""Phase 2: compute FDP between baseline (bf16) and every quant config.

Reads:
  outputs/traces/{model}_bf16.jsonl
  outputs/traces/{model}_{Q}.jsonl  for Q in {fp8_e5m2, fp8_e4m3, hqq_int4, hqq_int2}

Writes:
  outputs/fdps/{model}_{Q}.jsonl

No GPU required. MiniLM embedder is loaded lazily for the re-sync check.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kvtrace.config import load_all_configs
from kvtrace.fdp.finder import FDPParams, find_fdp
from kvtrace.fdp.semantic_resync import ReSyncChecker
from kvtrace.hf_hub.upload import resolve_repo_id, upload_dataset_file

log = logging.getLogger("fdp")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--traces_dir", default="outputs/traces")
    parser.add_argument("--out_dir", default="outputs/fdps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    models, quants, pipeline = load_all_configs(Path("config"))

    if args.model not in models:
        log.error("unknown model %r", args.model)
        return 2

    baseline_path = Path(args.traces_dir) / f"{args.model}_bf16.jsonl"
    if not baseline_path.exists():
        log.error("baseline missing: %s (run Phase 1 with --config bf16 first)", baseline_path)
        return 3

    baseline = _load_jsonl(baseline_path)
    baseline_by_idx = {r["idx"]: r for r in baseline}

    # Use the model's tokenizer for decoding token windows.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        models[args.model].hf_id, trust_remote_code=True
    )

    checker = ReSyncChecker(
        model_name=pipeline.fdp.embed_model,
        threshold=pipeline.fdp.cosmetic_cosine_threshold,
    )
    params = FDPParams(
        context_window=pipeline.fdp.context_window,
        resync_lookahead=pipeline.fdp.resync_lookahead,
        max_cosmetic_skips=pipeline.fdp.max_cosmetic_skips,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quant_keys = [k for k in quants if k != "bf16"]
    for qkey in quant_keys:
        qpath = Path(args.traces_dir) / f"{args.model}_{qkey}.jsonl"
        if not qpath.exists():
            log.warning("missing quant trace %s; skipping", qpath)
            continue

        quant_rows = _load_jsonl(qpath)
        out_file = out_dir / f"{args.model}_{qkey}.jsonl"
        n = 0
        with out_file.open("w", encoding="utf-8") as f:
            for qrow in quant_rows:
                brow = baseline_by_idx.get(qrow["idx"])
                if brow is None:
                    continue
                rec = find_fdp(
                    baseline_tokens=brow["token_ids"],
                    quant_tokens=qrow["token_ids"],
                    baseline_text=brow["raw_output"],
                    quant_text=qrow["raw_output"],
                    tokenizer=tokenizer,
                    resync_checker=checker,
                    params=params,
                    baseline_truncated=(brow.get("finish_reason") == "length"),
                    quant_truncated=(qrow.get("finish_reason") == "length"),
                    baseline_boxed=brow.get("boxed_answer"),
                    quant_boxed=qrow.get("boxed_answer"),
                    ground_truth=brow.get("ground_truth"),
                )
                out = {
                    "problem_idx": qrow["idx"],
                    "model": args.model,
                    "quant_method": qkey,
                    "source": qrow.get("source"),
                    "problem": qrow.get("problem"),
                    "ground_truth": qrow.get("ground_truth"),
                    "fdp_token_idx": rec.fdp_token_idx,
                    "cosmetic_skipped": rec.cosmetic_skipped,
                    "baseline_context": rec.baseline_context,
                    "quant_context": rec.quant_context,
                    "common_prefix": rec.common_prefix,
                    "boxed_match": rec.boxed_match,
                    "baseline_truncated": rec.baseline_truncated,
                    "quant_truncated": rec.quant_truncated,
                    "baseline_boxed": brow.get("boxed_answer"),
                    "quant_boxed": qrow.get("boxed_answer"),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                n += 1
        log.info("wrote %d FDP records to %s", n, out_file)
        if resolve_repo_id():
            upload_dataset_file(out_file, revision_tag=f"fdps-{args.model}-{qkey}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
