"""Phase 4: aggregate judgments into failure-signature report."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kvtrace.analysis.report import build_report, write_heatmap
from kvtrace.analysis.signatures import aggregate_counts, row_normalize

log = logging.getLogger("analyze")


def _load_all(judgments_dir: Path) -> list[dict]:
    out: list[dict] = []
    for f in sorted(judgments_dir.glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            out.extend(json.loads(ln) for ln in fh)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments_dir", default="outputs/judgments")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    judgments = _load_all(Path(args.judgments_dir))
    if not judgments:
        log.error("no judgments in %s", args.judgments_dir)
        return 3

    build_report(
        judgments,
        md_path=out_dir / "report.md",
        json_path=out_dir / "report.json",
    )

    # Global heatmap
    methods, counts = aggregate_counts(judgments)
    sigs = row_normalize(counts)
    write_heatmap(sigs, methods, out_dir / "plots" / "signatures_all.png")

    # Per-model heatmaps (if model field is present)
    by_model: dict[str, list[dict]] = {}
    for j in judgments:
        by_model.setdefault(j.get("model", "unknown"), []).append(j)

    for model, rows in by_model.items():
        ms, cs = aggregate_counts(rows)
        write_heatmap(
            row_normalize(cs),
            ms,
            out_dir / "plots" / f"signatures_{model}.png",
        )

    log.info("report: %s", out_dir / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
