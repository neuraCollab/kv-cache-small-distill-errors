"""Post-run sanity check for the kvtrace pipeline.

Walks `outputs/` and validates every artifact each phase should have
produced. Pairs that are missing in Phase 1 propagate as expected-missing
through Phase 2/3 — those don't trigger errors, but they're surfaced in
the summary so the operator can confirm the gap is intentional (e.g. our
known qwen3+HQQ and 7b+HQQ skips after the transformers 4.55 regression).

Exit codes:
  0 — clean (or warnings only)
  1 — at least one critical artifact is missing or malformed

Run from the repo root:
    python scripts/verify_run.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from kvtrace.config import load_all_configs

OUT = Path("outputs")
TRACES = OUT / "traces"
FDPS = OUT / "fdps"
JUDGMENTS = OUT / "judgments"
REPORT = OUT / "report.md"
PLOTS = OUT / "plots"
FAIL_LOG = Path("logs/run_all_failures.log")

# ANSI: harmless on terminals that don't grok them.
GREEN = "\033[32m"
YEL = "\033[33m"
RED = "\033[31m"
RST = "\033[0m"

errors: list[str] = []
warnings_: list[str] = []


def err(msg: str) -> None:
    print(f"{RED}✗ {msg}{RST}")
    errors.append(msg)


def warn(msg: str) -> None:
    print(f"{YEL}! {msg}{RST}")
    warnings_.append(msg)


def ok(msg: str) -> None:
    print(f"{GREEN}✓ {msg}{RST}")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} invalid JSON: {e}") from e
    return rows


# --------------------------------------------------------------------------
# Phase 1: traces
# --------------------------------------------------------------------------

def check_trace(model: str, qkey: str, expected_problems: int) -> bool:
    """Return True iff the trace file is on disk (regardless of health)."""
    path = TRACES / f"{model}_{qkey}.jsonl"
    if not path.exists():
        return False
    try:
        rows = load_jsonl(path)
    except ValueError as e:
        err(str(e))
        return True
    n = len(rows)
    if n == 0:
        err(f"{path.name}: empty file")
        return True
    if n != expected_problems:
        warn(f"{path.name}: {n} records, expected {expected_problems}")
    # required fields (per GenerationResult.to_dict + 01_generate_traces.py)
    sample = rows[0]
    needed = ("idx", "raw_output", "token_ids", "finish_reason",
              "num_prompt_tokens", "num_generated_tokens",
              "problem", "ground_truth")
    missing = [k for k in needed if k not in sample]
    if missing:
        err(f"{path.name}: row 0 missing fields: {missing}")
    fr = Counter(r.get("finish_reason") for r in rows)
    length_rate = fr.get("length", 0) / n
    if length_rate > 0.30:
        warn(f"{path.name}: {length_rate:.0%} of rows hit max_tokens "
             f"(may indicate runaway generation)")
    boxed = sum(1 for r in rows if r.get("boxed_answer"))
    if boxed / n < 0.50:
        warn(f"{path.name}: only {boxed}/{n} have boxed_answer "
             f"(model often didn't finish reasoning)")
    avg_gen = sum(r.get("num_generated_tokens", 0) for r in rows) / n
    ok(f"{path.name}: {n} rows, finish={dict(fr)}, "
       f"boxed={boxed}/{n}, avg_gen_tokens={avg_gen:.0f}")
    return True


# --------------------------------------------------------------------------
# Phase 2: FDPs
# --------------------------------------------------------------------------

def check_fdp(model: str, qkey: str) -> int | None:
    """Return number of records with fdp_token_idx != None, or None if missing."""
    path = FDPS / f"{model}_{qkey}.jsonl"
    if not path.exists():
        err(f"{path.name}: missing — Phase 2 didn't produce it though both "
            f"traces (bf16, {qkey}) exist")
        return None
    try:
        rows = load_jsonl(path)
    except ValueError as e:
        err(str(e))
        return None
    if not rows:
        warn(f"{path.name}: empty (no problems matched between baseline and quant)")
        return 0
    diverged = sum(1 for r in rows if r.get("fdp_token_idx") is not None)
    boxed_match = sum(1 for r in rows if r.get("boxed_match") is True)
    div_rate = diverged / len(rows)
    bm_rate = boxed_match / len(rows)
    ok(f"{path.name}: {len(rows)} pairs, "
       f"diverged={diverged} ({div_rate:.0%}), "
       f"boxed_match={boxed_match} ({bm_rate:.0%})")
    return diverged


# --------------------------------------------------------------------------
# Phase 3: judgments
# --------------------------------------------------------------------------

VALID_CATEGORIES = {"A", "B", "C", "D", "E", "F"}


def check_judgment(model: str, qkey: str, expected_diverged: int | None) -> None:
    path = JUDGMENTS / f"{model}_{qkey}.jsonl"
    if expected_diverged is None:
        return  # FDP file was missing or unreadable; already reported
    if expected_diverged == 0:
        if path.exists():
            try:
                rows = load_jsonl(path)
            except ValueError as e:
                err(str(e))
                return
            if rows:
                warn(f"{path.name}: {len(rows)} judgments though FDP file "
                     f"had 0 divergences — stale file?")
            else:
                ok(f"{path.name}: 0 records (no divergence to judge)")
        return
    if not path.exists():
        err(f"{path.name}: missing — expected ~{expected_diverged} judgments")
        return
    try:
        rows = load_jsonl(path)
    except ValueError as e:
        err(str(e))
        return
    n = len(rows)
    coverage = n / expected_diverged
    if coverage < 0.90:
        warn(f"{path.name}: {n}/{expected_diverged} divergences judged "
             f"({coverage:.0%}) — judge likely errored on some calls")
    elif coverage > 1.05:
        warn(f"{path.name}: {n}/{expected_diverged} — more judgments than "
             f"divergences; cache may have stale rows")
    cats = Counter(r.get("category") for r in rows)
    bad = [c for c in cats if c not in VALID_CATEGORIES]
    if bad:
        warn(f"{path.name}: out-of-taxonomy categories: {bad}")
    confs = [r.get("confidence", 0) for r in rows
             if isinstance(r.get("confidence"), (int, float))]
    avg_conf = sum(confs) / max(len(confs), 1)
    if avg_conf < 0.5:
        warn(f"{path.name}: avg_confidence={avg_conf:.2f} — judge is unsure")
    ok(f"{path.name}: {n} records, cats={dict(cats)}, avg_conf={avg_conf:.2f}")


# --------------------------------------------------------------------------
# Phase 4: report
# --------------------------------------------------------------------------

def check_report() -> None:
    if not REPORT.exists():
        err("outputs/report.md missing — Phase 4 didn't run or crashed")
        return
    size = REPORT.stat().st_size
    if size < 500:
        warn(f"report.md is {size} B — likely empty/template; check Phase 4 logs")
    else:
        ok(f"report.md: {size} B")

    if PLOTS.exists():
        plots = list(PLOTS.glob("*.png")) + list(PLOTS.glob("*.svg"))
        if plots:
            ok(f"plots/: {len(plots)} image(s)")
        else:
            warn("plots/ exists but contains no png/svg")
    # Plots aren't strictly required (depending on what 04_analyze produces)
    # so absence is fine.


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    if not OUT.exists():
        err("outputs/ does not exist — pipeline never ran")
        return 1

    models, quants, pipeline = load_all_configs(Path("config"))
    expected = pipeline.dataset.aime_24_count + pipeline.dataset.math_500_count

    print(f"=== Expected problems per (model, quant): {expected} ===")
    print(f"=== Models: {list(models)} ===")
    print(f"=== Quants: {list(quants)} ===\n")

    # Phase 1
    print("--- Phase 1: traces ---")
    trace_present: dict[tuple[str, str], bool] = {}
    for model in models:
        for qkey in quants:
            present = check_trace(model, qkey, expected)
            trace_present[(model, qkey)] = present
            if not present:
                if qkey == "bf16":
                    err(f"{model}_bf16.jsonl missing — no baseline, "
                        f"{model} cannot be analyzed")
                else:
                    warn(f"{model}_{qkey}.jsonl missing — pair excluded "
                         f"from Phase 2/3")
    print()

    # Phase 2
    print("--- Phase 2: FDPs ---")
    diverged: dict[tuple[str, str], int | None] = {}
    for model in models:
        if not trace_present.get((model, "bf16"), False):
            continue
        for qkey in quants:
            if qkey == "bf16":
                continue
            if not trace_present.get((model, qkey), False):
                continue
            diverged[(model, qkey)] = check_fdp(model, qkey)
    print()

    # Phase 3
    print("--- Phase 3: judgments ---")
    for (model, qkey), d in diverged.items():
        check_judgment(model, qkey, d)
    print()

    # Phase 4
    print("--- Phase 4: report ---")
    check_report()
    print()

    # Failure log review
    print("--- Failure log ---")
    if FAIL_LOG.exists() and FAIL_LOG.stat().st_size > 0:
        warn(f"{FAIL_LOG} non-empty:")
        print(FAIL_LOG.read_text(encoding="utf-8", errors="replace"))
    else:
        ok("logs/run_all_failures.log empty or absent")
    print()

    # Summary
    print("=== Summary ===")
    print(f"  warnings: {len(warnings_)}")
    print(f"  errors  : {len(errors)}")
    if errors:
        print(f"\n{RED}Critical:{RST}")
        for e in errors:
            print(f"  - {e}")
        return 1
    if warnings_:
        print(f"\n{YEL}Warnings (non-blocking):{RST}")
        for w in warnings_:
            print(f"  - {w}")
    print(f"\n{GREEN}OK — pipeline produced expected artifacts.{RST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
