"""Phase 6: capture Q/K/V/logits для Qwen3-1.7B на CPU.

Читает existing artifacts:
  - outputs/traces/<model>_bf16.jsonl       — bf16 reference token streams
  - outputs/fdps/<model>_<quant>.jsonl      — FDP indices per (problem, quant)

Пишет:
  - <output-dir>/<model>/<quant>_<mode>/<problem>.safetensors
  - <output-dir>/<model>/_run_metadata.json
  - <output-dir>/<model>/_skipped.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.manifest import RunManifest, SkipEntry
from kvtrace.capture.storage import save_capture
from kvtrace.capture.window import compute_window

log = logging.getLogger("capture")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_HF_IDS = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6 KV-matrix capture")
    p.add_argument("--model", default="qwen3-1.7b", choices=list(MODEL_HF_IDS))
    p.add_argument("--quants", nargs="+", default=["bf16", "fp8_e4m3", "fp8_e5m2"])
    p.add_argument("--modes", nargs="+", default=["tf", "ar"], choices=["tf", "ar"])
    p.add_argument("--problems", default="all",
                   help="'all' или '0,3,5' для подмножества")
    p.add_argument("--window-pre", type=int, default=150)
    p.add_argument("--window-post", type=int, default=100)
    p.add_argument("--traces-dir", type=Path, default=Path("outputs/traces"))
    p.add_argument("--fdps-dir", type=Path, default=Path("outputs/fdps"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/kv_capture"))
    p.add_argument("--dry-run", action="store_true",
                   help="Распечатать план без запуска forward'ов")
    p.add_argument("--resume", action="store_true",
                   help="Skip captures, чьи safetensors уже на диске")
    p.add_argument("--smoke", action="store_true",
                   help="Shortcut: --problems 0,1 --quants bf16 --modes tf (быстрый CI)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.smoke:
        args.problems = "0,1"
        args.quants = ["bf16"]
        args.modes = ["tf"]

    bf16_trace_path = args.traces_dir / f"{args.model}_bf16.jsonl"
    if not bf16_trace_path.exists():
        log.error("Missing bf16 trace: %s", bf16_trace_path)
        return 2
    bf16_traces = _load_jsonl(bf16_trace_path)

    # FDP: один файл per (model, fp8 quant). bf16 заимствует FDP от fp8_e4m3.
    fdp_by_quant: dict[str, dict[int, int]] = {}
    for q in args.quants:
        if q == "bf16":
            continue
        fdp_path = args.fdps_dir / f"{args.model}_{q}.jsonl"
        if not fdp_path.exists():
            log.error("Missing FDPs for quant=%s: %s", q, fdp_path)
            return 2
        fdp_by_quant[q] = {
            r["problem_idx"]: r["fdp_token_idx"]
            for r in _load_jsonl(fdp_path)
            if r.get("fdp_token_idx") is not None
        }
    # bf16 не имеет собственного FDP (расхождения только относительно квантов),
    # поэтому заимствует FDP-координаты от fp8_e4m3 как референс. Если других
    # квантов в этом ране нет (например --smoke с только bf16), грузим
    # fp8_e4m3 FDP-файл специально для этой цели.
    if "bf16" in args.quants:
        if "fp8_e4m3" in fdp_by_quant:
            fdp_by_quant["bf16"] = fdp_by_quant["fp8_e4m3"]
        elif fdp_by_quant:
            fdp_by_quant["bf16"] = next(iter(fdp_by_quant.values()))
        else:
            # bf16-only ран — подгружаем fp8_e4m3 FDP как координатный референс
            ref_path = args.fdps_dir / f"{args.model}_fp8_e4m3.jsonl"
            if not ref_path.exists():
                log.error("bf16-only run requires fp8_e4m3 FDP file as reference: %s", ref_path)
                return 2
            fdp_by_quant["bf16"] = {
                r["problem_idx"]: r["fdp_token_idx"]
                for r in _load_jsonl(ref_path)
                if r.get("fdp_token_idx") is not None
            }

    if args.problems == "all":
        problem_ids = sorted({t["idx"] for t in bf16_traces})
    else:
        problem_ids = [int(p) for p in args.problems.split(",")]

    out_root = args.output_dir / args.model
    manifest = RunManifest(output_dir=out_root)

    plan: list[tuple[int, str, str]] = []
    for pid in problem_ids:
        for q in args.quants:
            for m in args.modes:
                plan.append((pid, q, m))

    print(f"Total captures planned: {len(plan)}")
    for pid, q, m in plan:
        print(f"  problem={pid} quant={q} mode={m}")

    if args.dry_run:
        return 0

    # Real run: load model once.
    runner = CaptureRunner()
    log.info("Loading %s on CPU...", args.model)
    runner.load_model(MODEL_HF_IDS[args.model])
    manifest.write_metadata(
        model=args.model,
        model_revision_hash=runner._model_revision_hash,
        quants=args.quants,
        modes=args.modes,
        window_pre=args.window_pre,
        window_post=args.window_post,
    )

    trace_by_id = {t["idx"]: t for t in bf16_traces}
    for pid, quant, mode in plan:
        out_path = out_root / f"{quant}_{mode}" / f"{pid}.safetensors"
        if args.resume and out_path.exists():
            log.info("Skipping existing: %s", out_path)
            continue

        trace = trace_by_id.get(pid)
        if trace is None:
            manifest.log_skip(SkipEntry(pid, quant, mode, "trace_not_found"))
            continue
        fdp_idx = fdp_by_quant[quant].get(pid)
        if fdp_idx is None:
            manifest.log_skip(SkipEntry(pid, quant, mode, "no_fdp"))
            continue

        token_ids = trace["token_ids"]
        try:
            window = compute_window(
                fdp_idx=fdp_idx,
                trace_len=len(token_ids),
                pre=args.window_pre,
                post=args.window_post,
            )
        except ValueError as e:
            manifest.log_skip(SkipEntry(pid, quant, mode, f"window_invalid: {e}"))
            continue

        try:
            if mode == "tf":
                cap = runner.run_tf(
                    input_token_ids=token_ids,
                    window=window,
                    quant=quant,
                    problem_id=pid,
                    fdp_token_idx=fdp_idx,
                )
            else:  # ar
                prefix = token_ids[: max(1, fdp_idx - args.window_pre)]
                cap = runner.run_ar(
                    prefix_token_ids=prefix,
                    window=window,
                    quant=quant,
                    problem_id=pid,
                    fdp_token_idx=fdp_idx,
                    # +1 для inclusive верхней границы окна [FDP-pre, FDP+post]:
                    # генерируем pre+post+1 токенов чтобы покрыть позиции
                    # [prefix_len, prefix_len + pre + post + 1) = [ws, we)
                    max_new_tokens=args.window_pre + args.window_post + 1,
                )
        except Exception as e:
            log.exception("Capture failed for problem=%d quant=%s mode=%s", pid, quant, mode)
            manifest.log_skip(SkipEntry(pid, quant, mode, f"runtime_error: {e!r}"))
            continue

        save_capture(cap, out_path)
        log.info("Saved %s", out_path)

    runner.unload()
    return 0


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    sys.exit(main())
