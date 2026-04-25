"""Phase 1: generate traces for a (model, quant_config) pair.

Reads:
  - config/models.yaml, quant_methods.yaml, pipeline.yaml
  - HuggingFace dataset (cached)

Writes:
  - outputs/traces/{model}_{config}.jsonl
  - optional: HF dataset revision traces-{model}-{config}

Idempotent: skips if HF revision already exists AND --resume is given.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kvtrace.config import load_all_configs
from kvtrace.dataset_loader import load_study_mix
from kvtrace.generators.base import Generator
from kvtrace.generators.hf_gen import HFGenerator
from kvtrace.generators.vllm_gen import VLLMGenerator
from kvtrace.hf_hub.upload import resolve_repo_id, upload_dataset_file

log = logging.getLogger("generate")


def make_generator(engine: str, pipeline) -> Generator:
    if engine == "vllm":
        return VLLMGenerator(
            seed=pipeline.seed,
            sampling_temperature=pipeline.sampling.temperature,
            sampling_top_p=pipeline.sampling.top_p,
            sampling_max_tokens=pipeline.sampling.max_tokens,
        )
    if engine == "hf":
        return HFGenerator(
            sampling_temperature=pipeline.sampling.temperature,
            sampling_top_p=pipeline.sampling.top_p,
            sampling_max_tokens=pipeline.sampling.max_tokens,
        )
    raise ValueError(f"unknown engine {engine!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="key in config/models.yaml")
    parser.add_argument("--config", required=True, help="key in config/quant_methods.yaml")
    parser.add_argument("--output_dir", default="outputs/traces")
    parser.add_argument("--resume", action="store_true", help="skip if HF revision exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    models, quants, pipeline = load_all_configs(Path("config"))

    if args.model not in models:
        log.error("unknown model %r; available: %s", args.model, sorted(models))
        return 2
    if args.config not in quants:
        log.error("unknown config %r; available: %s", args.config, sorted(quants))
        return 2

    mcfg = models[args.model]
    qcfg = quants[args.config]

    # Per-model max_model_len overrides (see QuantCfg.overrides).
    override = qcfg.overrides.get("max_model_len_override", {}).get(args.model)
    if override is not None:
        log.info("override max_model_len %s -> %d for this run", mcfg.max_model_len, override)
        mcfg = mcfg.model_copy(update={"max_model_len": override})

    revision_tag = f"traces-{args.model}-{args.config}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.model}_{args.config}.jsonl"

    if args.resume and out_file.exists():
        log.info("resume: %s already present, skipping generation", out_file)
        return 0

    problems = load_study_mix(
        aime_n=pipeline.dataset.aime_24_count,
        math_n=pipeline.dataset.math_500_count,
    )
    log.info("loaded %d problems", len(problems))

    gen = make_generator(qcfg.engine, pipeline)
    try:
        gen.load(mcfg, qcfg)
        results = gen.generate(problems)
    finally:
        gen.unload()

    with out_file.open("w", encoding="utf-8") as f:
        for p, r in zip(problems, results):
            row = r.to_dict()
            row.update({
                "problem": p.problem,
                "ground_truth": p.answer,
                "source": p.source,
                "model": args.model,
                "quant_method": args.config,
                "seed": pipeline.seed,
            })
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("wrote %d records to %s", len(results), out_file)

    if resolve_repo_id():
        upload_dataset_file(out_file, revision_tag=revision_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
