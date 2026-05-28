"""Phase 10: per-layer ablation — какой слой больше всего портит логиты квантом.

Для каждого из N задач:
  1. Загружаем bf16 reference logits из outputs/.../bf16_tf/<pid>.safetensors
  2. Для каждого слоя L ∈ [0, 28):
     - Делаем TF forward с quant ТОЛЬКО на слое L (остальные — bf16)
     - Сохраняем logits
     - Считаем KL(bf16 || layer_L_only) per-position
  3. Усредняем

Результаты:
  - layer_ablation_<quant>.npz: [N_problems, 28_layers, W_window] KL per (problem, layer, pos)
  - per_layer_kl_<quant>.json: mean KL per layer aggregated
  - plots/layer_ablation_<quant>.png: bar chart layer → mean KL

Стоимость: N_problems × 28 forwards × ~5 sec ≈ 12 мин для N=5.

Usage:
    python scripts/10_layer_ablation.py --n-problems 5 --quant fp8_e4m3
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from kvtrace.capture.attention_hooks import install_capture_hooks
from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.fp8_sim import QUANT_FNS
from kvtrace.capture.storage import load_capture
from kvtrace.capture.window import compute_window

log = logging.getLogger("phase10")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 10: layer ablation")
    p.add_argument("--captures-dir", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b"))
    p.add_argument("--traces-dir", type=Path, default=Path("outputs/traces"))
    p.add_argument("--quant", default="fp8_e4m3",
                   choices=["fp8_e4m3", "fp8_e5m2", "hqq_int4", "hqq_int2"])
    p.add_argument("--n-problems", type=int, default=5)
    p.add_argument("--problems", default=None,
                   help="'0,1,2' for explicit list; overrides --n-problems")
    p.add_argument("--window-pre", type=int, default=150)
    p.add_argument("--window-post", type=int, default=100)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _load_bf16_trace_tokens(traces_dir: Path) -> dict[int, list[int]]:
    import json as _json
    path = traces_dir / "qwen3-1.7b_bf16.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = _json.loads(line)
        out[rec["idx"]] = rec["token_ids"]
    return out


def _kl_per_position(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """KL per position from two [W, vocab] logit arrays."""
    p_a = torch.softmax(log_a.float(), dim=-1)
    p_b = torch.softmax(log_b.float(), dim=-1)
    eps = 1e-12
    return (p_a * (torch.log(p_a + eps) - torch.log(p_b + eps))).sum(dim=-1)


def run_layer_only_forward(
    runner: CaptureRunner, input_token_ids: list[int], window_we: int,
    quant_fn, only_layer: int,
) -> torch.Tensor:
    """Run TF forward with quant active ONLY at one layer. Returns logits [we, vocab]."""
    handle = install_capture_hooks(
        runner._model,
        attention_modules=runner._attention_modules,
        quant_fn=quant_fn,
        quant_only_layers={only_layer},
    )
    try:
        feed_ids = torch.tensor([input_token_ids[:window_we]], dtype=torch.long)
        with torch.no_grad():
            out = runner._model(input_ids=feed_ids, use_cache=True)
        return out.logits[0].detach().clone()  # [we, vocab]
    finally:
        handle.remove()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.captures_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.problems:
        problem_ids = [int(p) for p in args.problems.split(",")]
    else:
        problem_ids = list(range(args.n_problems))

    quant_fn = QUANT_FNS[args.quant]

    log.info("Loading Qwen3-1.7B on CPU...")
    runner = CaptureRunner()
    runner.load_model("Qwen/Qwen3-1.7B")
    n_layers = runner._n_layers

    log.info("Loading bf16 traces...")
    bf16_tokens = _load_bf16_trace_tokens(args.traces_dir)

    log.info("Layer ablation: %d problems × %d layers × forward",
             len(problem_ids), n_layers)
    # [n_problems, n_layers, W] KL per (problem, layer, position-in-window)
    all_kl: list[np.ndarray] = []
    problem_meta: list[dict] = []

    for pid in problem_ids:
        # Get bf16 reference capture for window coords + ref logits
        bf16_cap_path = args.captures_dir / "bf16_tf" / f"{pid}.safetensors"
        if not bf16_cap_path.exists():
            log.warning("Missing bf16 capture for problem %d", pid)
            continue
        bf16_cap = load_capture(bf16_cap_path)
        ws, we, W = bf16_cap.meta["window_start"], bf16_cap.meta["window_end"], bf16_cap.meta["W"]
        fdp = bf16_cap.meta["fdp_token_idx"]
        ref_logits = bf16_cap.logits.float()  # [W, vocab]

        token_ids = bf16_tokens[pid]
        log.info("  problem %d: FDP=%d, window=[%d,%d), W=%d", pid, fdp, ws, we, W)

        per_layer_kl = np.zeros((n_layers, W), dtype=np.float32)
        for L in range(n_layers):
            logits = run_layer_only_forward(runner, token_ids, we, quant_fn, L)
            sliced = logits[ws:we].cpu()  # [W, vocab]
            kl = _kl_per_position(ref_logits, sliced).numpy()  # [W]
            per_layer_kl[L] = kl
            if L % 7 == 0:
                log.info("    layer %2d: mean KL=%.5f (max=%.5f)", L, kl.mean(), kl.max())

        all_kl.append(per_layer_kl)
        problem_meta.append({"problem_id": pid, "fdp": fdp, "ws": ws, "we": we, "W": W})

    runner.unload()

    if not all_kl:
        log.error("No problems processed")
        return 2

    # Pad to max W and stack
    max_w = max(arr.shape[1] for arr in all_kl)
    padded = []
    for arr in all_kl:
        if arr.shape[1] == max_w:
            padded.append(arr)
        else:
            out = np.full((n_layers, max_w), np.nan, dtype=np.float32)
            out[:, :arr.shape[1]] = arr
            padded.append(out)
    stack = np.stack(padded, axis=0)  # [n_problems, n_layers, max_w]

    out_npz = output_dir / f"layer_ablation_{args.quant}.npz"
    np.savez(out_npz, kl=stack, problem_meta=problem_meta)
    log.info("Saved %s, shape %s", out_npz, stack.shape)

    # Per-layer summary
    per_layer_mean = np.nanmean(stack, axis=(0, 2))  # [n_layers]
    per_layer_max = np.nanmax(stack, axis=(0, 2))
    per_layer_at_fdp: list[float] = []
    for arr, meta in zip(all_kl, problem_meta):
        fdp_in_window = meta["fdp"] - meta["ws"]
        if 0 <= fdp_in_window < arr.shape[1]:
            per_layer_at_fdp.append(arr[:, fdp_in_window].tolist())
    if per_layer_at_fdp:
        fdp_arr = np.array(per_layer_at_fdp)  # [n_problems, n_layers]
        per_layer_at_fdp_mean = fdp_arr.mean(axis=0)
    else:
        per_layer_at_fdp_mean = np.zeros(n_layers)

    summary = {
        "quant": args.quant,
        "n_problems": len(all_kl),
        "n_layers": n_layers,
        "per_layer": [
            {
                "layer": int(L),
                "mean_kl": float(per_layer_mean[L]),
                "max_kl": float(per_layer_max[L]),
                "kl_at_fdp_mean": float(per_layer_at_fdp_mean[L]),
            } for L in range(n_layers)
        ],
        "worst_layer_mean": int(np.argmax(per_layer_mean)),
        "worst_layer_at_fdp": int(np.argmax(per_layer_at_fdp_mean)),
    }
    out_json = output_dir / f"per_layer_kl_{args.quant}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log.info("Saved %s", out_json)

    log.info("=== Top 5 layers by mean KL contribution ===")
    ranked = sorted(range(n_layers), key=lambda L: -per_layer_mean[L])
    for rank, L in enumerate(ranked[:5]):
        log.info("  #%d: layer %d → mean KL=%.5f, max=%.5f, @FDP=%.5f",
                 rank+1, L, per_layer_mean[L], per_layer_max[L],
                 per_layer_at_fdp_mean[L])

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not installed, skipping plot")
            return 0
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(n_layers)
        ax1.bar(x, per_layer_mean, color="C0", alpha=0.7, label="mean (window)")
        ax1.bar(x, per_layer_at_fdp_mean, color="C3", alpha=0.7, label="@FDP")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("KL(bf16 || layer-L-only-quant)")
        ax1.set_title(f"Per-layer ablation contribution — {args.quant}\n"
                      f"(n={len(all_kl)} problems)")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(x, per_layer_max, color="C2", alpha=0.7)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("max KL over window")
        ax2.set_title(f"Per-layer MAX KL contribution — {args.quant}")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        out_png = plot_dir / f"layer_ablation_{args.quant}.png"
        plt.savefig(out_png, dpi=100)
        plt.close()
        log.info("Saved %s", out_png)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
