"""Phase 26: тест обученной CNN на 5 свежих задачах.

Pipeline:
  1. Тренируем CNN на ВСЕХ 80 capture'ах Qwen3-1.7B (full train, не CV)
  2. Для 5 новых MATH-500 задач (idx 80-84):
     a) Run bf16 greedy generation -> baseline tokens
     b) Run fp8_e4m3 greedy generation -> quant tokens (с hooks)
     c) Найти true FDP = первый расходящийся токен
     d) TF mode forward: feed [prompt + baseline_tokens[:FDP+100]],
        capture K_pre per layer на window [FDP-150 : FDP-25]
     e) Извлечь raw matrix features [28*8, 128]
     f) Predict FDP через CNN
  3. Сравнить predicted vs true FDP.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
p23 = import_module("23_fdp_predictor_extended")

from kvtrace.capture.attention_hooks import install_capture_hooks
from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.capture.fp8_sim import QUANT_FNS
from kvtrace.dataset_loader import load_math_dataset

log = logging.getLogger("phase26")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_USER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
N_LAYERS = 28
N_KV_HEADS = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--training-captures", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b/fp8_e4m3_tf"))
    p.add_argument("--training-fdp", type=Path,
                   default=Path("outputs/fdps/qwen3-1.7b_fp8_e4m3.jsonl"))
    p.add_argument("--n-new", type=int, default=5)
    p.add_argument("--max-gen", type=int, default=1200,
                   help="max tokens для search'a FDP")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b/analysis/cnn_new_problems_test.json"))
    return p.parse_args()


def _build_prompt_tokens(tokenizer, problem_text: str) -> list[int]:
    messages = [{"role": "user",
                 "content": f"{problem_text}\n\n{DEFAULT_USER_INSTRUCTION}"}]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=True, return_tensors="pt",
    )[0].tolist()
    return ids


def _greedy_generate(runner, prompt_tokens, max_new_tokens, quant_name):
    """Greedy AR generation with given quant. Returns generated tokens."""
    quant_fn = QUANT_FNS[quant_name]
    handle = install_capture_hooks(
        runner._model, attention_modules=runner._attention_modules,
        quant_fn=quant_fn,
    )
    try:
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        with torch.no_grad():
            out = runner._model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=runner._tokenizer.eos_token_id,
            )
        gen = out[0].tolist()[len(prompt_tokens):]
        return gen
    finally:
        handle.remove()


def _capture_k_window(runner, prompt_tokens, baseline_tokens, fdp_idx,
                      window_pre=150, window_post=100) -> list[torch.Tensor] | None:
    """TF mode forward: feed [prompt + baseline_tokens[:fdp+window_post]],
    capture K_pre at window [fdp-window_pre : fdp+window_post].

    Returns list of K_pre per layer at shape [window_size, n_kv_heads, head_dim].
    Если FDP < window_pre + len(prompt), window обрезается.
    """
    # Calculate absolute positions in full sequence
    abs_fdp = len(prompt_tokens) + fdp_idx
    abs_window_start = abs_fdp - window_pre
    abs_window_end = abs_fdp + window_post
    # Feed full sequence up to window_end through model
    feed_len = abs_window_end
    if feed_len > len(prompt_tokens) + len(baseline_tokens):
        log.warning(f"   window_end={abs_window_end} > available tokens {len(prompt_tokens) + len(baseline_tokens)}")
        return None
    feed_tokens = prompt_tokens + baseline_tokens[:feed_len - len(prompt_tokens)]
    # Run TF forward with hooks (no quant — we just want K_pre captured)
    handle = install_capture_hooks(
        runner._model, attention_modules=runner._attention_modules,
        quant_fn=QUANT_FNS["fp8_e4m3"],
    )
    try:
        input_ids = torch.tensor([feed_tokens], dtype=torch.long)
        with torch.no_grad():
            runner._model(input_ids)
        # handle.k_pre — список из (n_layers) tensors, каждый [1, h, seq_full, d]
        # Slice к window
        ws = max(0, abs_window_start)
        we = min(abs_window_end, feed_len)
        K_per_layer = []
        for L in range(N_LAYERS):
            K_full = handle.k_pre[L]  # [1, h, full_seq, d]
            if K_full.shape[2] < we:
                return None
            K_win = K_full[0, :, ws:we, :].permute(1, 0, 2).clone()  # [win_seq, h, d]
            K_per_layer.append(K_win)
        return K_per_layer
    finally:
        handle.remove()


def find_fdp(baseline_tokens: list[int], quant_tokens: list[int]) -> int:
    """First index где токены различаются. Returns len if no divergence в окне."""
    n = min(len(baseline_tokens), len(quant_tokens))
    for i in range(n):
        if baseline_tokens[i] != quant_tokens[i]:
            return i
    return n  # no divergence within both lengths


def train_cnn_full(captures_dir: Path, fdp_file: Path, n_lh: int, head_dim: int) -> nn.Module:
    """Train CNN on full 80-problem dataset."""
    fdps = p23.load_fdps(fdp_file)
    X_raw, y = [], []
    for cf in sorted(captures_dir.glob("*.safetensors"), key=lambda p: int(p.stem)):
        pid = int(cf.stem)
        if pid not in fdps:
            continue
        K = p23.load_K_decode_window(cf, half_only=True)
        if len(K) != N_LAYERS:
            continue
        X_raw.append(p23.kper_to_raw_matrix(K))
        y.append(fdps[pid])
    X_raw = np.array(X_raw, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    log.info(f"  Training CNN on n={len(y)} problems, X_raw={X_raw.shape}")
    mean = X_raw.mean(axis=(0, 2), keepdims=True)
    std = X_raw.std(axis=(0, 2), keepdims=True) + 1e-8
    Xn = (X_raw - mean) / std
    # Train
    model = p23.K1DCNN(n_lh, head_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n = len(y)
    for ep in range(400):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 8):
            idx = perm[i:i + 8]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        train_pred = model(Xt).numpy()
    from sklearn.metrics import r2_score, mean_absolute_error
    log.info(f"  CNN training done. Train R²={r2_score(y, train_pred):.3f}, "
             f"MAE={mean_absolute_error(y, train_pred):.1f}")
    return model, mean, std


def main() -> int:
    args = parse_args()

    log.info("Step 1: Train CNN on 80 existing problems")
    model, train_mean, train_std = train_cnn_full(
        args.training_captures, args.training_fdp,
        n_lh=N_LAYERS * N_KV_HEADS, head_dim=128,
    )

    log.info(f"Step 2: load {args.n_new} fresh MATH-500 problems (indices 80-{80 + args.n_new - 1})")
    problems = load_math_dataset("math-500", num_samples=80 + args.n_new, shuffle=False)
    new_probs = problems[80:80 + args.n_new]
    log.info(f"  Got {len(new_probs)} fresh problems")

    log.info("Step 3: Load Qwen3-1.7B")
    runner = CaptureRunner()
    runner.load_model(args.model)

    log.info("Step 4: For each new problem — find FDP, capture K, predict FDP")
    results = []
    for i, prob in enumerate(new_probs):
        prompt_tokens = _build_prompt_tokens(runner._tokenizer, prob.problem)
        log.info(f"  Problem {i + 1}/{args.n_new} (idx={prob.idx}): prompt_len={len(prompt_tokens)}")

        # bf16 baseline generation
        log.info(f"    Generating bf16 baseline (max {args.max_gen})...")
        bf_tokens = _greedy_generate(runner, prompt_tokens, args.max_gen, "bf16")
        log.info(f"      Got {len(bf_tokens)} tokens")

        # fp8_e4m3 generation
        log.info(f"    Generating fp8_e4m3 (max {args.max_gen})...")
        fp_tokens = _greedy_generate(runner, prompt_tokens, args.max_gen, "fp8_e4m3")
        log.info(f"      Got {len(fp_tokens)} tokens")

        # FDP
        true_fdp = find_fdp(bf_tokens, fp_tokens)
        log.info(f"    True FDP (relative to start of gen): {true_fdp}")
        # NB: paper's fdp_token_idx is absolute (= prompt_len + relative_fdp - prompt_len + ...)
        # Actually paper uses fdp_token_idx as fdp relative to baseline_response token stream.
        # In our training data fdp_token_idx ranges 1-1161. Let me match.
        # Looking at FDP files: fdp_token_idx for problem 0 = 1075, and prompt was ~100. So it's
        # POSITION IN GENERATION (relative), not absolute. Good — true_fdp matches semantically.

        if true_fdp >= args.max_gen:
            log.warning(f"    No divergence in {args.max_gen} tokens — skipping (need longer gen)")
            results.append({
                "problem_idx": int(prob.idx),
                "prompt_len": len(prompt_tokens),
                "true_fdp": int(true_fdp),
                "predicted_fdp": None,
                "status": "no_divergence_in_window",
            })
            continue

        # Capture K at window [FDP-150 : FDP-25]
        log.info(f"    Capturing K window centered on FDP={true_fdp}...")
        K = _capture_k_window(runner, prompt_tokens, bf_tokens, true_fdp,
                              window_pre=150, window_post=100)
        if K is None:
            log.warning(f"    Window capture failed for problem {prob.idx}")
            results.append({
                "problem_idx": int(prob.idx),
                "prompt_len": len(prompt_tokens),
                "true_fdp": int(true_fdp),
                "predicted_fdp": None,
                "status": "window_truncated",
            })
            continue

        # Truncate to first 125 positions (matches training)
        K_half = [Kl[:125] for Kl in K]
        if any(Kl.shape[0] < 30 for Kl in K_half):
            log.warning(f"    Window too short for problem {prob.idx}")
            results.append({
                "problem_idx": int(prob.idx),
                "prompt_len": len(prompt_tokens),
                "true_fdp": int(true_fdp),
                "predicted_fdp": None,
                "status": "window_too_short",
            })
            continue

        # Pad/truncate всех K_half к 125
        K_padded = []
        for Kl in K_half:
            if Kl.shape[0] < 125:
                pad_len = 125 - Kl.shape[0]
                Kl = torch.cat([Kl, torch.zeros(pad_len, *Kl.shape[1:], dtype=Kl.dtype)], dim=0)
            K_padded.append(Kl[:125])

        X_raw = p23.kper_to_raw_matrix(K_padded)  # [224, 128]
        Xn = (X_raw - train_mean[0]) / train_std[0]  # apply training stats
        Xt = torch.tensor(Xn[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            pred = float(model(Xt).numpy()[0])
        log.info(f"    Predicted FDP: {pred:.1f}, True FDP: {true_fdp}, Error: {pred - true_fdp:+.1f}")
        results.append({
            "problem_idx": int(prob.idx),
            "prompt_len": len(prompt_tokens),
            "true_fdp": int(true_fdp),
            "predicted_fdp": float(pred),
            "abs_error": abs(float(pred) - true_fdp),
            "status": "ok",
        })

    runner.unload()

    # Aggregate
    ok = [r for r in results if r["status"] == "ok"]
    out = {
        "n_new": len(new_probs),
        "n_evaluated": len(ok),
        "per_problem": results,
    }
    if ok:
        true_arr = np.array([r["true_fdp"] for r in ok])
        pred_arr = np.array([r["predicted_fdp"] for r in ok])
        from sklearn.metrics import mean_absolute_error, r2_score
        from scipy.stats import spearmanr
        mae = float(mean_absolute_error(true_arr, pred_arr))
        r2 = float(r2_score(true_arr, pred_arr)) if len(ok) >= 2 else None
        rho = float(spearmanr(true_arr, pred_arr)[0]) if len(ok) >= 2 else None
        out["MAE"] = mae
        out["R2"] = r2
        out["Spearman"] = rho
        log.info("=" * 60)
        log.info(f"5 new problems summary:")
        for r in results:
            if r["status"] == "ok":
                log.info(f"  Problem {r['problem_idx']}: true={r['true_fdp']}, pred={r['predicted_fdp']:.0f}, "
                         f"err={r['predicted_fdp'] - r['true_fdp']:+.0f}")
            else:
                log.info(f"  Problem {r['problem_idx']}: {r['status']}")
        log.info(f"  MAE = {mae:.1f}, R² = {r2}, Spearman = {rho}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    log.info(f"Saved {args.output}")

    # Plot
    if ok:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(true_arr, pred_arr, s=180, color="tab:blue",
                       edgecolors="black", linewidths=1, zorder=3, label="5 fresh problems")
            for r in ok:
                ax.annotate(f"#{r['problem_idx']}",
                            (r["true_fdp"], r["predicted_fdp"]),
                            xytext=(7, 7), textcoords="offset points", fontsize=9)
            mn = min(true_arr.min(), pred_arr.min()) * 0.9
            mx = max(true_arr.max(), pred_arr.max()) * 1.1
            ax.plot([mn, mx], [mn, mx], "r--", alpha=0.5, label="y=x (perfect)")
            ax.set_xlabel("True FDP token index")
            ax.set_ylabel("Predicted FDP token index")
            ax.set_title(f"CNN test on 5 NEW problems (idx 80-{80 + args.n_new - 1}, MATH-500)\n"
                          f"MAE={mae:.0f}, R²={r2:.3f}, ρ={rho:.3f}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plot_path = args.output.parent / "plots" / "cnn_new_problems_test.png"
            plot_path.parent.mkdir(exist_ok=True)
            plt.savefig(plot_path, dpi=130)
            log.info(f"Saved plot {plot_path}")
        except ImportError:
            pass

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
