"""Phase 27: bucket-classifier для FDP с адаптивной сеткой.

Идея: вместо точной регрессии FDP-токена — classifier на K бакетов,
где границы adaptive по training quantiles.

Buckets для Qwen3-1.7B (по training 80 FDPs):
  - quartile-based: 4 класса {very_early, early, mid, late}
  - границы: Q25, Q50 (median), Q75

Architecture: 1D-CNN как в Phase 23, но output K logits + CrossEntropyLoss.

Тест: 6 свежих задач (81-86) end-to-end:
  1. Train classifier на 80 capture'ах
  2. Для каждой fresh задачи — generate, find FDP, capture K
  3. Predict bucket + confidence
  4. Сравнить с true bucket
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
p23 = import_module("23_fdp_predictor_extended")
p26 = import_module("26_cnn_test_new_problems")

from kvtrace.capture.cpu_runner import CaptureRunner
from kvtrace.dataset_loader import load_math_dataset

log = logging.getLogger("phase27")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

N_LAYERS = 28
N_KV_HEADS = 8
N_BUCKETS = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--training-captures", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b/fp8_e4m3_tf"))
    p.add_argument("--training-fdp", type=Path,
                   default=Path("outputs/fdps/qwen3-1.7b_fp8_e4m3.jsonl"))
    p.add_argument("--start-idx", type=int, default=81)
    p.add_argument("--end-idx", type=int, default=86, help="inclusive")
    p.add_argument("--max-gen", type=int, default=400)
    p.add_argument("--n-buckets", type=int, default=4)
    p.add_argument("--output", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b/analysis/cnn_buckets_test.json"))
    return p.parse_args()


# ============================================================
# Bucket utilities
# ============================================================
def compute_bucket_edges(y_train: np.ndarray, n_buckets: int) -> np.ndarray:
    """Adaptive quantile-based edges. Returns sorted thresholds (n_buckets-1)."""
    quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]  # exclude 0 and 1
    edges = np.quantile(y_train, quantiles)
    return edges


def assign_bucket(y: float, edges: np.ndarray) -> int:
    """y → bucket [0, n_buckets-1]"""
    return int(np.searchsorted(edges, y, side="right"))


def bucket_label(bucket_idx: int, edges: np.ndarray) -> str:
    if bucket_idx == 0:
        return f"≤ {edges[0]:.0f}"
    if bucket_idx >= len(edges):
        return f"> {edges[-1]:.0f}"
    return f"({edges[bucket_idx - 1]:.0f}, {edges[bucket_idx]:.0f}]"


# ============================================================
# Classifier model
# ============================================================
class K1DCNNCls(nn.Module):
    """Та же architecture как Phase 23 K1DCNN, но output = K_buckets logits."""
    def __init__(self, n_layer_heads: int, head_dim: int, n_buckets: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_layer_heads, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_buckets),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def train_classifier(X_raw: np.ndarray, y_buckets: np.ndarray,
                     n_lh: int, head_dim: int, n_buckets: int,
                     epochs: int = 400, lr: float = 1e-3, batch: int = 8):
    model = K1DCNNCls(n_lh, head_dim, n_buckets)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # Class weighting для balanced training
    class_counts = np.bincount(y_buckets, minlength=n_buckets).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights /= class_weights.sum()
    log.info(f"  Bucket distribution: {class_counts.tolist()}")
    log.info(f"  Class weights: {class_weights.round(3).tolist()}")
    weight_t = torch.tensor(class_weights, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight_t)
    Xt = torch.tensor(X_raw, dtype=torch.float32)
    yt = torch.tensor(y_buckets, dtype=torch.long)
    n = len(y_buckets)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
    # Train accuracy
    model.eval()
    with torch.no_grad():
        train_preds = model(Xt).argmax(dim=-1).numpy()
    train_acc = float((train_preds == y_buckets).mean())
    log.info(f"  Training done, train accuracy = {train_acc:.3f}")
    return model, train_acc


# ============================================================
# Data
# ============================================================
def build_training_data(captures_dir: Path, fdp_file: Path) -> tuple:
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
    return np.array(X_raw, dtype=np.float32), np.array(y, dtype=np.float32)


def main() -> int:
    args = parse_args()

    log.info("Step 1: Load training captures")
    X_raw, y = build_training_data(args.training_captures, args.training_fdp)
    log.info(f"  X_raw={X_raw.shape}, y range=[{y.min():.0f}, {y.max():.0f}]")

    log.info(f"Step 2: Compute adaptive bucket edges (n_buckets={args.n_buckets})")
    edges = compute_bucket_edges(y, args.n_buckets)
    log.info(f"  Edges: {edges.round(0).tolist()}")
    for b in range(args.n_buckets):
        log.info(f"    Bucket {b}: {bucket_label(b, edges)}")
    y_buckets = np.array([assign_bucket(yi, edges) for yi in y], dtype=np.int64)

    log.info("Step 3: Normalize K (using training stats)")
    mean = X_raw.mean(axis=(0, 2), keepdims=True)
    std = X_raw.std(axis=(0, 2), keepdims=True) + 1e-8
    Xn = (X_raw - mean) / std

    log.info("Step 4: Train bucket classifier (CNN)")
    model, train_acc = train_classifier(
        Xn, y_buckets, n_lh=N_LAYERS * N_KV_HEADS, head_dim=128,
        n_buckets=args.n_buckets,
    )

    log.info(f"Step 5: Load fresh MATH-500 problems {args.start_idx}..{args.end_idx}")
    n_fresh = args.end_idx - args.start_idx + 1
    problems = load_math_dataset("math-500", num_samples=args.end_idx + 1, shuffle=False)
    fresh_probs = problems[args.start_idx:args.end_idx + 1]
    log.info(f"  Got {len(fresh_probs)} problems")

    log.info("Step 6: Load Qwen3 and run pipeline")
    runner = CaptureRunner()
    runner.load_model(args.model)

    results = []
    for i, prob in enumerate(fresh_probs):
        prompt_tokens = p26._build_prompt_tokens(runner._tokenizer, prob.problem)
        log.info(f"  Problem {i + 1}/{n_fresh} (idx={prob.idx}): prompt_len={len(prompt_tokens)}")
        log.info(f"    Generating bf16 (max {args.max_gen})...")
        bf = p26._greedy_generate(runner, prompt_tokens, args.max_gen, "bf16")
        log.info(f"    Generating fp8_e4m3 (max {args.max_gen})...")
        fp = p26._greedy_generate(runner, prompt_tokens, args.max_gen, "fp8_e4m3")
        true_fdp = p26.find_fdp(bf, fp)
        true_bucket = assign_bucket(true_fdp, edges)
        log.info(f"    True FDP={true_fdp}, true bucket={true_bucket} ({bucket_label(true_bucket, edges)})")
        if true_fdp >= args.max_gen:
            results.append({"problem_idx": int(prob.idx), "status": "no_divergence"})
            continue
        log.info(f"    Capturing K window...")
        K = p26._capture_k_window(runner, prompt_tokens, bf, true_fdp,
                                  window_pre=150, window_post=100)
        if K is None:
            results.append({"problem_idx": int(prob.idx), "true_fdp": int(true_fdp),
                            "status": "capture_failed"})
            continue
        K_half = [Kl[:125] for Kl in K]
        # Pad
        K_padded = []
        for Kl in K_half:
            if Kl.shape[0] < 125:
                pad_len = 125 - Kl.shape[0]
                Kl = torch.cat([Kl, torch.zeros(pad_len, *Kl.shape[1:], dtype=Kl.dtype)], dim=0)
            K_padded.append(Kl[:125])
        X = p23.kper_to_raw_matrix(K_padded)  # [224, 128]
        Xn_new = (X - mean[0]) / std[0]
        Xt = torch.tensor(Xn_new[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            logits = model(Xt)
            probs = torch.softmax(logits, dim=-1).numpy()[0]
            pred_bucket = int(np.argmax(probs))
        log.info(f"    Pred bucket={pred_bucket} ({bucket_label(pred_bucket, edges)}), "
                 f"confidence={probs[pred_bucket]:.2f}, all_probs={probs.round(2).tolist()}")
        results.append({
            "problem_idx": int(prob.idx),
            "prompt_len": len(prompt_tokens),
            "true_fdp": int(true_fdp),
            "true_bucket": true_bucket,
            "pred_bucket": pred_bucket,
            "correct": bool(pred_bucket == true_bucket),
            "probs": probs.tolist(),
            "max_prob": float(probs[pred_bucket]),
            "status": "ok",
        })

    runner.unload()

    ok = [r for r in results if r["status"] == "ok"]
    n_correct = sum(1 for r in ok if r["correct"])
    out = {
        "n_buckets": args.n_buckets,
        "bucket_edges": edges.tolist(),
        "bucket_labels": [bucket_label(b, edges) for b in range(args.n_buckets)],
        "training_train_accuracy": train_acc,
        "n_test_problems": n_fresh,
        "n_evaluated": len(ok),
        "n_correct": int(n_correct),
        "test_accuracy": float(n_correct / len(ok)) if ok else None,
        "per_problem": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    log.info(f"Saved {args.output}")
    log.info("=" * 60)
    log.info("RESULTS:")
    for r in results:
        if r["status"] == "ok":
            mark = "✓" if r["correct"] else "✗"
            log.info(f"  {mark} Problem {r['problem_idx']}: true_fdp={r['true_fdp']}, "
                     f"true_bucket={r['true_bucket']}, pred={r['pred_bucket']} "
                     f"(conf={r['max_prob']:.2f})")
        else:
            log.info(f"  ! Problem {r['problem_idx']}: {r['status']}")
    log.info(f"  Bucket accuracy: {n_correct}/{len(ok)} = {n_correct/max(len(ok),1):.2%}")

    # Plot confusion + confidence
    if ok:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            # Confusion-style scatter
            for r in ok:
                color = "tab:green" if r["correct"] else "tab:red"
                ax1.scatter(r["true_bucket"], r["pred_bucket"], s=300, color=color,
                            edgecolors="black", linewidths=1.5, alpha=0.7, zorder=3)
                ax1.annotate(f"#{r['problem_idx']}\nFDP={r['true_fdp']}",
                             (r["true_bucket"], r["pred_bucket"]),
                             xytext=(10, 10), textcoords="offset points", fontsize=9)
            mn = -0.5
            mx = args.n_buckets - 0.5
            ax1.plot([mn, mx], [mn, mx], "k--", alpha=0.3)
            ax1.set_xlabel("True bucket")
            ax1.set_ylabel("Predicted bucket")
            ax1.set_xticks(range(args.n_buckets))
            ax1.set_yticks(range(args.n_buckets))
            ax1.set_xlim(mn, mx)
            ax1.set_ylim(mn, mx)
            ax1.set_xticklabels([bucket_label(b, edges) for b in range(args.n_buckets)],
                                 rotation=20, ha="right", fontsize=9)
            ax1.set_yticklabels([bucket_label(b, edges) for b in range(args.n_buckets)],
                                 fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"Bucket classification on {len(ok)} fresh problems\n"
                          f"Accuracy = {n_correct}/{len(ok)} = {n_correct/len(ok):.0%}")
            # Probability bars
            probs_arr = np.array([r["probs"] for r in ok])
            x = np.arange(args.n_buckets)
            for i, r in enumerate(ok):
                offset = (i - len(ok)/2) * 0.15
                ax2.bar(x + offset, r["probs"], 0.13,
                        label=f"#{r['problem_idx']} (FDP={r['true_fdp']})")
            ax2.set_xticks(x)
            ax2.set_xticklabels([bucket_label(b, edges) for b in range(args.n_buckets)],
                                 rotation=20, ha="right", fontsize=9)
            ax2.set_ylabel("Predicted probability")
            ax2.set_title("Per-problem bucket probability distribution")
            ax2.legend(fontsize=8, loc="upper right")
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = args.output.parent / "plots" / "cnn_buckets_test.png"
            plot_path.parent.mkdir(exist_ok=True)
            plt.savefig(plot_path, dpi=130)
            log.info(f"Saved plot {plot_path}")
        except ImportError:
            pass

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
