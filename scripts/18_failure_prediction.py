"""Phase 18: failure prediction — can early KL trajectory predict final divergence?

Hypothesis: даже за первые ~30 decode-шагов KL trajectory несёт enough signal
чтобы classify "будет ли финальный ответ отличаться от bf16". Это бы дало
**practical detector**: дешёвая ранняя стопа квантованной генерации, если
prediction говорит "будет collapse".

Setup:
  Features (per problem, per quant):
    - KL[0..N_EARLY] из bf16_tf vs <quant>_tf trajectories (per position)
    - Aggregated stats: mean, max, slope, area under KL curve в первые N шагов
    - Use TF mode (deterministic) для clean signal-to-noise
  Labels (per problem, per quant):
    - 0 = "both_correct" or "baseline_only" (quant got it WRONG)
    - Actually: 1 = "different answer" iff baseline_boxed != quant_boxed
       (boolean from FDP file's boxed fields)

Model: logistic regression на flattened features. Cross-validated.
Report AUC and key features.

Outputs:
  failure_prediction_<quant>.json — AUC, feature importance
  plots/failure_prediction_<quant>.png — ROC curve

Note: ~80 problems is a small dataset для CV. Use leave-one-out CV или
5-fold CV with care.

Usage:
    python scripts/18_failure_prediction.py --quant fp8_e4m3 --n-early 30
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("phase18")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 18: failure prediction")
    p.add_argument("--captures-dir", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b"))
    p.add_argument("--fdps-dir", type=Path, default=Path("outputs/fdps"))
    p.add_argument("--quant", default="fp8_e4m3",
                   choices=["fp8_e4m3", "fp8_e5m2", "hqq_int4", "hqq_int2"])
    p.add_argument("--n-early", type=int, default=30,
                   help="How many early window positions to use as features")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _load_labels(fdps_dir: Path, model: str, quant: str) -> dict[int, int]:
    """Per-problem label: 1 if baseline answer != quant answer, else 0.

    For FP8 quants we have actual FDP files. For HQQ we copied fp8_e4m3 FDP,
    so we use the FP8 labels as proxy (this is conservative — actual HQQ
    might differ but we don't have its trace).
    """
    # Map hqq to fp8_e4m3 labels (since HQQ trace wasn't generated)
    label_quant = quant if quant.startswith("fp8") else "fp8_e4m3"
    fdp_path = fdps_dir / f"{model}_{label_quant}.jsonl"
    if not fdp_path.exists():
        raise FileNotFoundError(fdp_path)
    labels = {}
    for line in fdp_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        bb = r.get("baseline_boxed")
        qb = r.get("quant_boxed")
        if bb is None and qb is None:
            continue  # no_boxed — exclude
        labels[r["problem_idx"]] = int(bb != qb)
    return labels


def _compute_features(cap_bf16, cap_quant, n_early: int) -> dict:
    """Per-problem aggregated KL features from first n_early positions."""
    import torch
    if cap_bf16.meta["window_start"] != cap_quant.meta["window_start"]:
        return None
    log_b = cap_bf16.logits[:n_early].float()
    log_q = cap_quant.logits[:n_early].float()
    if log_b.shape[0] < 5 or log_q.shape[0] < 5:
        return None
    p_b = torch.softmax(log_b, dim=-1)
    p_q = torch.softmax(log_q, dim=-1)
    eps = 1e-12
    kl = (p_b * (torch.log(p_b + eps) - torch.log(p_q + eps))).sum(dim=-1).numpy()
    n = len(kl)
    # Margins (bf16 top1-top2 logit gap) early
    top2 = log_b.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).numpy()
    # Features
    feats = {
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "kl_std": float(kl.std()),
        "kl_slope": float(np.polyfit(np.arange(n), kl, 1)[0]) if n >= 3 else 0.0,
        "kl_auc": float(kl.sum()),  # area under curve
        "kl_last": float(kl[-1]),
        "kl_late_mean": float(kl[n//2:].mean()) if n >= 4 else float(kl.mean()),
        "margin_mean": float(margin.mean()),
        "margin_min": float(margin.min()),
    }
    return feats


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.captures_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = args.captures_dir.name

    from kvtrace.capture.storage import load_capture

    labels = _load_labels(args.fdps_dir, model, args.quant)
    log.info("Loaded %d problem labels for %s", len(labels), args.quant)

    bf16_dir = args.captures_dir / "bf16_tf"
    quant_dir = args.captures_dir / f"{args.quant}_tf"

    X_rows: list[list[float]] = []
    y: list[int] = []
    feature_names: list[str] | None = None
    pids: list[int] = []
    for pid, label in sorted(labels.items()):
        bf16_path = bf16_dir / f"{pid}.safetensors"
        quant_path = quant_dir / f"{pid}.safetensors"
        if not bf16_path.exists() or not quant_path.exists():
            continue
        cap_b = load_capture(bf16_path)
        cap_q = load_capture(quant_path)
        feats = _compute_features(cap_b, cap_q, args.n_early)
        if feats is None:
            continue
        if feature_names is None:
            feature_names = sorted(feats.keys())
        X_rows.append([feats[k] for k in feature_names])
        y.append(label)
        pids.append(pid)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y, dtype=np.int32)
    log.info("Dataset: %d samples, %d features, positive rate %.2f%%",
             len(y), X.shape[1], 100*y.mean())

    if y.sum() == 0 or y.sum() == len(y):
        log.error("Degenerate labels (all 0 or all 1) — can't train classifier")
        return 1

    # Train + evaluate via leave-one-out CV
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # LOO-CV predictions
    n = len(y)
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if y[mask].sum() == 0 or y[mask].sum() == mask.sum():
            loo_preds[i] = y[mask].mean()
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xs[mask], y[mask])
        loo_preds[i] = clf.predict_proba(Xs[i:i+1])[0, 1]

    auc = float(roc_auc_score(y, loo_preds))
    log.info("=== Results ===")
    log.info("  LOO-CV AUC = %.3f", auc)

    # Feature importance from full-data fit
    clf_full = LogisticRegression(max_iter=1000, C=1.0).fit(Xs, y)
    coefs = clf_full.coef_[0]
    importance = sorted(
        [(name, float(c)) for name, c in zip(feature_names, coefs)],
        key=lambda t: -abs(t[1])
    )
    log.info("  Top 5 feature coefficients (standardized):")
    for name, c in importance[:5]:
        log.info("    %s: %+.3f", name, c)

    summary = {
        "quant": args.quant,
        "n_early": args.n_early,
        "n_samples": int(n),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "loo_cv_auc": auc,
        "feature_importance": [{"name": n, "coef": c} for n, c in importance],
        "per_problem_predictions": [
            {"pid": int(p), "label": int(l), "pred": float(pred)}
            for p, l, pred in zip(pids, y, loo_preds)
        ],
    }
    out_json = output_dir / f"failure_prediction_{args.quant}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log.info("Saved %s", out_json)

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return 0
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        fpr, tpr, _ = roc_curve(y, loo_preds)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(fpr, tpr, "C0-", linewidth=2, label=f"LOO AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"Failure prediction ROC — {args.quant}\n"
                     f"(first {args.n_early} decode steps, n={n} problems)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"failure_prediction_{args.quant}.png", dpi=100)
        plt.close()
        log.info("Saved ROC plot")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
