"""Phase 24: проверка overfitting для FDP-предсказателя.

Три проверки:
  1. Label shuffle: рандомизировать y, переобучить — если R² высокий,
     значит модель учит шум, не signal.
  2. Learning curve: train на 30/50/70, валидация на rest. Если plateau
     уже на 30 — generalize OK; если train R² >> val R², overfit.
  3. Train vs val gap: train R² на same folds — если близко к val, нет overfit.

Применяется к 3 моделям: GBM, MLP, CNN на decode-window features
fp8_e4m3 Qwen3-1.7B (тот же setup что Phase 23).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Reuse from phase 23
import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
p23 = import_module("23_fdp_predictor_extended")

log = logging.getLogger("phase24")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--captures-dir", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b/fp8_e4m3_tf"))
    p.add_argument("--fdp-file", type=Path,
                   default=Path("outputs/fdps/qwen3-1.7b_fp8_e4m3.jsonl"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b/analysis/fdp_overfitting_check.json"))
    p.add_argument("--seed-base", type=int, default=42)
    return p.parse_args()


def train_gbm(Xt, yt, Xv, return_train_pred=False):
    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                    learning_rate=0.05, random_state=42)
    gbm.fit(Xt, yt)
    pred_v = gbm.predict(Xv)
    if return_train_pred:
        return pred_v, gbm.predict(Xt)
    return pred_v


def train_cnn_with_curves(Xt, yt, Xv, yv, n_lh, head_dim,
                          epochs=200, lr=1e-3, batch=8) -> tuple:
    """Returns (val_pred, train_loss_curve, val_loss_curve, train_r2_curve, val_r2_curve)."""
    model = p23.K1DCNN(n_lh, head_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    yt_t = torch.tensor(yt, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    yv_t = torch.tensor(yv, dtype=torch.float32)
    n = len(Xt)
    train_losses, val_losses, train_r2s, val_r2s = [], [], [], []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(Xt_t[idx]), yt_t[idx])
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                tp = model(Xt_t).numpy()
                vp = model(Xv_t).numpy()
            train_losses.append(float(((tp - yt) ** 2).mean()))
            val_losses.append(float(((vp - yv) ** 2).mean()))
            train_r2s.append(r2_score(yt, tp))
            val_r2s.append(r2_score(yv, vp))
    model.eval()
    with torch.no_grad():
        return (model(Xv_t).numpy(),
                train_losses, val_losses,
                train_r2s, val_r2s)


def build_data(captures_dir: Path, fdp_file: Path) -> tuple:
    fdps = p23.load_fdps(fdp_file)
    X_feat, X_raw, y, ids = [], [], [], []
    for cf in sorted(captures_dir.glob("*.safetensors"),
                     key=lambda p: int(p.stem)):
        pid = int(cf.stem)
        if pid not in fdps:
            continue
        K = p23.load_K_decode_window(cf, half_only=True)
        if len(K) != p23.N_LAYERS:
            continue
        X_feat.append(p23.extract_features(K))
        X_raw.append(p23.kper_to_raw_matrix(K))
        y.append(fdps[pid])
        ids.append(pid)
    return (np.array(X_feat, dtype=np.float32),
            np.array(X_raw, dtype=np.float32),
            np.array(y, dtype=np.float32),
            ids)


def evaluate_metrics(yt, yp) -> dict:
    return {
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "spearman": float(spearmanr(yt, yp)[0]) if len(set(yp)) > 1 else float("nan"),
    }


def check_1_label_shuffle(X_feat, X_raw, y) -> dict:
    """Shuffle y, train, evaluate. R² должен быть near 0."""
    log.info("=" * 60)
    log.info("Check 1: label shuffle (real signal vs noise floor)")
    rng = np.random.default_rng(42)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {"GBM_real": [], "GBM_shuffled": [],
               "CNN_real": [], "CNN_shuffled": []}
    n_lh, head_dim = X_raw.shape[1], X_raw.shape[2]
    for fold, (tr, vl) in enumerate(kf.split(X_feat)):
        # Real y
        s = StandardScaler()
        Xs = s.fit_transform(X_feat[tr])
        Xv_s = s.transform(X_feat[vl])
        # GBM
        pred = train_gbm(Xs, y[tr], Xv_s)
        results["GBM_real"].append(evaluate_metrics(y[vl], pred))
        pred = train_gbm(Xs, y_shuf[tr], Xv_s)
        results["GBM_shuffled"].append(evaluate_metrics(y_shuf[vl], pred))
        # CNN
        mean = X_raw[tr].mean(axis=(0, 2), keepdims=True)
        std = X_raw[tr].std(axis=(0, 2), keepdims=True) + 1e-8
        Xt_r = (X_raw[tr] - mean) / std
        Xv_r = (X_raw[vl] - mean) / std
        pred_real = p23.train_cnn(Xt_r, y[tr], Xv_r, n_lh, head_dim,
                                  epochs=200, lr=1e-3, batch=8)
        results["CNN_real"].append(evaluate_metrics(y[vl], pred_real))
        pred_shuf = p23.train_cnn(Xt_r, y_shuf[tr], Xv_r, n_lh, head_dim,
                                  epochs=200, lr=1e-3, batch=8)
        results["CNN_shuffled"].append(evaluate_metrics(y_shuf[vl], pred_shuf))
        log.info(f"  fold {fold + 1}: GBM real R²={results['GBM_real'][-1]['r2']:.3f}, "
                 f"GBM shuf R²={results['GBM_shuffled'][-1]['r2']:.3f}, "
                 f"CNN real R²={results['CNN_real'][-1]['r2']:.3f}, "
                 f"CNN shuf R²={results['CNN_shuffled'][-1]['r2']:.3f}")
    summary = {}
    for k, vs in results.items():
        r2s = [v["r2"] for v in vs]
        maes = [v["mae"] for v in vs]
        summary[k] = {"r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
                      "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes))}
        log.info(f"  >>> {k}: R²={summary[k]['r2_mean']:.3f}±{summary[k]['r2_std']:.3f}, "
                 f"MAE={summary[k]['mae_mean']:.1f}")
    # Verdict
    gap_gbm = summary["GBM_real"]["r2_mean"] - summary["GBM_shuffled"]["r2_mean"]
    gap_cnn = summary["CNN_real"]["r2_mean"] - summary["CNN_shuffled"]["r2_mean"]
    log.info(f"  GBM R² gap (real - shuf): {gap_gbm:.3f}")
    log.info(f"  CNN R² gap (real - shuf): {gap_cnn:.3f}")
    return summary


def check_2_learning_curve(X_feat, X_raw, y) -> dict:
    """Train на разных размерах train, validate на 20% hold-out."""
    log.info("=" * 60)
    log.info("Check 2: learning curve (overfit detection via train size)")
    rng = np.random.default_rng(42)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    val_idx = idx[:16]  # 20% val
    train_pool = idx[16:]  # 80%
    results = {}
    sizes = [20, 30, 45, 60, len(train_pool)]
    n_lh, head_dim = X_raw.shape[1], X_raw.shape[2]
    for sz in sizes:
        train_idx = train_pool[:sz]
        # GBM
        s = StandardScaler()
        Xt = s.fit_transform(X_feat[train_idx])
        Xv = s.transform(X_feat[val_idx])
        pred_v, pred_t = train_gbm(Xt, y[train_idx], Xv, return_train_pred=True)
        gbm_train_r2 = r2_score(y[train_idx], pred_t)
        gbm_val_r2 = r2_score(y[val_idx], pred_v)
        # CNN
        mean = X_raw[train_idx].mean(axis=(0, 2), keepdims=True)
        std = X_raw[train_idx].std(axis=(0, 2), keepdims=True) + 1e-8
        Xt_r = (X_raw[train_idx] - mean) / std
        Xv_r = (X_raw[val_idx] - mean) / std
        cnn_v_pred = p23.train_cnn(Xt_r, y[train_idx], Xv_r, n_lh, head_dim,
                                   epochs=200, lr=1e-3, batch=8)
        # train predictions
        Xt_t_torch = torch.tensor(Xt_r, dtype=torch.float32)
        model = p23.K1DCNN(n_lh, head_dim)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        for _ in range(200):
            perm = torch.randperm(len(train_idx))
            for i in range(0, len(train_idx), 8):
                bidx = perm[i:i + 8]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(
                    model(Xt_t_torch[bidx]),
                    torch.tensor(y[train_idx][bidx], dtype=torch.float32))
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            cnn_t_pred = model(Xt_t_torch).numpy()
        cnn_train_r2 = r2_score(y[train_idx], cnn_t_pred)
        cnn_val_r2 = r2_score(y[val_idx], cnn_v_pred)
        results[str(sz)] = {
            "n_train": int(sz),
            "gbm_train_r2": float(gbm_train_r2),
            "gbm_val_r2": float(gbm_val_r2),
            "gbm_gap": float(gbm_train_r2 - gbm_val_r2),
            "cnn_train_r2": float(cnn_train_r2),
            "cnn_val_r2": float(cnn_val_r2),
            "cnn_gap": float(cnn_train_r2 - cnn_val_r2),
        }
        log.info(f"  n_train={sz}: GBM train R²={gbm_train_r2:.3f}, val={gbm_val_r2:.3f}, gap={gbm_train_r2-gbm_val_r2:.3f}")
        log.info(f"  n_train={sz}: CNN train R²={cnn_train_r2:.3f}, val={cnn_val_r2:.3f}, gap={cnn_train_r2-cnn_val_r2:.3f}")
    return results


def check_3_loss_curves(X_raw, y) -> dict:
    """Train R² и val R² на каждой эпохе. Detect overfitting from divergence point."""
    log.info("=" * 60)
    log.info("Check 3: training curves (epoch-by-epoch R²)")
    rng = np.random.default_rng(42)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    train_idx = idx[16:]
    val_idx = idx[:16]
    mean = X_raw[train_idx].mean(axis=(0, 2), keepdims=True)
    std = X_raw[train_idx].std(axis=(0, 2), keepdims=True) + 1e-8
    Xt = (X_raw[train_idx] - mean) / std
    Xv = (X_raw[val_idx] - mean) / std
    yt, yv = y[train_idx], y[val_idx]
    _, train_losses, val_losses, train_r2s, val_r2s = train_cnn_with_curves(
        Xt, yt, Xv, yv, X_raw.shape[1], X_raw.shape[2],
        epochs=400, lr=1e-3, batch=8)
    # log epochs and metrics
    epochs = list(range(0, 400, 10)) + [399]
    log.info(f"  epoch 0: train R²={train_r2s[0]:.3f}, val R²={val_r2s[0]:.3f}")
    log.info(f"  epoch 100: train R²={train_r2s[10]:.3f}, val R²={val_r2s[10]:.3f}")
    log.info(f"  epoch 200: train R²={train_r2s[20]:.3f}, val R²={val_r2s[20]:.3f}")
    log.info(f"  epoch 399: train R²={train_r2s[-1]:.3f}, val R²={val_r2s[-1]:.3f}")
    return {
        "epochs": epochs[:len(train_r2s)],
        "train_r2": train_r2s,
        "val_r2": val_r2s,
        "train_loss": train_losses,
        "val_loss": val_losses,
    }


def check_4_noise_input(X_raw, y) -> dict:
    """Train на random gaussian вместо реальных K. R² должен быть near 0."""
    log.info("=" * 60)
    log.info("Check 4: train CNN на random matrices (noise floor)")
    rng = np.random.default_rng(123)
    X_noise = rng.standard_normal(X_raw.shape).astype(np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    n_lh, head_dim = X_raw.shape[1], X_raw.shape[2]
    r2s = []
    for fold, (tr, vl) in enumerate(kf.split(X_noise)):
        # No standardization needed (already gaussian)
        pred = p23.train_cnn(X_noise[tr], y[tr], X_noise[vl], n_lh, head_dim,
                              epochs=200, lr=1e-3, batch=8)
        r2 = r2_score(y[vl], pred)
        r2s.append(r2)
        log.info(f"  fold {fold + 1}: CNN noise R²={r2:.3f}")
    return {"cnn_noise_r2_mean": float(np.mean(r2s)),
            "cnn_noise_r2_std": float(np.std(r2s))}


def main() -> int:
    args = parse_args()
    log.info("Loading captures...")
    X_feat, X_raw, y, ids = build_data(args.captures_dir, args.fdp_file)
    log.info(f"  X_feat={X_feat.shape}, X_raw={X_raw.shape}, y range=[{y.min():.0f},{y.max():.0f}]")

    results = {
        "n_problems": int(len(y)),
        "fdp_range": [float(y.min()), float(y.max())],
        "fdp_mean": float(y.mean()),
        "fdp_std": float(y.std()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(results, indent=2))

    results["check1_label_shuffle"] = check_1_label_shuffle(X_feat, X_raw, y)
    save()
    results["check2_learning_curve"] = check_2_learning_curve(X_feat, X_raw, y)
    save()
    results["check3_training_curves"] = check_3_loss_curves(X_raw, y)
    save()
    results["check4_noise_input"] = check_4_noise_input(X_raw, y)
    save()

    # Verdict
    log.info("=" * 60)
    log.info("VERDICTS:")
    c1 = results["check1_label_shuffle"]
    log.info(f"  CNN real R² = {c1['CNN_real']['r2_mean']:.3f}, "
             f"shuffled R² = {c1['CNN_shuffled']['r2_mean']:.3f}")
    log.info(f"  Gap = {c1['CNN_real']['r2_mean'] - c1['CNN_shuffled']['r2_mean']:.3f}")
    log.info(f"  CNN noise R² = {results['check4_noise_input']['cnn_noise_r2_mean']:.3f}")
    log.info(f"  CNN full-train gap = {results['check2_learning_curve'][str(64)]['cnn_gap']:.3f}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        # Plot 1: learning curve
        sizes = [int(k) for k in results["check2_learning_curve"].keys()]
        gbm_train = [results["check2_learning_curve"][str(s)]["gbm_train_r2"] for s in sizes]
        gbm_val = [results["check2_learning_curve"][str(s)]["gbm_val_r2"] for s in sizes]
        cnn_train = [results["check2_learning_curve"][str(s)]["cnn_train_r2"] for s in sizes]
        cnn_val = [results["check2_learning_curve"][str(s)]["cnn_val_r2"] for s in sizes]
        axes[0].plot(sizes, gbm_train, "o-", label="GBM train", color="blue", alpha=0.5)
        axes[0].plot(sizes, gbm_val, "o-", label="GBM val", color="blue")
        axes[0].plot(sizes, cnn_train, "s-", label="CNN train", color="red", alpha=0.5)
        axes[0].plot(sizes, cnn_val, "s-", label="CNN val", color="red")
        axes[0].set_xlabel("Train size")
        axes[0].set_ylabel("R²")
        axes[0].set_title("Learning curves: train vs val R²")
        axes[0].axhline(0, color="black", linewidth=0.5)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # Plot 2: epoch curves
        c3 = results["check3_training_curves"]
        axes[1].plot(c3["epochs"], c3["train_r2"], "o-", label="train R²", color="red", alpha=0.5)
        axes[1].plot(c3["epochs"], c3["val_r2"], "o-", label="val R²", color="red")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("R²")
        axes[1].set_title("CNN: epoch-by-epoch train vs val R²")
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = args.output.parent / "plots" / "fdp_overfitting_check.png"
        plt.savefig(plot_path, dpi=130)
        log.info(f"Saved plot {plot_path}")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
