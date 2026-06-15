"""Phase 22: FDP regression от K-матриц на 80 задачах.

Задача: предсказать fdp_token_idx (целое 0..2000+) для пары
(problem, quant=fp8_e4m3) на Qwen3-1.7B.

Признаки извлекаются из K_pre (pre-quant K тензор captured в TF режиме).
Используем только первую половину window [0:125] = positions [FDP-150, FDP-25]
чтобы не лить таргет.

Модели:
  1. Baseline: GradientBoostingRegressor (~XGBoost-style)
  2. MLP: маленькая FC сетка
  3. 1D-CNN: над per-layer K каналов

Metrics: MAE, RMSE, R^2, Spearman correlation.
5-fold CV для всех моделей.

Usage:
    python scripts/22_fdp_predictor.py --captures-dir outputs/kv_capture/qwen3-1.7b/fp8_e4m3_tf \\
        --fdp-file outputs/fdps/qwen3-1.7b_fp8_e4m3.jsonl
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
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("phase22")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FDP regression от K-матриц")
    p.add_argument("--captures-dir", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b/fp8_e4m3_tf"))
    p.add_argument("--fdp-file", type=Path,
                   default=Path("outputs/fdps/qwen3-1.7b_fp8_e4m3.jsonl"))
    p.add_argument("--n-layers", type=int, default=28)
    p.add_argument("--use-half-window-only", action="store_true", default=True,
                   help="Использовать только positions [0:125] (до FDP), default True")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b/analysis/fdp_predictor_results.json"))
    return p.parse_args()


# ============================================================
# Feature extraction
# ============================================================
def extract_features(K_per_layer: list[torch.Tensor]) -> dict:
    """K_per_layer: list of 28 tensors, each [seq=125, h=8, d=128]
    Returns: dict с aggregated features.
    """
    feats = {}
    n_layers = len(K_per_layer)
    # Per-(layer, head) statistics
    max_abs_lh = np.zeros((n_layers, 8))      # [L, h]
    mean_abs_lh = np.zeros((n_layers, 8))
    std_lh = np.zeros((n_layers, 8))
    # Per-layer concentration
    top1_frac_l = np.zeros(n_layers)
    top10_frac_l = np.zeros(n_layers)
    # Per-layer norm
    fro_norm_l = np.zeros(n_layers)
    # Per-(layer, head) head_dim concentration: top-1 channel fraction
    top1_ch_per_lh = np.zeros((n_layers, 8))

    for L, K in enumerate(K_per_layer):
        K_abs = K.float().abs()  # [seq, h, d]
        # Per (head): aggregate over seq+dim
        max_abs_lh[L] = K_abs.amax(dim=(0, 2)).numpy()
        mean_abs_lh[L] = K_abs.mean(dim=(0, 2)).numpy()
        std_lh[L] = K.float().std(dim=(0, 2)).numpy()
        fro_norm_l[L] = float(K.float().norm())
        # Per-(head): top-1 channel concentration (max-channel fraction of sum of squares)
        ch_energy = (K.float() ** 2).sum(dim=0)  # [h, d]
        for h in range(8):
            sorted_desc, _ = ch_energy[h].sort(descending=True)
            total = float(sorted_desc.sum())
            if total > 0:
                top1_ch_per_lh[L, h] = float(sorted_desc[0]) / total
        # Per-layer top-1, top-10 channel concentration (over all h × d)
        flat_energy = ch_energy.flatten()
        sorted_flat, _ = flat_energy.sort(descending=True)
        total_layer = float(sorted_flat.sum())
        if total_layer > 0:
            top1_frac_l[L] = float(sorted_flat[0]) / total_layer
            top10_frac_l[L] = float(sorted_flat[:10].sum()) / total_layer

    # Flatten all into single vector
    feature_vec = np.concatenate([
        max_abs_lh.flatten(),      # 28*8 = 224
        mean_abs_lh.flatten(),     # 224
        std_lh.flatten(),          # 224
        top1_frac_l,               # 28
        top10_frac_l,              # 28
        fro_norm_l,                # 28
        top1_ch_per_lh.flatten(),  # 224
    ])  # total = 980
    return feature_vec


def load_K_pre_window(path: Path, n_layers: int, half_only: bool) -> list[torch.Tensor]:
    """Load K_pre[0:half] for each layer (half = 125 if half_only else 251)."""
    t = load_file(path)
    end = 125 if half_only else 251
    K_per_layer = []
    for L in range(n_layers):
        key = f"k_pre_l{L}"
        if key in t:
            K_per_layer.append(t[key][:end])
    return K_per_layer


# ============================================================
# Models
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 32)):
        super().__init__()
        dims = [in_dim, *hidden, 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val,
              in_dim: int, epochs: int = 200, lr: float = 1e-3, batch: int = 16) -> tuple:
    """Returns (predictions on val, final train_loss, final val_loss)."""
    model = MLP(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    n = len(X_train)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(X_train_t[idx])
            loss = nn.functional.mse_loss(pred, y_train_t[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).numpy()
    return val_pred


def evaluate_predictions(y_true, y_pred, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    log.info(f"  {label}: MAE={mae:.1f}, RMSE={rmse:.1f}, R²={r2:.3f}, Spearman={rho:.3f}")
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2), "spearman": float(rho)}


# ============================================================
# Main
# ============================================================
def main() -> int:
    args = parse_args()

    log.info("Loading FDP labels...")
    fdps = {}
    for line in args.fdp_file.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        fdps[d["problem_idx"]] = d["fdp_token_idx"]
    log.info(f"  Got {len(fdps)} FDP labels, range [{min(fdps.values())}, {max(fdps.values())}]")

    log.info("Loading captures + extracting features...")
    X, y, problem_ids = [], [], []
    capture_files = sorted(args.captures_dir.glob("*.safetensors"),
                           key=lambda p: int(p.stem))
    for cf in capture_files:
        pid = int(cf.stem)
        if pid not in fdps:
            continue
        K_per_layer = load_K_pre_window(cf, args.n_layers, args.use_half_window_only)
        if len(K_per_layer) != args.n_layers:
            log.warning(f"  problem {pid}: got {len(K_per_layer)} layers, skipping")
            continue
        feat = extract_features(K_per_layer)
        X.append(feat)
        y.append(fdps[pid])
        problem_ids.append(pid)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    log.info(f"  Built X={X.shape}, y={y.shape} (min={y.min():.0f}, max={y.max():.0f}, mean={y.mean():.0f})")

    # 5-fold CV
    log.info("Running 5-fold CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {"gbm": [], "mlp": [], "mean_baseline": []}
    pred_records = {"gbm": [], "mlp": [], "mean": [], "true": []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        log.info(f"--- Fold {fold + 1}/5 ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # Baseline: predict mean
        mean_pred = np.full_like(y_val, y_train.mean())
        m = evaluate_predictions(y_val, mean_pred, "mean baseline")
        results["mean_baseline"].append(m)

        # Gradient Boosting Regressor
        gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                        learning_rate=0.05, random_state=42)
        gbm.fit(X_train_s, y_train)
        gbm_pred = gbm.predict(X_val_s)
        m = evaluate_predictions(y_val, gbm_pred, "GBM (boosting)")
        results["gbm"].append(m)

        # MLP
        mlp_pred = train_mlp(X_train_s, y_train, X_val_s, y_val,
                             in_dim=X_train_s.shape[1])
        m = evaluate_predictions(y_val, mlp_pred, "MLP")
        results["mlp"].append(m)

        pred_records["true"].extend(y_val.tolist())
        pred_records["gbm"].extend(gbm_pred.tolist())
        pred_records["mlp"].extend(mlp_pred.tolist())
        pred_records["mean"].extend(mean_pred.tolist())

    # Aggregate
    log.info("=" * 50)
    log.info("5-fold CV mean (± std):")
    summary = {}
    for model_name, fold_results in results.items():
        agg = {}
        for metric in ["mae", "rmse", "r2", "spearman"]:
            vals = [fr[metric] for fr in fold_results]
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals))
        summary[model_name] = agg
        log.info(f"  {model_name}:")
        for metric in ["mae", "rmse", "r2", "spearman"]:
            log.info(f"    {metric}: {agg[f'{metric}_mean']:.3f} ± {agg[f'{metric}_std']:.3f}")

    # Save
    out = {
        "model_target": "fp8_e4m3 Qwen3-1.7B FDP token index regression",
        "n_problems": len(y),
        "n_features": int(X.shape[1]),
        "use_half_window_only": args.use_half_window_only,
        "fdp_range": [float(y.min()), float(y.max())],
        "fdp_mean": float(y.mean()),
        "fdp_std": float(y.std()),
        "cv_summary": summary,
        "per_fold_results": results,
        "predictions": pred_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    log.info(f"Saved {args.output}")

    # Quick plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, model_name in zip(axes, ["gbm", "mlp"]):
            ax.scatter(pred_records["true"], pred_records[model_name],
                       alpha=0.6, s=40, edgecolors="black", linewidths=0.5)
            mn, mx = min(pred_records["true"]), max(pred_records["true"])
            ax.plot([mn, mx], [mn, mx], "r--", alpha=0.5, label="y=x")
            ax.set_xlabel("True FDP token index")
            ax.set_ylabel("Predicted FDP token index")
            ax.set_title(f"{model_name.upper()}: R²={summary[model_name]['r2_mean']:.2f} ± {summary[model_name]['r2_std']:.2f}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = args.output.parent / "plots" / "fdp_predictor_scatter.png"
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=120)
        log.info(f"Saved plot {plot_path}")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
