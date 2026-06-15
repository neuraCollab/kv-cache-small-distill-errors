"""Phase 25: FDP-предсказатель на DeepSeek-R1-Distill-Qwen-1.5B.

Воспроизводим Phase 23 setup на DeepSeek-R1-Distill-Qwen-1.5B:
  - 28 layers (как Qwen3-1.7B), но 2 KV-heads (GQA-2, не 8)
  - Variable seq length captures (window truncated если FDP near start)

Также проверяем cross-model transfer (Qwen3 trained → DeepSeek test) на
head-pooled features.
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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
p23 = import_module("23_fdp_predictor_extended")

log = logging.getLogger("phase25")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

N_LAYERS = 28


def extract_features_flexible(K_per_layer: list[torch.Tensor]) -> np.ndarray:
    """Same as p23.extract_features но без hardcoded N_KV_HEADS.
    Если разное n_kv_heads — pool по heads."""
    n_layers = len(K_per_layer)
    feats = []
    for L, K in enumerate(K_per_layer):
        Kf = K.float()  # [seq, h, d]
        K_abs = Kf.abs()
        n_h = K.shape[1]
        # Per-(layer, head) — but pool over heads чтобы переносить размерность
        max_abs_pool = float(K_abs.amax(dim=(0, 2)).mean())   # avg over heads
        max_abs_max = float(K_abs.amax())                       # max over all
        mean_abs = float(K_abs.mean())
        std_v = float(Kf.std())
        fro_norm = float(Kf.norm())
        # Per (head): top-1 channel concentration averaged + max
        top1_ch = []
        for h in range(n_h):
            ch_e = (Kf[:, h, :] ** 2).sum(dim=0)
            sd, _ = ch_e.sort(descending=True)
            tot = float(sd.sum())
            top1_ch.append(float(sd[0]) / tot if tot > 0 else 0)
        top1_ch_mean = float(np.mean(top1_ch))
        top1_ch_max = float(np.max(top1_ch))
        # Per-layer top1, top10 (over h*d)
        ch_e_flat = (Kf ** 2).sum(dim=0).flatten()
        sd, _ = ch_e_flat.sort(descending=True)
        tot = float(sd.sum())
        top1_frac = float(sd[0]) / tot if tot > 0 else 0
        top10_frac = float(sd[:10].sum()) / tot if tot > 0 else 0
        feats.extend([max_abs_pool, max_abs_max, mean_abs, std_v, fro_norm,
                      top1_ch_mean, top1_ch_max, top1_frac, top10_frac])
    return np.array(feats, dtype=np.float32)


def load_K_window(path: Path) -> list[torch.Tensor]:
    """Load K_pre per layer; can have any seq length."""
    t = load_file(str(path))
    K = []
    for L in range(N_LAYERS):
        key = f"k_pre_l{L}"
        if key in t:
            K_l = t[key]
            # Use first half of available seq (analogous to half_only=True)
            half = max(K_l.shape[0] // 2, 30)  # min 30 positions
            K.append(K_l[:half])
    return K


def kper_to_raw_matrix_flexible(K_per_layer: list[torch.Tensor],
                                 head_dim: int = 128) -> np.ndarray:
    """[n_layers * n_kv_heads, head_dim] — per-(L, h) max_abs over seq."""
    rows = []
    for K in K_per_layer:
        for h in range(K.shape[1]):
            rows.append(K[:, h, :].float().abs().amax(dim=0).numpy())
    return np.array(rows, dtype=np.float32)


def build_data(captures_dir: Path, fdp_file: Path) -> tuple:
    fdps = p23.load_fdps(fdp_file)
    X_feat, X_raw_list, y, ids = [], [], [], []
    for cf in sorted(captures_dir.glob("*.safetensors"), key=lambda p: int(p.stem)):
        pid = int(cf.stem)
        if pid not in fdps:
            continue
        K = load_K_window(cf)
        if len(K) != N_LAYERS:
            continue
        X_feat.append(extract_features_flexible(K))
        X_raw_list.append(kper_to_raw_matrix_flexible(K))
        y.append(fdps[pid])
        ids.append(pid)
    # Pad X_raw to common shape (different sequences shouldn't differ here — n_lh same)
    if X_raw_list:
        shape0 = X_raw_list[0].shape
        for x in X_raw_list:
            assert x.shape == shape0, f"X_raw shape mismatch: {x.shape} vs {shape0}"
    return (np.array(X_feat, dtype=np.float32),
            np.array(X_raw_list, dtype=np.float32) if X_raw_list else np.empty((0,)),
            np.array(y, dtype=np.float32), ids)


def eval_metrics(yt, yp) -> dict:
    return {
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "spearman": float(spearmanr(yt, yp)[0]) if len(set(yp)) > 1 else float("nan"),
    }


def cv_train_eval(X_feat: np.ndarray, X_raw: np.ndarray, y: np.ndarray) -> dict:
    """5-fold CV for GBM, MLP, CNN."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gbm_r, mlp_r, cnn_r = [], [], []
    n_lh, head_dim = X_raw.shape[1], X_raw.shape[2]
    for fold, (tr, vl) in enumerate(kf.split(X_feat)):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_feat[tr])
        Xv = scaler.transform(X_feat[vl])
        gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                        learning_rate=0.05, random_state=42)
        gbm.fit(Xt, y[tr])
        gp = gbm.predict(Xv)
        gbm_r.append(eval_metrics(y[vl], gp))

        mp = p23.train_mlp(Xt, y[tr], Xv, in_dim=X_feat.shape[1])
        mlp_r.append(eval_metrics(y[vl], mp))

        # CNN
        mean = X_raw[tr].mean(axis=(0, 2), keepdims=True)
        std = X_raw[tr].std(axis=(0, 2), keepdims=True) + 1e-8
        Xtr_n = (X_raw[tr] - mean) / std
        Xv_n = (X_raw[vl] - mean) / std
        cp = p23.train_cnn(Xtr_n, y[tr], Xv_n, n_lh, head_dim,
                            epochs=200, lr=1e-3, batch=8)
        cnn_r.append(eval_metrics(y[vl], cp))
        log.info(f"  fold {fold + 1}: GBM R²={gbm_r[-1]['r2']:.3f}, "
                 f"MLP={mlp_r[-1]['r2']:.3f}, CNN={cnn_r[-1]['r2']:.3f}")
    def agg(rs):
        out = {}
        for m in ["r2", "mae", "spearman"]:
            vals = [r[m] for r in rs if not (m == "spearman" and np.isnan(r[m]))]
            out[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
            out[f"{m}_std"] = float(np.std(vals)) if vals else float("nan")
        return out
    return {
        "GBM": agg(gbm_r),
        "MLP": agg(mlp_r),
        "CNN": agg(cnn_r),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--captures-dir", type=Path,
                   default=Path("outputs/kv_capture/deepseek-r1-distill-qwen-1.5b/fp8_e4m3_tf"))
    p.add_argument("--fdp-file", type=Path,
                   default=Path("outputs/fdps/deepseek-r1-distill-qwen-1.5b_fp8_e4m3.jsonl"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/kv_capture/deepseek-r1-distill-qwen-1.5b/analysis/fdp_predictor.json"))
    args = p.parse_args()

    log.info("Loading DeepSeek-R1-Distill captures...")
    X_feat, X_raw, y, ids = build_data(args.captures_dir, args.fdp_file)
    log.info(f"  X_feat={X_feat.shape}, X_raw={X_raw.shape}, "
             f"y range=[{y.min():.0f}, {y.max():.0f}], mean={y.mean():.0f}")

    log.info("Running 5-fold CV on DeepSeek...")
    results = cv_train_eval(X_feat, X_raw, y)

    out = {
        "model": "DeepSeek-R1-Distill-Qwen-1.5B",
        "captures_dir": str(args.captures_dir),
        "n_problems": int(len(y)),
        "n_features": int(X_feat.shape[1]),
        "raw_shape": list(X_raw.shape),
        "fdp_range": [float(y.min()), float(y.max())],
        "fdp_mean": float(y.mean()),
        "fdp_std": float(y.std()),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    log.info(f"Saved {args.output}")

    for m, agg in results.items():
        log.info(f">>> {m}: R²={agg['r2_mean']:.3f} ± {agg['r2_std']:.3f}, "
                 f"MAE={agg['mae_mean']:.1f}, ρ={agg['spearman_mean']:.3f}")

    # Compare with Qwen3
    qwen_p = Path("outputs/kv_capture/qwen3-1.7b/analysis/fdp_predictor_extended.json")
    if qwen_p.exists():
        qd = json.load(open(qwen_p))
        log.info("\nCompare to Qwen3-1.7B:")
        for model in ["GBM", "MLP", "CNN"]:
            try:
                q_r2 = qd["baseline_decode_window"][model.lower()]["r2_mean"] if model != "CNN" \
                        else qd["D_cnn"]["r2_mean"]
                d_r2 = results[model]["r2_mean"]
                log.info(f"  {model}: Qwen3={q_r2:.3f}, DeepSeek={d_r2:.3f}, diff={d_r2 - q_r2:+.3f}")
            except KeyError:
                pass

    # Side-by-side bar plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        models = ["GBM", "MLP", "CNN"]
        qwen_r2 = []
        ds_r2 = []
        for m in models:
            d_r2 = results[m]["r2_mean"]
            ds_r2.append(d_r2)
            if qwen_p.exists():
                qd = json.load(open(qwen_p))
                try:
                    q_r2 = qd["baseline_decode_window"][m.lower()]["r2_mean"] if m != "CNN" \
                            else qd["D_cnn"]["r2_mean"]
                except KeyError:
                    q_r2 = 0
                qwen_r2.append(q_r2)
            else:
                qwen_r2.append(0)
        x = np.arange(len(models))
        w = 0.35
        ax.bar(x - w/2, qwen_r2, w, label="Qwen3-1.7B\n(28L, 8 KV-heads)",
               color="tab:blue")
        ax.bar(x + w/2, ds_r2, w, label="DeepSeek-R1-Distill-Qwen-1.5B\n(28L, 2 KV-heads)",
               color="tab:cyan")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("R² (5-fold CV)")
        ax.set_title("FDP regression — same predictor on two models\n(decode-window features, fp8_e4m3)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        for xi, q, d in zip(x, qwen_r2, ds_r2):
            ax.text(xi - w/2, q + 0.02, f"{q:.2f}", ha="center", fontsize=9)
            ax.text(xi + w/2, d + 0.02, f"{d:.2f}", ha="center", fontsize=9)
        ax.set_ylim(min(min(qwen_r2 + ds_r2) - 0.1, 0), 1.05)
        plt.tight_layout()
        out_plot = args.output.parent / "plots" / "fdp_predictor_deepseek_vs_qwen3.png"
        out_plot.parent.mkdir(exist_ok=True)
        plt.savefig(out_plot, dpi=130)
        log.info(f"Saved comparison plot {out_plot}")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
