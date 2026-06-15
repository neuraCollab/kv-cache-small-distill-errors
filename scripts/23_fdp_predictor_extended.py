"""Phase 23: расширенный FDP-предсказатель — 6 экспериментов:
  A. Honest prompt-only baseline (только prefill K, не decode-window)
  B. Two-stage: classifier early/late + per-cluster regression
  C. Feature importance из GBM
  D. 1D-CNN на сырых K-матрицах
  E. Cross-quant transfer
  F. Cross-model transfer Qwen3-1.7B -> DeepSeek-R1-Distill

Все результаты в одном JSON + comparison plot.
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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, accuracy_score)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("phase23")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_USER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# ==============================================================
# Feature extraction (общая)
# ==============================================================
N_LAYERS = 28
N_KV_HEADS = 8


def feature_names() -> list[str]:
    names = []
    for s in ["max_abs", "mean_abs", "std", "top1_ch_lh"]:
        for L in range(N_LAYERS):
            for h in range(N_KV_HEADS):
                names.append(f"{s}_L{L}_H{h}")
    for s in ["top1_frac", "top10_frac", "fro_norm"]:
        for L in range(N_LAYERS):
            names.append(f"{s}_L{L}")
    return names


def extract_features(K_per_layer: list[torch.Tensor]) -> np.ndarray:
    n_layers = len(K_per_layer)
    max_abs_lh = np.zeros((n_layers, N_KV_HEADS))
    mean_abs_lh = np.zeros((n_layers, N_KV_HEADS))
    std_lh = np.zeros((n_layers, N_KV_HEADS))
    top1_frac_l = np.zeros(n_layers)
    top10_frac_l = np.zeros(n_layers)
    fro_norm_l = np.zeros(n_layers)
    top1_ch_per_lh = np.zeros((n_layers, N_KV_HEADS))
    for L, K in enumerate(K_per_layer):
        Kf = K.float()
        K_abs = Kf.abs()
        max_abs_lh[L] = K_abs.amax(dim=(0, 2)).numpy()
        mean_abs_lh[L] = K_abs.mean(dim=(0, 2)).numpy()
        std_lh[L] = Kf.std(dim=(0, 2)).numpy()
        fro_norm_l[L] = float(Kf.norm())
        ch_energy = (Kf ** 2).sum(dim=0)  # [h, d]
        for h in range(N_KV_HEADS):
            sd, _ = ch_energy[h].sort(descending=True)
            tot = float(sd.sum())
            if tot > 0:
                top1_ch_per_lh[L, h] = float(sd[0]) / tot
        flat = ch_energy.flatten()
        sf, _ = flat.sort(descending=True)
        tot_l = float(sf.sum())
        if tot_l > 0:
            top1_frac_l[L] = float(sf[0]) / tot_l
            top10_frac_l[L] = float(sf[:10].sum()) / tot_l
    return np.concatenate([
        max_abs_lh.flatten(), mean_abs_lh.flatten(),
        std_lh.flatten(), top1_ch_per_lh.flatten(),
        top1_frac_l, top10_frac_l, fro_norm_l,
    ]).astype(np.float32)


def load_K_decode_window(path: Path, half_only: bool = True) -> list[torch.Tensor]:
    """K_pre с capture file [251, 8, 128] per layer, optionally first 125 positions."""
    t = load_file(str(path))
    end = 125 if half_only else 251
    return [t[f"k_pre_l{L}"][:end] for L in range(N_LAYERS) if f"k_pre_l{L}" in t]


def evaluate(y_true, y_pred, label: str) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    try:
        rho = float(spearmanr(y_true, y_pred)[0])
    except Exception:
        rho = float("nan")
    log.info(f"  {label}: MAE={mae:.1f}, RMSE={rmse:.1f}, R²={r2:.3f}, ρ={rho:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": rho}


# ==============================================================
# Models
# ==============================================================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 32)):
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


def train_mlp(Xt, yt, Xv, in_dim: int, epochs: int = 150, lr: float = 1e-3, batch: int = 16) -> np.ndarray:
    model = MLP(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    yt_t = torch.tensor(yt, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt_t[idx]), yt_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(Xv_t).numpy()


class K1DCNN(nn.Module):
    """Per-(layer, head) max_abs over seq → [n_layers*n_heads, head_dim] matrix.
    Conv1d по head_dim direction.
    """
    def __init__(self, n_layer_heads: int, head_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_layer_heads, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # pool to 8
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, n_layer_heads, head_dim]
        f = self.conv(x)
        return self.fc(f).squeeze(-1)


def train_cnn(Xt, yt, Xv, n_layer_heads, head_dim,
              epochs=200, lr=1e-3, batch=8) -> np.ndarray:
    model = K1DCNN(n_layer_heads, head_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    yt_t = torch.tensor(yt, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt_t[idx]), yt_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(Xv_t).numpy()


# ==============================================================
# Data builders
# ==============================================================
def load_fdps(path: Path) -> dict[int, int]:
    fdps = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        fdps[d["problem_idx"]] = d["fdp_token_idx"]
    return fdps


def build_decode_features(captures_dir: Path, fdps: dict, half_only=True) -> tuple:
    X, y, ids = [], [], []
    for cf in sorted(captures_dir.glob("*.safetensors"), key=lambda p: int(p.stem)):
        pid = int(cf.stem)
        if pid not in fdps:
            continue
        K = load_K_decode_window(cf, half_only)
        if len(K) != N_LAYERS:
            continue
        X.append(extract_features(K))
        y.append(fdps[pid])
        ids.append(pid)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), ids


def collect_prefill_K(runner, prompts: list[tuple[int, str]]) -> dict[int, list[torch.Tensor]]:
    """Run bf16 prefill для каждого prompt, capture K_pre per layer."""
    from kvtrace.capture.attention_hooks import install_capture_hooks
    from kvtrace.capture.fp8_sim import QUANT_FNS
    result = {}
    for pid, problem_text in prompts:
        messages = [{"role": "user",
                     "content": f"{problem_text}\n\n{DEFAULT_USER_INSTRUCTION}"}]
        ids = runner._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=True, return_tensors="pt"
        )[0].tolist()
        handle = install_capture_hooks(
            runner._model,
            attention_modules=runner._attention_modules,
            quant_fn=QUANT_FNS["bf16"],
        )
        try:
            with torch.no_grad():
                runner._model(torch.tensor([ids], dtype=torch.long))
            result[pid] = [handle.k_pre[L].squeeze(0).permute(1, 0, 2).clone()
                           for L in range(N_LAYERS)]
        finally:
            handle.remove()
    return result


def kper_to_features(K_per_layer: list[torch.Tensor]) -> np.ndarray:
    return extract_features(K_per_layer)


def kper_to_raw_matrix(K_per_layer: list[torch.Tensor]) -> np.ndarray:
    """[28*8, 128] — per-(layer, head) max_abs vector along head_dim."""
    rows = []
    for K in K_per_layer:
        # K: [seq, h, d]
        for h in range(K.shape[1]):
            rows.append(K[:, h, :].float().abs().amax(dim=0).numpy())
    return np.array(rows, dtype=np.float32)  # [n_layers*n_heads, head_dim]


# ==============================================================
# Experiments
# ==============================================================
def exp_A_prompt_only(fdp_file: Path) -> dict:
    log.info("=" * 60)
    log.info("Exp A: prompt-only baseline (re-running bf16 prefill on 80 prompts)")
    from kvtrace.capture.cpu_runner import CaptureRunner
    fdps = load_fdps(fdp_file)
    prompts = []
    for line in fdp_file.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        prompts.append((d["problem_idx"], d["problem"]))
    log.info(f"  Loading Qwen3-1.7B for prefill...")
    runner = CaptureRunner()
    runner.load_model("Qwen/Qwen3-1.7B")
    log.info(f"  Running prefill on {len(prompts)} problems...")
    K_dict = collect_prefill_K(runner, prompts)
    runner.unload()
    X = np.array([kper_to_features(K_dict[pid]) for pid, _ in prompts], dtype=np.float32)
    y = np.array([fdps[pid] for pid, _ in prompts], dtype=np.float32)
    log.info(f"  X={X.shape}, y={y.shape}")
    return _cv_eval(X, y, label="prompt-only")


def exp_B_two_stage(X: np.ndarray, y: np.ndarray, threshold: int = 200) -> dict:
    log.info("=" * 60)
    log.info(f"Exp B: two-stage (early/late, threshold={threshold})")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    preds_all, true_all = [], []
    for fold, (tr, vl) in enumerate(kf.split(X)):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])
        Xv = scaler.transform(X[vl])
        yt = y[tr]
        yv = y[vl]
        # Stage 1: classifier
        zt = (yt >= threshold).astype(int)
        zv = (yv >= threshold).astype(int)
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
        clf.fit(Xt, zt)
        zv_pred = clf.predict(Xv)
        clf_acc = float(accuracy_score(zv, zv_pred))
        # Stage 2: per-class regressor
        rgs = {}
        for cls in [0, 1]:
            mask = (zt == cls)
            if mask.sum() < 5:
                rgs[cls] = None
                continue
            r = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                          learning_rate=0.05, random_state=42)
            r.fit(Xt[mask], yt[mask])
            rgs[cls] = r
        # Combined predictions
        pred = np.zeros(len(vl))
        for i, c in enumerate(zv_pred):
            if rgs[c] is not None:
                pred[i] = rgs[c].predict(Xv[i:i + 1])[0]
            else:
                pred[i] = yt[zt == c].mean() if (zt == c).any() else yt.mean()
        m = evaluate(yv, pred, f"two-stage (cls_acc={clf_acc:.2f})")
        m["classifier_accuracy"] = clf_acc
        results.append(m)
        preds_all.extend(pred.tolist())
        true_all.extend(yv.tolist())
    return _agg(results, "two-stage", preds_all, true_all)


def exp_C_feature_importance(X: np.ndarray, y: np.ndarray, top: int = 20) -> dict:
    log.info("=" * 60)
    log.info("Exp C: feature importance from GBM")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                    learning_rate=0.05, random_state=42)
    gbm.fit(Xs, y)
    importances = gbm.feature_importances_
    names = feature_names()
    idx = np.argsort(-importances)[:top]
    top_feats = [(names[i], float(importances[i])) for i in idx]
    log.info(f"  Top-{top} features:")
    for n, im in top_feats:
        log.info(f"    {n:30s}  {im:.5f}")
    # Group by metric type
    by_metric: dict[str, float] = {}
    by_layer = np.zeros(N_LAYERS)
    for i, imp in enumerate(importances):
        nm = names[i]
        mt = nm.split("_L")[0]
        by_metric[mt] = by_metric.get(mt, 0) + float(imp)
        # layer
        if "_L" in nm:
            try:
                L = int(nm.split("_L")[1].split("_")[0])
                by_layer[L] += imp
            except Exception:
                pass
    log.info("  Aggregated importance by metric:")
    for mt, v in sorted(by_metric.items(), key=lambda kv: -kv[1]):
        log.info(f"    {mt:18s}  {v:.4f}")
    log.info("  Top-5 most important layers:")
    top_layers = np.argsort(-by_layer)[:5]
    for L in top_layers:
        log.info(f"    L{L:2d}: {by_layer[L]:.4f}")
    return {
        "top_features": top_feats,
        "importance_by_metric": by_metric,
        "importance_by_layer": by_layer.tolist(),
        "top_5_layers": top_layers.tolist(),
    }


def exp_D_cnn(X_raw: np.ndarray, y: np.ndarray) -> dict:
    """X_raw: [n_problems, n_layer_heads=224, head_dim=128]"""
    log.info("=" * 60)
    log.info("Exp D: 1D-CNN на raw K-matrices [n, 224, 128]")
    log.info(f"  X_raw shape: {X_raw.shape}")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    n_lh = X_raw.shape[1]
    head_dim = X_raw.shape[2]
    results = []
    preds_all, true_all = [], []
    for fold, (tr, vl) in enumerate(kf.split(X_raw)):
        # Scale per-row (per problem, per layer-head)
        mean = X_raw[tr].mean(axis=(0, 2), keepdims=True)
        std = X_raw[tr].std(axis=(0, 2), keepdims=True) + 1e-8
        Xt = (X_raw[tr] - mean) / std
        Xv = (X_raw[vl] - mean) / std
        pred = train_cnn(Xt, y[tr], Xv, n_lh, head_dim)
        m = evaluate(y[vl], pred, f"CNN fold {fold + 1}")
        results.append(m)
        preds_all.extend(pred.tolist())
        true_all.extend(y[vl].tolist())
    return _agg(results, "CNN", preds_all, true_all)


def exp_E_cross_quant(X_train: np.ndarray, y_train: np.ndarray,
                       captures_root: Path, fdp_root: Path) -> dict:
    log.info("=" * 60)
    log.info("Exp E: cross-quant transfer (train fp8_e4m3 -> test fp8_e5m2 / hqq_int4)")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                    learning_rate=0.05, random_state=42)
    gbm.fit(Xs, y_train)
    results = {}
    for tgt in ["fp8_e5m2", "hqq_int4"]:
        log.info(f"  Target quant: {tgt}")
        fdps = load_fdps(fdp_root / f"qwen3-1.7b_{tgt}.jsonl")
        X_test, y_test, _ = build_decode_features(captures_root / f"{tgt}_tf",
                                                   fdps, half_only=True)
        if len(X_test) == 0:
            log.warning(f"    no captures found for {tgt}, skip")
            continue
        Xt_s = scaler.transform(X_test)
        pred = gbm.predict(Xt_s)
        m = evaluate(y_test, pred, f"cross-quant -> {tgt}")
        results[tgt] = m
    return results


def extract_features_head_pooled(K_per_layer: list[torch.Tensor]) -> np.ndarray:
    """Head-agnostic features: для cross-model где число KV-heads различается.
    [n_layers, 7 stats] = 196 features (для 28 слоёв).
    """
    feats = []
    for L, K in enumerate(K_per_layer):
        Kf = K.float()
        K_abs = Kf.abs()
        # Per-layer stats (pooled over h, seq, d)
        max_abs = float(K_abs.amax())
        mean_abs = float(K_abs.mean())
        std = float(Kf.std())
        fro_norm = float(Kf.norm())
        # Channel concentration (pool over heads): per-channel energy, flat
        ch_energy_flat = (Kf ** 2).sum(dim=0).flatten()  # [h * d]
        sf, _ = ch_energy_flat.sort(descending=True)
        tot = float(sf.sum())
        top1_frac = float(sf[0]) / tot if tot > 0 else 0.0
        top10_frac = float(sf[:10].sum()) / tot if tot > 0 else 0.0
        # Outlier-amplitude ratio: max_abs / mean_abs
        amp_ratio = max_abs / max(mean_abs, 1e-9)
        feats.extend([max_abs, mean_abs, std, fro_norm,
                      top1_frac, top10_frac, amp_ratio])
    return np.array(feats, dtype=np.float32)


def exp_F_cross_model(fdp_root: Path) -> dict:
    """Cross-model: re-train GBM on HEAD-POOLED Qwen3-1.7B prefill features,
    transfer to DeepSeek-R1-Distill-Qwen-1.5B prefill features.
    """
    log.info("=" * 60)
    log.info("Exp F: cross-model transfer (head-pooled features) — Qwen3-1.7B -> DeepSeek-R1-Distill-Qwen-1.5B")
    from kvtrace.capture.cpu_runner import CaptureRunner

    # ===== Source: Qwen3-1.7B prefill, head-pooled features =====
    src_fdp = fdp_root / "qwen3-1.7b_fp8_e4m3.jsonl"
    src_fdps = load_fdps(src_fdp)
    src_prompts = []
    for line in src_fdp.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        src_prompts.append((d["problem_idx"], d["problem"]))
    log.info(f"  [source] loading Qwen3-1.7B and running prefill on {len(src_prompts)} prompts...")
    runner = CaptureRunner()
    runner.load_model("Qwen/Qwen3-1.7B")
    K_src = collect_prefill_K(runner, src_prompts)
    runner.unload()
    X_src = np.array([extract_features_head_pooled(K_src[pid]) for pid, _ in src_prompts],
                     dtype=np.float32)
    y_src = np.array([src_fdps[pid] for pid, _ in src_prompts], dtype=np.float32)
    log.info(f"  [source] X={X_src.shape}, y range=[{y_src.min():.0f},{y_src.max():.0f}]")

    # ===== Target: DeepSeek-R1-Distill, head-pooled features =====
    tgt_fdp = fdp_root / "deepseek-r1-distill-qwen-1.5b_fp8_e4m3.jsonl"
    tgt_fdps = load_fdps(tgt_fdp)
    tgt_prompts = []
    for line in tgt_fdp.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        tgt_prompts.append((d["problem_idx"], d["problem"]))
    log.info(f"  [target] loading DeepSeek-R1-Distill-Qwen-1.5B and running prefill on {len(tgt_prompts)} prompts...")
    runner = CaptureRunner()
    runner.load_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    K_tgt = collect_prefill_K(runner, tgt_prompts)
    runner.unload()
    X_tgt = np.array([extract_features_head_pooled(K_tgt[pid]) for pid, _ in tgt_prompts],
                     dtype=np.float32)
    y_tgt = np.array([tgt_fdps[pid] for pid, _ in tgt_prompts], dtype=np.float32)
    log.info(f"  [target] X={X_tgt.shape}, y range=[{y_tgt.min():.0f},{y_tgt.max():.0f}]")

    # ===== Train on source prefill, test on target prefill =====
    # NB: prefill features only — cross-model honest, no decode leak.
    scaler = StandardScaler()
    Xs_s = scaler.fit_transform(X_src)
    Xt_s = scaler.transform(X_tgt)
    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                    learning_rate=0.05, random_state=42)
    gbm.fit(Xs_s, y_src)
    pred = gbm.predict(Xt_s)
    m_cross = evaluate(y_tgt, pred, "cross-model GBM (prefill head-pooled)")
    # ===== Source self-CV для baseline (что максимально достижимо на prefill) =====
    log.info("  [source] 5-fold CV on source prefill head-pooled features (upper bound)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    src_cv = []
    for fold, (tr, vl) in enumerate(kf.split(X_src)):
        s = StandardScaler()
        gbm2 = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                          learning_rate=0.05, random_state=42)
        gbm2.fit(s.fit_transform(X_src[tr]), y_src[tr])
        p = gbm2.predict(s.transform(X_src[vl]))
        src_cv.append(evaluate(y_src[vl], p, f"  src self-fold{fold + 1}"))
    src_self = _agg(src_cv, "source self-CV (prefill head-pooled, GBM)", None, None)
    return {
        "cross_model_transfer": m_cross,
        "source_self_cv_prefill_pooled": src_self,
    }


# ==============================================================
# Helpers
# ==============================================================
def _cv_eval(X: np.ndarray, y: np.ndarray, label: str = "") -> dict:
    """5-fold CV для GBM + MLP."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gbm_metrics, mlp_metrics = [], []
    preds_gbm_all, preds_mlp_all, true_all = [], [], []
    for fold, (tr, vl) in enumerate(kf.split(X)):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[tr])
        Xv = scaler.transform(X[vl])
        # GBM
        gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                        learning_rate=0.05, random_state=42)
        gbm.fit(Xt, y[tr])
        gp = gbm.predict(Xv)
        gbm_metrics.append(evaluate(y[vl], gp, f"GBM fold{fold + 1} ({label})"))
        # MLP
        mp = train_mlp(Xt, y[tr], Xv, in_dim=X.shape[1])
        mlp_metrics.append(evaluate(y[vl], mp, f"MLP fold{fold + 1} ({label})"))
        preds_gbm_all.extend(gp.tolist())
        preds_mlp_all.extend(mp.tolist())
        true_all.extend(y[vl].tolist())
    return {
        "gbm": _agg(gbm_metrics, "GBM " + label, preds_gbm_all, true_all),
        "mlp": _agg(mlp_metrics, "MLP " + label, preds_mlp_all, true_all),
    }


def _agg(fold_metrics: list[dict], label: str,
         preds: list = None, true: list = None) -> dict:
    out = {}
    for m in ["mae", "rmse", "r2", "spearman"]:
        vals = [fm[m] for fm in fold_metrics if not (m == "spearman" and np.isnan(fm[m]))]
        out[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{m}_std"] = float(np.std(vals)) if vals else float("nan")
    if preds is not None:
        out["predictions"] = preds
    if true is not None:
        out["true"] = true
    log.info(f"  >>> {label}: MAE={out['mae_mean']:.1f}±{out['mae_std']:.1f}, "
             f"R²={out['r2_mean']:.3f}±{out['r2_std']:.3f}, "
             f"ρ={out['spearman_mean']:.3f}±{out['spearman_std']:.3f}")
    return out


# ==============================================================
# Main
# ==============================================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--captures-root", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b"))
    p.add_argument("--fdp-root", type=Path, default=Path("outputs/fdps"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/kv_capture/qwen3-1.7b/analysis/fdp_predictor_extended.json"))
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["A", "B", "C", "D", "E", "F"])
    args = p.parse_args()

    all_results = {}

    # Build base dataset: decode-window features for fp8_e4m3 (Phase 22 data)
    log.info("Building base dataset (decode-window features for fp8_e4m3)...")
    fdps_e4m3 = load_fdps(args.fdp_root / "qwen3-1.7b_fp8_e4m3.jsonl")
    X_dec, y_dec, ids_dec = build_decode_features(
        args.captures_root / "fp8_e4m3_tf", fdps_e4m3, half_only=True
    )
    log.info(f"  X_dec={X_dec.shape}, y_dec=range[{y_dec.min():.0f},{y_dec.max():.0f}]")
    all_results["dataset"] = {
        "n": int(X_dec.shape[0]),
        "n_features": int(X_dec.shape[1]),
        "fdp_min": float(y_dec.min()),
        "fdp_max": float(y_dec.max()),
        "fdp_mean": float(y_dec.mean()),
        "fdp_std": float(y_dec.std()),
    }

    # Baseline (for comparison): decode-window GBM/MLP (Phase 22 redo)
    log.info("=" * 60)
    log.info("Baseline (Phase 22 redo): decode-window features, GBM + MLP")
    all_results["baseline_decode_window"] = _cv_eval(X_dec, y_dec, label="decode")

    # Resilient save: after each experiment
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2))

    save()
    if "A" not in args.skip:
        try:
            all_results["A_prompt_only"] = exp_A_prompt_only(args.fdp_root / "qwen3-1.7b_fp8_e4m3.jsonl")
        except Exception as e:
            log.exception(f"Exp A failed: {e}")
            all_results["A_prompt_only_error"] = str(e)
        save()
    if "B" not in args.skip:
        try:
            all_results["B_two_stage"] = exp_B_two_stage(X_dec, y_dec, threshold=200)
        except Exception as e:
            log.exception(f"Exp B failed: {e}")
            all_results["B_two_stage_error"] = str(e)
        save()
    if "C" not in args.skip:
        try:
            all_results["C_feature_importance"] = exp_C_feature_importance(X_dec, y_dec)
        except Exception as e:
            log.exception(f"Exp C failed: {e}")
            all_results["C_feature_importance_error"] = str(e)
        save()
    if "D" not in args.skip:
        try:
            log.info("Loading raw K-matrices for CNN...")
            X_raw = []
            for cf in sorted((args.captures_root / "fp8_e4m3_tf").glob("*.safetensors"),
                             key=lambda p: int(p.stem)):
                pid = int(cf.stem)
                if pid not in fdps_e4m3:
                    continue
                K = load_K_decode_window(cf, half_only=True)
                if len(K) != N_LAYERS:
                    continue
                X_raw.append(kper_to_raw_matrix(K))
            X_raw = np.array(X_raw, dtype=np.float32)
            all_results["D_cnn"] = exp_D_cnn(X_raw, y_dec)
        except Exception as e:
            log.exception(f"Exp D failed: {e}")
            all_results["D_cnn_error"] = str(e)
        save()
    if "E" not in args.skip:
        try:
            all_results["E_cross_quant"] = exp_E_cross_quant(X_dec, y_dec,
                                                              args.captures_root, args.fdp_root)
        except Exception as e:
            log.exception(f"Exp E failed: {e}")
            all_results["E_cross_quant_error"] = str(e)
        save()
    if "F" not in args.skip:
        try:
            all_results["F_cross_model"] = exp_F_cross_model(args.fdp_root)
        except Exception as e:
            log.exception(f"Exp F failed: {e}")
            all_results["F_cross_model_error"] = str(e)
        save()
    log.info(f"Saved {args.output}")

    # Comparison plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        labels, r2_means, r2_stds = [], [], []

        def _push(label, d):
            if d and "r2_mean" in d:
                labels.append(label)
                r2_means.append(d["r2_mean"])
                r2_stds.append(d.get("r2_std", 0))

        b = all_results["baseline_decode_window"]
        _push("Baseline GBM\n(decode window)", b.get("gbm", {}))
        _push("Baseline MLP\n(decode window)", b.get("mlp", {}))
        if "A" not in args.skip:
            a = all_results["A_prompt_only"]
            _push("A: Prompt-only GBM", a.get("gbm", {}))
            _push("A: Prompt-only MLP", a.get("mlp", {}))
        if "B" not in args.skip:
            _push("B: Two-stage GBM", all_results["B_two_stage"])
        if "D" not in args.skip:
            _push("D: 1D-CNN raw K", all_results["D_cnn"])
        if "E" not in args.skip:
            for q, d in all_results["E_cross_quant"].items():
                _push(f"E: -> {q}", {"r2_mean": d["r2"]})
        if "F" not in args.skip and "F_cross_model" in all_results:
            f_res = all_results["F_cross_model"]
            if isinstance(f_res, dict) and "cross_model_transfer" in f_res:
                _push("F: Qwen3->DeepSeek\n(prefill, head-pool)",
                      {"r2_mean": f_res["cross_model_transfer"]["r2"]})
                _push("F: Qwen3 self-CV\n(prefill, head-pool)",
                      f_res["source_self_cv_prefill_pooled"])

        x = np.arange(len(labels))
        colors = ["tab:blue"] * 2 + ["tab:green"] * 2 + ["tab:orange"] + ["tab:purple"]
        colors += ["tab:red"] * (len(labels) - len(colors))
        ax.bar(x, r2_means, yerr=r2_stds, color=colors[:len(labels)], capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("R² (cross-validated or held-out)")
        ax.set_title("FDP regression: comparison of 6 approaches")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        for xi, r in zip(x, r2_means):
            ax.text(xi, max(r, 0) + 0.02, f"{r:.2f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(args.output.parent / "plots" / "fdp_predictor_comparison.png", dpi=130)
        log.info(f"Saved comparison plot")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
