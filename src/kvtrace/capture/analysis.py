"""Analysis utilities for KV-matrix captures (Phase 7).

Все функции принимают `CaptureData` и возвращают pure-Python / torch / numpy
объекты. Никакого I/O — это для лёгкого тестирования.

Note about Q layout: q в captures имеет shape [W, hidden] (пост-q_proj,
до RoPE и до reshape в multi-head). Для analytical reshape в
[W, num_heads, head_dim]: `q.view(W, num_heads, head_dim)`.

K/V — post-RoPE (взяты из cache после attention's RoPE-применения),
shape [W, num_kv_heads, head_dim].

Attention map analysis НЕ включён здесь потому что требует совмещения
post-RoPE K с post-RoPE Q — а Q у нас pre-RoPE. Если нужно — отдельный
hook на post-RoPE Q (не в этой итерации).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from kvtrace.capture.storage import CaptureData, load_capture


@dataclass(frozen=True)
class LayerHeadError:
    """Quant error per (layer, head): ||K_pre - K_post||_F / ||K_pre||_F."""
    k_relative_error: torch.Tensor  # [n_layers, num_kv_heads], fp32
    v_relative_error: torch.Tensor


def compute_relative_quant_error(cap: CaptureData) -> LayerHeadError:
    """Frobenius relative error per (layer, head).

    Per head: ||K_p[:, head, :] - K_q[:, head, :]||_F / ||K_p[:, head, :]||_F.
    For bf16 (no quantization) этот error будет 0.
    """
    n_layers = len(cap.k_pre)
    num_kv_heads = cap.k_pre[0].shape[1]
    eps = 1e-12

    k_err = torch.zeros(n_layers, num_kv_heads)
    v_err = torch.zeros(n_layers, num_kv_heads)

    for layer in range(n_layers):
        k_pre = cap.k_pre[layer].float()
        k_post = cap.k_post[layer].float()
        v_pre = cap.v_pre[layer].float()
        v_post = cap.v_post[layer].float()

        for head in range(num_kv_heads):
            kp = k_pre[:, head, :]
            kq = k_post[:, head, :]
            k_err[layer, head] = (kp - kq).norm() / (kp.norm() + eps)

            vp = v_pre[:, head, :]
            vq = v_post[:, head, :]
            v_err[layer, head] = (vp - vq).norm() / (vp.norm() + eps)

    return LayerHeadError(k_relative_error=k_err, v_relative_error=v_err)


def compute_logits_kl_at_fdp(cap_a: CaptureData, cap_b: CaptureData) -> dict:
    """KL(logits_a || logits_b) at FDP position.

    Requires same window (same window_start + window_end). Returns dict with
    top1 match, top5 overlap, KL и JS дивергенциями.
    """
    if cap_a.meta["window_start"] != cap_b.meta["window_start"]:
        raise ValueError(
            f"Captures have different windows: "
            f"{cap_a.meta['window_start']} vs {cap_b.meta['window_start']}. "
            f"Cannot compare logits at same position."
        )
    fdp_in_window = cap_a.meta["fdp_token_idx"] - cap_a.meta["window_start"]
    W = cap_a.meta["W"]
    if fdp_in_window < 0 or fdp_in_window >= W:
        raise ValueError(f"FDP {cap_a.meta['fdp_token_idx']} outside window")

    log_a = cap_a.logits[fdp_in_window].float()
    log_b = cap_b.logits[fdp_in_window].float()

    p_a = torch.softmax(log_a, dim=-1)
    p_b = torch.softmax(log_b, dim=-1)

    eps = 1e-12
    kl_ab = float((p_a * torch.log((p_a + eps) / (p_b + eps))).sum())
    kl_ba = float((p_b * torch.log((p_b + eps) / (p_a + eps))).sum())
    js = 0.5 * (kl_ab + kl_ba)

    top1_a = int(log_a.argmax())
    top1_b = int(log_b.argmax())
    top5_a = log_a.topk(5).indices.tolist()
    top5_b = log_b.topk(5).indices.tolist()

    return {
        "fdp_token_idx": cap_a.meta["fdp_token_idx"],
        "top1_a": top1_a,
        "top1_b": top1_b,
        "top1_match": top1_a == top1_b,
        "top5_overlap": len(set(top5_a) & set(top5_b)),
        "kl_a_to_b": kl_ab,
        "kl_b_to_a": kl_ba,
        "js_divergence": js,
    }


def compute_kv_value_stats_per_layer(cap: CaptureData) -> list[dict]:
    """Per-layer statistics: max abs, mean abs, std, outlier counts.

    Outliers thresholds:
      - 448 (e4m3 max) — values that e4m3 would clip / overflow to NaN
      - 57344 (e5m2 max) — same for e5m2
    """
    stats = []
    for layer in range(len(cap.k_pre)):
        k = cap.k_pre[layer].float()
        v = cap.v_pre[layer].float()
        stats.append({
            "layer": layer,
            "k_max_abs": float(k.abs().max()),
            "k_mean_abs": float(k.abs().mean()),
            "k_std": float(k.std()),
            "k_outliers_pct_448": float((k.abs() > 448).float().mean() * 100),
            "k_outliers_pct_57344": float((k.abs() > 57344).float().mean() * 100),
            "v_max_abs": float(v.abs().max()),
            "v_mean_abs": float(v.abs().mean()),
            "v_std": float(v.std()),
            "v_outliers_pct_448": float((v.abs() > 448).float().mean() * 100),
            "v_outliers_pct_57344": float((v.abs() > 57344).float().mean() * 100),
        })
    return stats


def load_captures_for_quant(root: Path, quant: str, mode: str = "tf") -> list[CaptureData]:
    """Load all capture-файлы для (quant, mode), отсортированы по problem_id."""
    folder = root / f"{quant}_{mode}"
    if not folder.exists():
        return []
    files = sorted(folder.glob("*.safetensors"), key=lambda p: int(p.stem))
    return [load_capture(f) for f in files]


def captures_share_window(cap_a: CaptureData, cap_b: CaptureData) -> bool:
    """True если captures имеют одинаковый абсолютный window (одинаковые FDP)."""
    return (
        cap_a.meta["window_start"] == cap_b.meta["window_start"]
        and cap_a.meta["window_end"] == cap_b.meta["window_end"]
    )


def align_captures_by_absolute_position(
    cap_a: CaptureData, cap_b: CaptureData
) -> tuple[slice, slice] | None:
    """Найти пересечение окон двух captures как (slice_a, slice_b).

    Возвращает None если окна не пересекаются.

    Использование:
        sl_a, sl_b = align_captures_by_absolute_position(cap_a, cap_b)
        k_a_aligned = cap_a.k_pre[layer][sl_a]  # same absolute positions
        k_b_aligned = cap_b.k_pre[layer][sl_b]
    """
    a_start, a_end = cap_a.meta["window_start"], cap_a.meta["window_end"]
    b_start, b_end = cap_b.meta["window_start"], cap_b.meta["window_end"]
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    if overlap_start >= overlap_end:
        return None
    return (
        slice(overlap_start - a_start, overlap_end - a_start),
        slice(overlap_start - b_start, overlap_end - b_start),
    )
