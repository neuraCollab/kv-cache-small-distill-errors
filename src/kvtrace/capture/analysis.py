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


def logit_kl_trajectory(cap_a: CaptureData, cap_b: CaptureData) -> torch.Tensor:
    """KL(p_a || p_b) at every position in the overlapping absolute-position window.

    Возвращает Tensor[overlap_length] где [i] = KL дивергенция логит-распределений
    в i-й позиции overlapping region (absolute coords пересечения окон).

    В AR-mode может различаться actual logits.shape[0] vs meta["W"] (early EOS
    или off-by-one в run_ar). Используем фактические длины тензоров и
    выравниваем по absolute position.

    Raises ValueError если окна вообще не пересекаются.
    """
    a_start = cap_a.meta["window_start"]
    b_start = cap_b.meta["window_start"]
    # Use actual tensor lengths (may be < meta W if AR early EOS)
    a_end = a_start + cap_a.logits.shape[0]
    b_end = b_start + cap_b.logits.shape[0]
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    if overlap_start >= overlap_end:
        raise ValueError(
            f"Captures have no overlapping window: "
            f"a=[{a_start},{a_end}) b=[{b_start},{b_end})"
        )
    sl_a = slice(overlap_start - a_start, overlap_end - a_start)
    sl_b = slice(overlap_start - b_start, overlap_end - b_start)
    log_a = cap_a.logits[sl_a].float()
    log_b = cap_b.logits[sl_b].float()
    p_a = torch.softmax(log_a, dim=-1)
    p_b = torch.softmax(log_b, dim=-1)
    eps = 1e-12
    kl = (p_a * (torch.log(p_a + eps) - torch.log(p_b + eps))).sum(dim=-1)
    return kl  # [overlap_length]


def bf16_margin_trajectory(cap_bf16: CaptureData) -> torch.Tensor:
    """Margin (top-1 logit - top-2 logit) at every position.

    Малый margin → модель не уверена в выборе токена → лёгко перекинуть
    argmax квант-шумом. Большой margin → робастность к шуму.

    Возвращает Tensor[W].
    """
    logits = cap_bf16.logits.float()  # [W, vocab]
    top2 = logits.topk(2, dim=-1).values  # [W, 2]
    return (top2[:, 0] - top2[:, 1])  # [W]


def per_position_kv_quant_noise(cap: CaptureData) -> dict[str, torch.Tensor]:
    """Per-(layer, position) относительная L2-норма квант-шума ΔK, ΔV.

    Для каждого слоя ℓ и позиции t:
        k_noise[ℓ, t] = ||K_post[t, :, :] - K_pre[t, :, :]||_2 / (||K_pre[t, :, :]||_2 + eps)

    Возвращает {"k_noise": [n_layers, W], "v_noise": [n_layers, W]}.
    Для bf16 capture везде нули.
    """
    n_layers = len(cap.k_pre)
    W = cap.k_pre[0].shape[0]
    eps = 1e-12
    k_noise = torch.zeros(n_layers, W)
    v_noise = torch.zeros(n_layers, W)
    for layer in range(n_layers):
        kp = cap.k_pre[layer].float()  # [W, num_kv_heads, head_dim]
        kq = cap.k_post[layer].float()
        vp = cap.v_pre[layer].float()
        vq = cap.v_post[layer].float()
        # L2 norm per position (over heads × dim)
        kp_norm = kp.flatten(1).norm(dim=1)  # [W]
        kq_diff = (kq - kp).flatten(1).norm(dim=1)
        k_noise[layer] = kq_diff / (kp_norm + eps)
        vp_norm = vp.flatten(1).norm(dim=1)
        vq_diff = (vq - vp).flatten(1).norm(dim=1)
        v_noise[layer] = vq_diff / (vp_norm + eps)
    return {"k_noise": k_noise, "v_noise": v_noise}


def top_outlier_channels(
    cap: CaptureData, threshold: float = 448.0, top_n_per_layer: int = 5
) -> list[dict]:
    """Найти каналы (layer, kv_head, head_dim_channel) где |K| или |V| превышает threshold.

    threshold = 448 (e4m3 max) показывает каналы, которые e4m3-квант обрезает.
    threshold = 57344 (e5m2 max) — то же для e5m2.

    Возвращает list[{layer, kind: 'k'|'v', max_channels: [(head, channel, max_abs)]}].
    """
    results = []
    for layer in range(len(cap.k_pre)):
        for kind in ("k", "v"):
            tensor = (cap.k_pre[layer] if kind == "k" else cap.v_pre[layer]).float()
            # tensor shape: [W, num_kv_heads, head_dim]
            # max abs per (head, channel): max over W
            max_per_channel = tensor.abs().amax(dim=0)  # [num_kv_heads, head_dim]
            num_heads, head_dim = max_per_channel.shape
            flat = max_per_channel.flatten()
            n_outliers = int((flat > threshold).sum())
            # Топ-N каналов по max abs
            top_vals, top_idx = flat.topk(min(top_n_per_layer, len(flat)))
            channels = []
            for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
                head = idx // head_dim
                ch = idx % head_dim
                channels.append({"head": head, "channel": ch, "max_abs": v})
            results.append({
                "layer": layer,
                "kind": kind,
                "n_channels_above_threshold": n_outliers,
                "top_channels": channels,
            })
    return results


def attention_shift_kl(cap: CaptureData) -> torch.Tensor:
    """Per-(layer, position) KL между attention(Q, K_pre) и attention(Q, K_post).

    Quant noise в K вызывает сдвиг attention-распределения. Эта функция считает
    насколько именно — для каждого слоя и каждой query-позиции KL between two
    attention distributions over all key positions.

    Требует q_post_rope (RoPE applied). Если capture без него (legacy bf16 от
    старого кода), raises ValueError.

    Returns Tensor[n_layers, W] — KL per (layer, query_position) usrедненный по headam
    (для GQA repeat KV heads до num_q_heads).

    Note: Q_pre-RoPE attention с K_post-RoPE даёт неверный результат (mixed
    RoPE state). Q_post_rope обязателен.
    """
    if cap.q_post_rope is None:
        raise ValueError(
            "capture has no q_post_rope — re-capture with newer code "
            "(see install_capture_hooks rope patch)"
        )

    n_layers = len(cap.q_post_rope)
    W = cap.q_post_rope[0].shape[0]
    head_dim = cap.q_post_rope[0].shape[-1]
    num_q_heads = cap.q_post_rope[0].shape[1]
    num_kv_heads = cap.k_pre[0].shape[1]
    repeat = num_q_heads // num_kv_heads  # GQA factor

    eps = 1e-12
    scale = 1.0 / (head_dim ** 0.5)

    # Causal mask: query i can only attend to key 0..i
    causal_mask = torch.triu(torch.ones(W, W, dtype=torch.bool), diagonal=1)

    out = torch.zeros(n_layers, W)
    for layer in range(n_layers):
        q = cap.q_post_rope[layer].float()  # [W, num_q_heads, head_dim]
        k_pre = cap.k_pre[layer].float()    # [W, num_kv_heads, head_dim]
        k_post = cap.k_post[layer].float()
        # Repeat K for GQA: [W, num_kv_heads, head_dim] → [W, num_q_heads, head_dim]
        if repeat > 1:
            k_pre = k_pre.repeat_interleave(repeat, dim=1)
            k_post = k_post.repeat_interleave(repeat, dim=1)

        # Compute Q @ K^T per head, then softmax
        # q: [W, H, D], k: [W, H, D]
        # scores[h, i, j] = q[i, h] · k[j, h]
        # → [W, H, W] = q_einsum
        q_t = q.transpose(0, 1)  # [H, W, D]
        kp_t = k_pre.transpose(0, 1)  # [H, W, D]
        kq_t = k_post.transpose(0, 1)
        scores_pre = torch.matmul(q_t, kp_t.transpose(-1, -2)) * scale  # [H, W, W]
        scores_post = torch.matmul(q_t, kq_t.transpose(-1, -2)) * scale
        # Causal mask
        scores_pre = scores_pre.masked_fill(causal_mask, float("-inf"))
        scores_post = scores_post.masked_fill(causal_mask, float("-inf"))
        a_pre = torch.softmax(scores_pre, dim=-1)   # [H, W, W]
        a_post = torch.softmax(scores_post, dim=-1)
        # KL per (head, query): Σ_k a_pre log(a_pre/a_post)
        kl = (a_pre * (torch.log(a_pre + eps) - torch.log(a_post + eps))).sum(dim=-1)  # [H, W]
        # Mean over heads
        out[layer] = kl.mean(dim=0)
    return out


def outlier_channel_impact(
    cap: CaptureData, top_n_channels: int = 10
) -> dict:
    """Какая доля общего K-quant-noise приходится на top-N outlier-каналов?

    Для каждого слоя ℓ:
      total_noise = ||K_post - K_pre||_F²
      Identifies каналы (kv_head, channel) с максимальным per-channel вкладом
      в noise = ||(K_post - K_pre)[:, h, c]||² per (h, c).
      Сортирует, возвращает top-N + их fraction of total.

    Returns list[dict] per layer:
      {layer, total_noise, top_channels: [{head, channel, noise, frac}],
       top_n_fraction: total fraction explained by top_n}
    """
    results = []
    for layer in range(len(cap.k_pre)):
        kp = cap.k_pre[layer].float()
        kq = cap.k_post[layer].float()
        delta = kq - kp  # [W, num_kv_heads, head_dim]
        # Per-(head, channel) squared noise summed over W positions
        per_channel = (delta ** 2).sum(dim=0)  # [num_kv_heads, head_dim]
        total = float(per_channel.sum())
        if total < 1e-12:
            results.append({"layer": layer, "total_noise": 0.0,
                            "top_channels": [], "top_n_fraction": 0.0})
            continue
        flat = per_channel.flatten()
        num_heads, head_dim = per_channel.shape
        top_vals, top_idx = flat.topk(min(top_n_channels, len(flat)))
        channels = []
        for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
            head = idx // head_dim
            ch = idx % head_dim
            channels.append({
                "head": head, "channel": ch,
                "noise": float(v),
                "frac": float(v / total),
            })
        top_n_fraction = sum(c["frac"] for c in channels)
        results.append({
            "layer": layer,
            "total_noise": total,
            "top_channels": channels,
            "top_n_fraction": top_n_fraction,
        })
    return results


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
