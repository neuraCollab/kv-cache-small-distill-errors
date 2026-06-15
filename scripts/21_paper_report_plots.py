"""Дополнительные plot'ы для report.tex: cross-model bar + concentration→recovery."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out_dir = Path("docs/paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Live cross-model results (median per-layer over fresh probs)
    models = [
        ("Qwen3-1.7B",                "outputs/kv_capture/qwen3-1.7b/analysis/paper_validation_live_n20.json",                    "tab:blue"),
        ("DeepSeek-R1-\nDistill-Qwen-1.5B", "outputs/kv_capture/deepseek-r1-distill-qwen-1.5b/analysis/paper_validation_live.json", "tab:cyan"),
        ("Qwen2.5-1.5B",              "outputs/kv_capture/qwen2.5-1.5b/analysis/paper_validation_live.json",                       "tab:purple"),
        ("SmolLM2-1.7B",              "outputs/kv_capture/smollm2-1.7b/analysis/paper_validation_live.json",                        "tab:orange"),
    ]
    rows = []
    for label, path, color in models:
        d = json.load(open(path))["per_quant"]["fp8_e4m3"]
        rows.append({
            "label": label,
            "top10_median": d["top10_ch_noise_frac_median"] * 100,
            "defense_pct": -d["defense_change_pct"],  # положительный = recovery
            "color": color,
        })

    # ===== Figure 1: cross-model side-by-side =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    bars1 = ax1.bar(x, [r["top10_median"] for r in rows], color=[r["color"] for r in rows])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, fontsize=9)
    ax1.set_ylabel("Top-10 channel noise fraction, %")
    ax1.set_title("FP8 e4m3 noise concentration (live, prefill K)")
    ax1.axhline(51.7, color="red", linestyle="--", alpha=0.6, label="Paper claim (51.7%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for b, r in zip(bars1, rows):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.7,
                 f'{r["top10_median"]:.1f}%', ha='center', fontsize=9)

    bars2 = ax2.bar(x, [r["defense_pct"] for r in rows], color=[r["color"] for r in rows])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0, fontsize=9)
    ax2.set_ylabel("K-error recovery, % (positive = better)")
    ax2.set_title("Per-channel defense efficacy (N=10)")
    ax2.axhline(34, color="red", linestyle="--", alpha=0.6, label="Paper claim (-34%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    for b, r in zip(bars2, rows):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                 f'{r["defense_pct"]:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_cross_model.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir/'fig_cross_model.png'}")

    # ===== Figure 2: concentration → recovery linear relationship =====
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = np.array([r["top10_median"] for r in rows])
    ys = np.array([r["defense_pct"] for r in rows])
    # Add paper's n=80 measurement too
    xs_full = np.append(xs, 51.7)
    ys_full = np.append(ys, 34.1)
    labels_full = labels + ["Qwen3-1.7B\n(paper n=80 lab)"]
    colors_full = [r["color"] for r in rows] + ["tab:red"]
    for xi, yi, li, ci in zip(xs_full, ys_full, labels_full, colors_full):
        ax.scatter(xi, yi, s=180, color=ci, edgecolors="black", linewidths=1, zorder=3)
        ax.annotate(li, (xi, yi), xytext=(7, 7), textcoords="offset points", fontsize=9)
    # Linear fit through origin
    slope = (xs_full * ys_full).sum() / (xs_full * xs_full).sum()
    xx = np.linspace(0, 60, 100)
    ax.plot(xx, slope * xx, color="gray", linestyle="--", alpha=0.7,
            label=f"Linear fit: recovery $\\approx$ {slope:.2f} $\\times$ concentration")
    ax.set_xlabel("Top-10 noise concentration, % (fp8_e4m3)")
    ax.set_ylabel("K-error recovery from defense N=10, %")
    ax.set_title("Defense efficacy scales linearly with outlier concentration")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 40)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_recovery_vs_concentration.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir/'fig_recovery_vs_concentration.png'}")

    # ===== Figure 3: K-error magnitude across formats × models =====
    fig, ax = plt.subplots(figsize=(10, 5))
    formats = ["fp8_e4m3", "fp8_e5m2", "hqq_int4"]
    fmt_labels = ["FP8 e4m3", "FP8 e5m2", "HQQ INT4"]
    width = 0.2
    x = np.arange(len(formats))
    for i, (label, path, color) in enumerate(models):
        d = json.load(open(path))["per_quant"]
        errs = [d[f]["layer_rel_err_mean"] * 100 for f in formats]
        ax.bar(x + (i - 1.5) * width, errs, width, label=label.replace("\n", " "), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(fmt_labels)
    ax.set_ylabel("Mean relative K-error, %")
    ax.set_title("FP8/HQQ K-error magnitude — architecture-invariant across 4 models")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_kerr_invariance.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir/'fig_kerr_invariance.png'}")


if __name__ == "__main__":
    main()
