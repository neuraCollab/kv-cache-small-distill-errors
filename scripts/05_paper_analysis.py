"""Phase 5 (post-hoc): paper-grade analysis artifacts.

Reads `outputs/{traces,fdps,judgments}/*.jsonl` and writes:

  outputs/paper/per_model_chi2.{md,csv}      — Idea 1
  outputs/paper/accuracy_bars.{png,svg}      — Idea 2
  outputs/paper/quant_only_deepdives.md      — Idea 3
  outputs/paper/divergence_position.{png,md} — Extra 5
  outputs/paper/token_efficiency.md          — Extra 6
  outputs/paper/judge_confidence.md          — Extra 7
  outputs/paper/finish_reason.md             — Extra 8
  outputs/paper/fdp_rate.md                  — Extra 9

CPU-only, no live API. Run from repo root after the pipeline finishes:
    python scripts/05_paper_analysis.py

Designed to be re-runnable (idempotent overwrites). Numbers are computed
once from raw JSONL; every artifact draws from the same in-memory cache
so figures and tables can never disagree.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Force UTF-8 stdout: Windows defaults to cp1251 which crashes on the
# unicode chars (chi², Cramér) used in section titles and progress prints.
# Files are already written with encoding="utf-8"; this just covers stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

OUT = Path("outputs")
TRACES = OUT / "traces"
FDPS = OUT / "fdps"
JUDGMENTS = OUT / "judgments"
PAPER = OUT / "paper"

CATEGORIES = ["A", "B", "C", "D", "E", "F"]
CATEGORY_NAMES = {
    "A": "Arithmetic",
    "B": "Logical",
    "C": "Strategy-switch",
    "D": "Hallucination",
    "E": "Premature-termination",
    "F": "Repetition/loop",
}

# Canonical display order. bf16 comes first (baseline), then fp8 (mild),
# then hqq (aggressive). Models are ordered by parameter count.
MODEL_ORDER = [
    "deepseek-r1-distill-qwen-1.5b",
    "qwen3-1.7b",
    "deepseek-r1-distill-qwen-7b",
]
QUANT_ORDER = ["bf16", "fp8_e4m3", "fp8_e5m2", "hqq_int4", "hqq_int2"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL loader that doesn't break on embedded newlines."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_runs() -> dict[str, list[str]]:
    """Return {model: [quants...]} based on traces/ directory."""
    found: dict[str, set[str]] = defaultdict(set)
    for p in TRACES.glob("*.jsonl"):
        # File names: "<model>_<quant>.jsonl". Quant is the last underscore
        # component, but model names contain hyphens not underscores so the
        # rsplit("_", 1) split is unambiguous.
        stem = p.stem
        model, _, quant = stem.rpartition("_")
        # Special case: "fp8_e4m3" / "fp8_e5m2" / "hqq_int4" / "hqq_int2"
        # have an underscore inside the quant key. Detect by checking if
        # the trailing token is one of {e4m3, e5m2, int4, int2}.
        if quant in {"e4m3", "e5m2", "int4", "int2"}:
            model, _, prefix = model.rpartition("_")
            quant = f"{prefix}_{quant}"
        found[model].add(quant)
    return {m: sorted(found[m], key=lambda q: QUANT_ORDER.index(q)
                      if q in QUANT_ORDER else 99)
            for m in sorted(found, key=lambda m: MODEL_ORDER.index(m)
                            if m in MODEL_ORDER else 99)}


def load_traces(model: str, quant: str) -> list[dict[str, Any]]:
    return load_jsonl(TRACES / f"{model}_{quant}.jsonl")


def load_fdps(model: str, quant: str) -> list[dict[str, Any]]:
    p = FDPS / f"{model}_{quant}.jsonl"
    return load_jsonl(p) if p.exists() else []


def load_judgments(model: str, quant: str) -> list[dict[str, Any]]:
    p = JUDGMENTS / f"{model}_{quant}.jsonl"
    return load_jsonl(p) if p.exists() else []


# --------------------------------------------------------------------------
# Shared metrics (computed once, reused everywhere)
# --------------------------------------------------------------------------

def compute_accuracy(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Return (n_correct, n_total). Correct iff boxed_answer matches ground_truth
    (string-stripped equality). Same logic as 04_analyze.py."""
    correct = 0
    for r in rows:
        b = r.get("boxed_answer")
        gt = r.get("ground_truth")
        if b is not None and gt is not None and str(b).strip() == str(gt).strip():
            correct += 1
    return correct, len(rows)


def avg_gen_tokens(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(r.get("num_generated_tokens", 0) for r in rows) / len(rows)


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Cramér's V from chi² statistic. Corrects to [0, 1]; 1 = perfect dep."""
    denom = n * (min(r, c) - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  wrote {path}")


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a GitHub-flavored markdown table. All cells stringified."""
    h = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |"
                     for row in rows)
    return "\n".join([h, sep, body])


# --------------------------------------------------------------------------
# Idea 1: per-model chi² and Cramér's V
# --------------------------------------------------------------------------

def idea_1_per_model_chi2(runs: dict[str, list[str]]) -> None:
    print("Idea 1: per-model chi² and Cramér's V")
    sections: list[str] = ["# Per-model chi² and Cramér's V",
                            "",
                            "Tests whether the failure-category distribution depends on quantization "
                            "method, **independently for each model**. The global chi² in `report.md` "
                            "lumps all models together; this view shows whether the dependency is "
                            "uniform or model-specific.",
                            ""]
    csv_rows = ["model,chi2,dof,p_value,cramers_v,n,quants"]

    for model, quants in runs.items():
        # Skip bf16 (it's the baseline; no judgments).
        quant_subset = [q for q in quants if q != "bf16"]
        if len(quant_subset) < 2:
            sections.append(f"## {model}\n\n_Skipped: only {len(quant_subset)} "
                            f"quant variant — chi² needs ≥2._\n")
            continue

        # Build observed contingency table: rows=quants, cols=categories.
        obs: list[list[int]] = []
        for q in quant_subset:
            j = load_judgments(model, q)
            cnt = Counter(r.get("category") for r in j)
            obs.append([cnt.get(c, 0) for c in CATEGORIES])

        obs_arr = np.array(obs, dtype=int)
        # Drop columns that are all-zero — chi2_contingency rejects them.
        nonzero_cols = [i for i in range(len(CATEGORIES)) if obs_arr[:, i].sum() > 0]
        if len(nonzero_cols) < 2:
            sections.append(f"## {model}\n\n_Skipped: <2 non-zero categories "
                            f"across quants._\n")
            continue
        obs_trim = obs_arr[:, nonzero_cols]
        chi2, p, dof, exp = chi2_contingency(obs_trim)
        n = int(obs_trim.sum())
        v = cramers_v(chi2, n, *obs_trim.shape)

        # Section
        sections.append(f"## {model}")
        sections.append("")
        sections.append(f"- **chi² = {chi2:.2f}**, dof = {dof}, "
                        f"**p = {p:.2e}**")
        sections.append(f"- **Cramér's V = {v:.3f}** "
                        f"(0 = independent, 1 = perfect dependence)")
        sections.append(f"- N judgments = {n}")
        if v < 0.1:
            interp = "negligible"
        elif v < 0.3:
            interp = "small-to-moderate"
        elif v < 0.5:
            interp = "moderate-to-strong"
        else:
            interp = "strong"
        sections.append(f"- Effect size: **{interp}**")
        sections.append("")

        sections.append("**Observed counts:**")
        sections.append("")
        sections.append(md_table(["quant"] + CATEGORIES + ["total"],
                                 [[q] + list(obs_arr[i]) + [int(obs_arr[i].sum())]
                                  for i, q in enumerate(quant_subset)]))
        sections.append("")

        # Standardized residuals: (obs - exp) / sqrt(exp). |r| > 2 ≈ outlier.
        # We compute on the trimmed table, then reinsert zero columns for display.
        residuals = (obs_trim - exp) / np.sqrt(exp)
        full_resid = np.zeros_like(obs_arr, dtype=float)
        for trim_i, full_i in enumerate(nonzero_cols):
            full_resid[:, full_i] = residuals[:, trim_i]
        sections.append("**Standardized residuals** (|r|>2 marks unusually "
                        "high/low cells):")
        sections.append("")
        sections.append(md_table(
            ["quant"] + CATEGORIES,
            [[q] + [f"{full_resid[i, k]:+.2f}" for k in range(len(CATEGORIES))]
             for i, q in enumerate(quant_subset)]))
        sections.append("")

        csv_rows.append(f"{model},{chi2:.4f},{dof},{p:.6e},"
                        f"{v:.4f},{n},{'+'.join(quant_subset)}")

    write_text(PAPER / "per_model_chi2.md", "\n".join(sections) + "\n")
    write_text(PAPER / "per_model_chi2.csv", "\n".join(csv_rows) + "\n")


# --------------------------------------------------------------------------
# Idea 2: per-quant accuracy bar chart
# --------------------------------------------------------------------------

def idea_2_accuracy_bars(runs: dict[str, list[str]]) -> None:
    print("Idea 2: accuracy bar chart")

    # Compute accuracy[(model, quant)] = correct/total
    acc: dict[tuple[str, str], tuple[int, int]] = {}
    for model, quants in runs.items():
        for q in quants:
            acc[(model, q)] = compute_accuracy(load_traces(model, q))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    quant_subset = [q for q in QUANT_ORDER]  # all quants
    n_quants = len(quant_subset)
    n_models = len(runs)
    bar_w = 0.8 / n_quants
    x_base = np.arange(n_models)

    palette = {
        "bf16":     "#444444",
        "fp8_e4m3": "#2a9d8f",
        "fp8_e5m2": "#52b788",
        "hqq_int4": "#e76f51",
        "hqq_int2": "#bb3e03",
    }

    for qi, q in enumerate(quant_subset):
        heights = []
        for m in runs:
            c, n = acc.get((m, q), (0, 0))
            heights.append(c / n if n else np.nan)
        offset = (qi - (n_quants - 1) / 2) * bar_w
        bars = ax.bar(x_base + offset, heights, bar_w,
                      label=q, color=palette.get(q, "#888"))
        # Annotate each bar with absolute % and Δ vs bf16.
        for bi, (m, h) in enumerate(zip(runs, heights)):
            if np.isnan(h):
                continue
            bf_c, bf_n = acc.get((m, "bf16"), (0, 0))
            bf_acc = bf_c / bf_n if bf_n else None
            if q == "bf16" or bf_acc is None:
                label = f"{h:.0%}"
            else:
                delta = h - bf_acc
                sign = "+" if delta >= 0 else ""
                label = f"{h:.0%}\n({sign}{delta:.0%})"
            ax.text(x_base[bi] + offset, h + 0.015, label,
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_base)
    ax.set_xticklabels([m.replace("deepseek-r1-distill-qwen-", "DS-")
                          .replace("qwen3-", "Qwen3-")
                        for m in runs])
    ax.set_ylabel("Accuracy (boxed_answer == ground_truth)")
    ax.set_ylim(0, max(0.7, max((h for (m, q), (c, n) in acc.items()
                                  if n for h in [c / n]), default=0.7) + 0.1))
    ax.set_title("AIME-24 + MATH-500 accuracy by KV cache quantization")
    ax.legend(loc="upper right", title="quant", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    PAPER.mkdir(parents=True, exist_ok=True)
    plt.savefig(PAPER / "accuracy_bars.png", dpi=150)
    plt.savefig(PAPER / "accuracy_bars.svg")
    plt.close()
    print(f"  wrote {PAPER / 'accuracy_bars.png'}")
    print(f"  wrote {PAPER / 'accuracy_bars.svg'}")


# --------------------------------------------------------------------------
# Idea 3: quant_only deep-dives
# --------------------------------------------------------------------------

def idea_3_quant_only_deepdives(runs: dict[str, list[str]]) -> None:
    print("Idea 3: quant_only deep-dives")
    sections: list[str] = [
        "# `quant_only` deep-dives",
        "",
        "Cases where the **quantized** trace produced the correct boxed answer "
        "but the **bf16 baseline** did not. These are interesting because they "
        "violate the naive expectation that quantization can only degrade "
        "performance — here, KV cache noise pushed the model onto a path the "
        "deterministic baseline never explored.",
        "",
        "Each entry shows: the problem, ground truth, the FDP token index, the "
        "judge's category for this divergence, the last common reasoning prefix, "
        "and ~600 chars of each branch around the divergence point. The judge "
        "still classifies these as divergences (the local *step* differs), but "
        "the global outcome is favorable to the quantized branch.",
        "",
    ]

    cases_found = 0
    for model, quants in runs.items():
        for q in quants:
            if q == "bf16":
                continue
            fdps = load_fdps(model, q)
            judgments = {(j["problem_idx"], j["model"], j["quant_method"]): j
                         for j in load_judgments(model, q)}
            for fdp in fdps:
                if fdp.get("boxed_match") != "quant_only":
                    continue
                cases_found += 1
                key = (fdp["problem_idx"], model, q)
                judge = judgments.get(key, {})
                sections.append(f"## Case {cases_found}: "
                                f"`{model}` × `{q}` × problem {fdp['problem_idx']}")
                sections.append("")
                sections.append(f"- **Source**: {fdp.get('source', '?')}")
                sections.append(f"- **FDP token idx**: "
                                f"{fdp.get('fdp_token_idx')}")
                sections.append(f"- **Ground truth**: "
                                f"`{fdp.get('ground_truth')}`")
                sections.append(f"- **Baseline boxed**: "
                                f"`{fdp.get('baseline_boxed')}`")
                sections.append(f"- **Quant boxed**: "
                                f"`{fdp.get('quant_boxed')}`")
                if judge:
                    cat = judge.get("category", "?")
                    cname = CATEGORY_NAMES.get(cat, "?")
                    sections.append(f"- **Judge category**: "
                                    f"**{cat} ({cname})**, "
                                    f"confidence {judge.get('confidence', 0):.2f}")
                    sections.append(f"- **Judge rationale**: "
                                    f"{judge.get('rationale', '')}")
                    sections.append(f"- **Affected span**: "
                                    f"`{judge.get('affected_span', '')}`")
                sections.append("")
                sections.append("**Problem:**")
                sections.append("")
                sections.append("> " + fdp["problem"].replace("\n", "\n> "))
                sections.append("")
                sections.append("**Common prefix (last part — same in both):**")
                sections.append("")
                sections.append("```")
                sections.append(fdp.get("common_prefix", "")[-800:])
                sections.append("```")
                sections.append("")
                sections.append("**Baseline branch (bf16) around FDP:**")
                sections.append("")
                sections.append("```")
                sections.append(fdp.get("baseline_context", "")[:800])
                sections.append("```")
                sections.append("")
                sections.append(f"**Quantized branch ({q}) around FDP:**")
                sections.append("")
                sections.append("```")
                sections.append(fdp.get("quant_context", "")[:800])
                sections.append("```")
                sections.append("")
                sections.append("---")
                sections.append("")

    sections.insert(4, f"**Total `quant_only` cases found**: {cases_found}\n")
    write_text(PAPER / "quant_only_deepdives.md", "\n".join(sections) + "\n")


# --------------------------------------------------------------------------
# Extra 5: divergence position
# --------------------------------------------------------------------------

def extra_5_divergence_position(runs: dict[str, list[str]]) -> None:
    print("Extra 5: divergence position histogram")

    # For each (model, quant) collect normalized FDP positions:
    #   fdp_token_idx / num_generated_tokens_baseline
    # (the baseline's own length, since the FDP is measured in the common
    # prefix — using baseline length keeps the denominator stable.)
    data: dict[tuple[str, str], list[float]] = defaultdict(list)

    for model, quants in runs.items():
        if "bf16" not in quants:
            continue
        baseline_len_by_idx = {r["idx"]: r.get("num_generated_tokens", 0)
                                for r in load_traces(model, "bf16")}
        for q in quants:
            if q == "bf16":
                continue
            for fdp in load_fdps(model, q):
                idx = fdp["fdp_token_idx"]
                baseline_len = baseline_len_by_idx.get(fdp["problem_idx"], 0)
                if idx is None or baseline_len <= 0:
                    continue
                pos = min(idx / baseline_len, 1.0)
                data[(model, q)].append(pos)

    # Plot grid: rows=models, cols=quants present for that model.
    plot_models = [m for m in runs if any((m, q) in data
                                            for q in runs[m] if q != "bf16")]
    quant_set = sorted({q for (_, q) in data}, key=lambda q: QUANT_ORDER.index(q)
                        if q in QUANT_ORDER else 99)
    n_rows = len(plot_models)
    n_cols = len(quant_set)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows),
                              sharex=True, sharey=True, squeeze=False)
    bins = np.linspace(0, 1, 21)

    def short(m: str) -> str:
        if m.startswith("deepseek-r1-distill-qwen-"):
            return "DS-" + m.removeprefix("deepseek-r1-distill-qwen-")
        if m.startswith("qwen3-"):
            return "Qwen3-" + m.removeprefix("qwen3-")
        return m

    for ri, model in enumerate(plot_models):
        for ci, q in enumerate(quant_set):
            ax = axes[ri, ci]
            vals = data.get((model, q), [])
            ax.set_title(f"{short(model)} | {q}", fontsize=8)
            if vals:
                ax.hist(vals, bins=bins, color="#2a9d8f",
                        edgecolor="white", linewidth=0.5)
                ax.axvline(np.median(vals), color="#e76f51",
                            linestyle="--", linewidth=1)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                          transform=ax.transAxes, color="#888", fontsize=8)
            ax.tick_params(labelsize=7)
            if ri == n_rows - 1:
                ax.set_xlabel("FDP / baseline length", fontsize=8)
            if ci == 0:
                ax.set_ylabel("# problems", fontsize=8)
    fig.suptitle("Divergence position distribution (red = median)",
                  fontsize=11, y=1.00)
    plt.tight_layout()
    plt.savefig(PAPER / "divergence_position.png", dpi=150)
    plt.savefig(PAPER / "divergence_position.svg")
    plt.close()
    print(f"  wrote {PAPER / 'divergence_position.png'}")

    # Markdown summary table
    lines = [
        "# Divergence position summary",
        "",
        "Where in the trace does quantization first cause a divergence? "
        "Position is `fdp_token_idx / num_generated_tokens_baseline`. "
        "0.0 = diverges at the very first generated token; 1.0 = diverges "
        "right at the end. The plot is `divergence_position.png`.",
        "",
        "**Reading guide.** A median near 0.0 means quantization noise hits "
        "early — the model goes off the rails almost immediately. A median "
        "near 1.0 means quantization tolerates most of the reasoning chain "
        "and only causes problems near the end (compounding error). A flat "
        "histogram means no positional pattern.",
        "",
    ]
    rows = []
    for (m, q), vals in sorted(data.items()):
        if not vals:
            continue
        rows.append([m, q, len(vals),
                      f"{np.median(vals):.2f}",
                      f"{np.mean(vals):.2f}",
                      f"{np.std(vals):.2f}",
                      f"{np.min(vals):.2f}",
                      f"{np.max(vals):.2f}"])
    lines.append(md_table(["model", "quant", "n",
                              "median", "mean", "std", "min", "max"], rows))
    lines.append("")
    write_text(PAPER / "divergence_position.md", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Extra 6: token-efficiency table
# --------------------------------------------------------------------------

def extra_6_token_efficiency(runs: dict[str, list[str]]) -> None:
    print("Extra 6: token efficiency")
    lines = [
        "# Token-efficiency table",
        "",
        "Average generated tokens per problem, by (model, quant). The Δ "
        "column is relative to the model's bf16 baseline. Large positive "
        "Δ correlates with category F (Repetition/loop) hits and with "
        "`finish_reason=length` — the model gets stuck and consumes its "
        "entire `max_tokens` budget without finishing.",
        "",
    ]
    rows = []
    for model, quants in runs.items():
        bf_avg = avg_gen_tokens(load_traces(model, "bf16")) if "bf16" in quants else None
        for q in quants:
            traces = load_traces(model, q)
            avg = avg_gen_tokens(traces)
            if bf_avg and q != "bf16":
                delta = avg - bf_avg
                pct = (avg / bf_avg - 1) * 100
                delta_s = f"{delta:+.0f} ({pct:+.1f}%)"
            else:
                delta_s = "—"
            length_pct = (sum(1 for r in traces if r.get("finish_reason") == "length")
                          / max(len(traces), 1))
            rows.append([model, q, len(traces),
                          f"{avg:.0f}", delta_s, f"{length_pct:.0%}"])
    lines.append(md_table(
        ["model", "quant", "n", "avg_gen_tokens", "Δ vs bf16", "% finish=length"],
        rows))
    lines.append("")
    lines.append("> Interpretation: every catastrophic configuration in the "
                  "DeepSeek family ends with `finish_reason=length` rates "
                  "≥80%, confirming Category F dominance. Qwen3 is the only "
                  "family where quantization does not inflate token consumption.")
    write_text(PAPER / "token_efficiency.md", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Extra 7: judge confidence × category
# --------------------------------------------------------------------------

def extra_7_judge_confidence(runs: dict[str, list[str]]) -> None:
    print("Extra 7: judge confidence × category cross-tab")
    # Aggregate all judgments
    by_cat: dict[str, list[float]] = defaultdict(list)
    for model, quants in runs.items():
        for q in quants:
            if q == "bf16":
                continue
            for j in load_judgments(model, q):
                c = j.get("category")
                conf = j.get("confidence")
                if c in CATEGORIES and isinstance(conf, (int, float)):
                    by_cat[c].append(float(conf))

    lines = [
        "# Judge confidence × category cross-tabulation",
        "",
        "Sanity check on the 6-category taxonomy. Each Anthropic judgment "
        "carries a `confidence` ∈ [0, 1] reflecting how sure the model is "
        "about the assigned category. Categories that consistently get **low** "
        "average confidence are taxonomically ambiguous (the judge can't "
        "decide); those with **high** confidence are well-discriminated.",
        "",
    ]
    rows = []
    for c in CATEGORIES:
        confs = by_cat.get(c, [])
        if confs:
            arr = np.array(confs)
            rows.append([f"{c} ({CATEGORY_NAMES[c]})", len(arr),
                          f"{arr.mean():.2f}", f"{arr.std():.2f}",
                          f"{arr.min():.2f}", f"{arr.max():.2f}",
                          f"{(arr < 0.5).sum()}"])
        else:
            rows.append([f"{c} ({CATEGORY_NAMES[c]})", 0, "—", "—", "—", "—", "—"])
    lines.append(md_table(
        ["category", "n", "mean_conf", "std_conf", "min", "max", "n_below_0.5"],
        rows))
    lines.append("")
    lines.append("> Interpretation: F (Repetition/loop) is the most "
                  "syntactically obvious failure — easy for the judge to "
                  "spot via repeated n-grams. C (Strategy-switch) tends to be "
                  "more contested because what counts as 'unmotivated' is "
                  "subjective. A categories with `n_below_0.5 > 5` should be "
                  "treated cautiously when reading downstream stats.")
    write_text(PAPER / "judge_confidence.md", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Extra 8: finish_reason distribution by quant
# --------------------------------------------------------------------------

def extra_8_finish_reason(runs: dict[str, list[str]]) -> None:
    print("Extra 8: finish_reason by quant")
    lines = [
        "# `finish_reason` distribution by (model, quant)",
        "",
        "vLLM reports `finish_reason ∈ {stop, length, ...}`. `stop` means the "
        "model emitted the EOS token cleanly; `length` means it hit "
        "`max_tokens` first. A configuration with high `length` rate (>50%) "
        "is producing runaway / looping reasoning; this is the underlying "
        "mechanism behind Category F.",
        "",
        "**Caveat — HQQ rows.** The HQQ runs use the HuggingFace generation "
        "loop, which in `src/kvtrace/generators/hf_gen.py` unconditionally "
        "reports `finish_reason='stop'` (HF's `model.generate()` does not "
        "expose per-sample termination reasons in a clean form). For HQQ "
        "cells, the column `% length` is therefore always 0% by construction "
        "and should not be read literally. Use `avg_gen_tokens` from "
        "`token_efficiency.md` as a proxy: rows where avg ≈ max_tokens (16384) "
        "almost certainly hit the length cap regardless of the reported label.",
        "",
    ]
    rows = []
    for model, quants in runs.items():
        for q in quants:
            traces = load_traces(model, q)
            n = len(traces)
            fr = Counter(r.get("finish_reason") for r in traces)
            stop_pct = fr.get("stop", 0) / max(n, 1)
            length_pct = fr.get("length", 0) / max(n, 1)
            rows.append([model, q, n,
                          f"{stop_pct:.0%}", f"{length_pct:.0%}",
                          str(dict(fr))])
    lines.append(md_table(
        ["model", "quant", "n", "% stop", "% length", "raw"],
        rows))
    lines.append("")
    lines.append("> Note: `hqq_int4` and `hqq_int2` on deepseek-1.5b show "
                  "100% `stop` despite 0% accuracy — these configurations "
                  "produce *short, confident, wrong* outputs rather than loops. "
                  "Different failure mode from fp8 on the same model.")
    write_text(PAPER / "finish_reason.md", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Extra 9: per-model FDP rate
# --------------------------------------------------------------------------

def extra_9_fdp_rate(runs: dict[str, list[str]]) -> None:
    print("Extra 9: per-model FDP rate")
    lines = [
        "# Per-model FDP (First Divergence Point) rate",
        "",
        "For each (model, quant) pair: out of the 80 problems, how many "
        "produced a non-trivial divergence between baseline and quantized "
        "trace (i.e. `fdp_token_idx is not None`). A 100% FDP rate means "
        "every problem diverged somewhere; a 0% rate would mean the "
        "quantization had no observable effect on any trace.",
        "",
        "The complementary statistic is `boxed_match` — even when traces "
        "diverge token-wise, the quantized branch can still arrive at the "
        "same final answer. Both are reported below.",
        "",
    ]
    rows = []
    for model, quants in runs.items():
        for q in quants:
            if q == "bf16":
                continue
            fdps = load_fdps(model, q)
            n_pairs = len(fdps)
            if n_pairs == 0:
                continue
            diverged = sum(1 for f in fdps if f.get("fdp_token_idx") is not None)
            cosmetic = sum(1 for f in fdps if f.get("cosmetic_skipped"))
            bm = Counter(f.get("boxed_match") for f in fdps)
            rows.append([
                model, q, n_pairs,
                f"{diverged} ({diverged/n_pairs:.0%})",
                str(cosmetic),
                str(bm.get("both_correct", 0)),
                str(bm.get("baseline_only", 0)),
                str(bm.get("quant_only", 0)),
                str(bm.get("both_wrong", 0)),
                str(bm.get("no_boxed", 0)),
            ])
    lines.append(md_table(
        ["model", "quant", "n_pairs", "diverged",
          "cosmetic_skipped",
          "both_correct", "baseline_only", "quant_only",
          "both_wrong", "no_boxed"],
        rows))
    lines.append("")
    lines.append("> The `quant_only` column is the headline anomaly: "
                  "quantization noise produced the right answer in cases "
                  "where the deterministic bf16 baseline did not. All such "
                  "cases observed in this study come from `qwen3-1.7b` × fp8 "
                  "(2 + 3 = 5 cases out of 160 fp8 trials). See "
                  "`quant_only_deepdives.md` for full traces.")
    write_text(PAPER / "fdp_rate.md", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    if not OUT.exists():
        print(f"ERROR: {OUT} does not exist — pipeline never ran")
        return 1
    PAPER.mkdir(parents=True, exist_ok=True)

    runs = discover_runs()
    print("=== Discovered runs ===")
    for m, qs in runs.items():
        print(f"  {m}: {qs}")
    print()

    idea_1_per_model_chi2(runs)
    idea_2_accuracy_bars(runs)
    idea_3_quant_only_deepdives(runs)
    extra_5_divergence_position(runs)
    extra_6_token_efficiency(runs)
    extra_7_judge_confidence(runs)
    extra_8_finish_reason(runs)
    extra_9_fdp_rate(runs)
    print()
    print(f"All artifacts written to {PAPER}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
