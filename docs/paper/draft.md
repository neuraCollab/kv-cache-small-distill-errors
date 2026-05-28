# Why Reasoning Models Diverge under KV-Cache Quantization:
## A Layer-by-Layer Mechanistic Study with Defense Recipes

**Anonymous authors**  
*Paper under double-blind review*

---

## Abstract

Compact reasoning models (1–8B parameters) routinely generate chain-of-thought outputs spanning thousands of tokens, making the KV-cache the dominant memory cost during inference. KV-cache quantization (FP8 e4m3/e5m2, HQQ INT4/INT2) is the standard scaling trick — yet its effect on reasoning quality is typically evaluated only at the aggregate accuracy level, leaving the underlying mechanism opaque. We instrument the K, V tensors of Qwen3-1.7B at 28 layers across 80 AIME-24 + MATH-500 problems and produce the first per-(layer, head, position, channel) decomposition of how K-quantization noise propagates into attention-map perturbation, hidden-state drift, and ultimately argmax flips at the first-divergence point (FDP). Our headline findings:

1. **K-noise is concentrated**: in FP8 e4m3, 10/1024 key channels (1%) carry 51.7% of total K-quant error; 99 channels (10%) carry 80%. This concentration is reproducible across random seeds (Jaccard 0.88).
2. **Concentration is format-dependent**: HQQ INT4 spreads noise 4× more uniformly (top-10 = 13.1%) because its per-group scale + zero-point already adapts within each block.
3. **Per-channel defense (keep top-10 outlier channels in bf16) recovers 34% of FP8 K-error and 15% of attention-map shift** — at a cost of 1% bf16 storage. For HQQ the same defense is *harmful* (+6% error).
4. **Layer-importance ranking is partially universal**: layers 0 and 5 are top-5 contributors under all four quant formats; layers 23-27 are uniformly the safest. Mid-layer policy is format-specific (Spearman ρ=0.80 between FP8 e4m3 and HQQ INT4, but only 0.52 between FP8 e4m3 and e5m2).
5. **Argmax flips are predictable from local margin, not from quant noise magnitude alone**: median bf16 top-1/top-2 margin reaches its trajectory minimum (2.19 nats) exactly at FDP-1; the quant-induced logit shift is two orders of magnitude smaller than the margin on average, so flips only occur where the margin is already small.

All experiments run on CPU; capture pipeline, ~80 GB of raw K/V/logits artifacts, and analysis scripts are released.

---

## 1. Introduction

Modern reasoning-distilled language models such as DeepSeek-R1 and Qwen3 produce explicit chain-of-thought sequences of 5–20k tokens. The KV-cache — two tensors of shape `[batch, num_kv_heads, seq, head_dim]` per layer — typically holds 8–12 GB of bf16 data at the end of a long generation. Quantizing this cache to FP8 (Micikevicius et al., 2022) or to INT4/INT2 with HQQ (Badri & Shaji, 2023) is the lever every production inference stack pulls first.

Yet the literature provides almost no *mechanistic* understanding of how this quantization affects reasoning quality. Existing studies report either (a) aggregate accuracy on a benchmark (often noisy), or (b) average-case quant error in synthetic settings. **What no one has shown is the per-layer, per-position chain of cause and effect: K is quantized → attention map shifts → hidden state drifts → final logit flips → wrong token sampled → cascade of wrong reasoning.**

This paper closes that gap. We:

1. Capture Q (post-RoPE), K_pre / K_post, V_pre / V_post, and logits at every layer of Qwen3-1.7B for 80 problems × 4 quant configs × {teacher-forced, autoregressive} modes, with multi-seed variance estimates.
2. Trace the K-quant noise from input to logit, attribute its share to outlier channels and to per-layer cascades, and measure where in the architecture and trajectory the argmax actually flips.
3. Translate the diagnosis into a practical defense recipe (per-channel scaling) and a cross-architecture / cross-quant generalization study.

The capture pipeline runs entirely on CPU, takes 6–8 hours per full run on a laptop-class machine, and the entire analysis is reproducible from a single git commit.

**Contributions.**
- A per-(layer, head, position, channel) decomposition of K/V quant noise on a real reasoning model.
- A reproducibility-grade dataset of 480 capture files plus per-seed variance JSONs.
- A directly verifiable defense recipe: protect top-10 K outlier channels in bf16, recover 34% of FP8 quant error and 15% of attention shift.
- A negative result on cross-quant generalization of per-layer defenses, with a refined universal/format-specific decomposition.
- A negative result on early-trajectory failure prediction: KL in the first 30–200 decode steps is at most weakly predictive (LOO AUC 0.42–0.57).

---

## 2. Background and Notation

### 2.1 KV-cache quantization formats

We study four formats actively shipped in production inference stacks:

- **FP8 e4m3**: 4 exponent bits, 3 mantissa bits, range ±448, ~12.5% relative precision per step within range. PyTorch native cast (`torch.float8_e4m3fn`); vLLM's default `kv_cache_dtype="fp8_e4m3"`.
- **FP8 e5m2**: 5 exponent bits, 2 mantissa bits, range ±57344, wider range but coarser precision per binade.
- **HQQ INT4**: per-group (group_size=64) asymmetric integer quantization with proximal half-quadratic optimization of scale and zero-point (Badri & Shaji, 2023). 4 bits per weight, group_size scales of bf16.
- **HQQ INT2**: same machinery with 2 bits per weight; 4 representable values per group.

For each layer ℓ we denote the per-head, per-position key tensor $K^{(\ell)}_t \in \mathbb{R}^{H_{kv} \times d_h}$ with $H_{kv}$ KV heads and $d_h$ head dimension. After quantization $K^{(\ell)}_{t,\text{post}} = \text{quant}(K^{(\ell)}_{t,\text{pre}})$. We define the per-(layer, head) relative Frobenius error

$$
\varepsilon^{(\ell)}_h = \frac{\|K^{(\ell)}_{:, h, :, \text{pre}} - K^{(\ell)}_{:, h, :, \text{post}}\|_F}{\|K^{(\ell)}_{:, h, :, \text{pre}}\|_F}.
$$

### 2.2 First-divergence point (FDP)

Given a bf16 baseline trace $(t_0, t_1, \ldots)$ and a quantized trace $(t'_0, t'_1, \ldots)$ produced by the same model under greedy decoding from the same prompt, the FDP is the smallest position at which $t_i \neq t'_i$. We define FDP per (problem, quant) pair using the vLLM-generated traces from prior work.

### 2.3 Capture pipeline

We re-implement the model forward pass with three hook strategies:
1. A `forward_hook` on every `q_proj` linear that captures Q post-projection.
2. A monkey-patch of `transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb` that captures Q post-RoPE.
3. A class-level monkey-patch of `DynamicCache.update` that intercepts K, V *before* they are stored, applies the quant function, captures both pre and post tensors, and stores the post version so subsequent attention computation sees the actual quantized cache.

The third patch is critical: a naive `forward_hook` on the attention block fires *after* attention has already used bf16 K/V, so any cache mutation in that hook arrives too late and the logits are unchanged. We verified this gives logit differences that match a fully simulated vLLM forward (golden test: first 10 decode tokens match vLLM under fp8_e4m3 on AIME-24 problem 0).

Per (problem, quant, mode) we save Q (pre-RoPE and post-RoPE), K_pre, K_post, V_pre, V_post, logits over a window of $W=251$ positions centered on the FDP ([FDP−150, FDP+100]). Total artifact size ~80 GB across 480 capture files plus 180 multi-seed files at $T=0.6$.

---

## 3. K-quantization Noise: Structure and Concentration

We start by characterizing where K-quant error lives in the tensor.

### 3.1 Layer-wise relative error

Mean relative Frobenius error per (layer, head), averaged across 80 problems:

| Quant | mean $\varepsilon$ | max layer × head $\varepsilon$ |
|---|---|---|
| fp8_e4m3 | 2.65% | 3.50% |
| fp8_e5m2 | 5.25% | 7.32% |
| hqq_int4 | 10.3% | n/a |
| hqq_int2 | 60.7% | n/a |

The FP8 ratio is exactly 2×, consistent with the one-mantissa-bit difference. HQQ INT4 carries 4× more error than FP8 e4m3.

### 3.2 Per-channel concentration: a few channels dominate K-noise

For each layer we sort the 1024 channels (8 KV heads × 128 head_dim) by their squared contribution to $\|K_\text{pre} - K_\text{post}\|_F^2$ and report the cumulative fraction of layer-total noise covered by the top-$k$ channels.

**Result (median across 28 layers × 80 problems):**

| Quant | top-1 channel | top-10 | #ch for 50% | #ch for 80% |
|---|---|---|---|---|
| fp8_e4m3 | **10.3%** | **51.7%** | 8 | 99 |
| fp8_e5m2 | 10.0% | 50.4% | 10 | 102 |
| hqq_int4 | 1.5% | 13.1% | 53 | 252 |
| hqq_int2 | 1.6% | 14.4% | 46 | 251 |

**Two robust findings:**

1. **For FP8, one channel out of 1024 carries 10% of all K-quant noise; ten channels carry over half.** This concentration is the structural signature of the per-tensor FP8 scale: outlier channels with large activation magnitudes force the entire tensor's quantization step to be large, while inlier channels suffer disproportionately.

2. **For HQQ, the same noise is 4× more uniform.** Each group of 64 elements has its own scale and zero-point, so outlier channels do not impose their step on inliers. This is structurally why HQQ already does — at the format level — what per-channel defense methods (SmoothQuant, AWQ) try to do *on top of* FP8.

### 3.3 Reproducibility of the outlier set

Across 3 random seeds at $T=0.6$, the *identity* of the top-10 outlier channels per (problem, layer) is stable: median Jaccard overlap = 0.879 (IQR [0.78, 1.00]). The 51.7% concentration figure is robust: 52.5% ± 26.6% across seeds (mean ± std), confirming that outlier channels are an architectural property of the weight matrices, not an artifact of sampling.

---

## 4. From K-Noise to Attention Shift to Logits

K-quantization noise does not affect inference directly — it affects attention scores. We measure this empirically.

### 4.1 Attention map KL: direct measurement

For each layer ℓ and each query position $t$, we form

$$
\text{att}^{(\ell)}_{\text{pre}, t} = \text{softmax}\!\left(\frac{Q^{(\ell)}_t K^{(\ell)\top}_{\text{pre}}}{\sqrt{d_h}}\right),\quad
\text{att}^{(\ell)}_{\text{post}, t} = \text{softmax}\!\left(\frac{Q^{(\ell)}_t K^{(\ell)\top}_{\text{post}}}{\sqrt{d_h}}\right)
$$

with the same post-RoPE $Q^{(\ell)}_t$ in both expressions, then compute $\text{KL}(\text{att}_\text{pre} \| \text{att}_\text{post})$ per (layer, position).

**Results across 80 problems, mean of per-head KL:**

| Quant | global mean attn-KL | @FDP |
|---|---|---|
| fp8_e4m3 | 0.00358 | 0.00352 |
| fp8_e5m2 | 0.01405 | 0.01334 |
| hqq_int4 | 0.391 | 0.381 |
| hqq_int2 | 4.81 | 4.78 |

HQQ INT4 produces an attention-map KL **108× larger** than FP8 e4m3. The softmax nonlinearity amplifies a 4× difference in K-error into a two-orders-of-magnitude difference in attention shift, because softmax behaves nonlinearly when score magnitudes are comparable to the attention temperature $\sqrt{d_h}$.

### 4.2 Attention shift is structural, not trajectory-dependent

We measured attention-shift KL in both teacher-forced and autoregressive modes. The values are nearly identical (fp8_e4m3: 0.00358 in TF vs 0.00358 in AR, to three significant figures). **This shows that attention shift is determined by K-quant noise alone, not by the trajectory's specific token sequence.** The autoregressive trajectory drift we observe later is therefore not a property of the attention computation per se — it is a property of how attention-output perturbations *compound* through the residual stream.

### 4.3 Mismatch between attention shift and final logit impact

For each layer we compare its mean attention-shift KL with its mean *logit-impact* KL measured via single-layer ablation (quantize only that layer, keep others bf16):

| Rank | Attention shift (fp8_e4m3) | Logit impact (fp8_e4m3) |
|---|---|---|
| 1 | L5 (0.00666) | L3 (0.00131) |
| 2 | L3 (0.00500) | L15 |
| 3 | L4 | L0 |
| 4 | L13 | L5 |
| 5 | L7 | L9 |

Only L3 and L5 are in both top-5. Layer 5 has the largest attention shift but only the 4th largest logit impact: its attention perturbation is partially compensated by downstream layers. Layer 0 has small local attention shift but large logit impact: small perturbations at layer 0 are amplified through 27 layers of residual cascade. **Logit divergence is the integral of two distinct quantities — local attention shift magnitude and downstream cascade depth — and the worst layers for a model trade off the two differently.**

---

## 5. Failure Mode: Where and Why Argmax Flips

### 5.1 Logit divergence trajectory in autoregressive mode

In AR mode, fp8_e4m3 diverges from bf16 along a curve that empirically fits a saturating exponential:

$$
\text{KL}_\text{logit}(t) = K_\infty\!\left(1 - e^{-t/\tau}\right)
$$

At $T=0$ (greedy decoding): $\tau = 130$ tokens, $K_\infty \approx 18$ nats, $R^2 = 0.80$ on the median over 80 problems. At $T=0.6$ with 3 random seeds, the median curve gives $\tau = 47 \pm 11$, $K_\infty = 28 \pm 1$ for fp8_e5m2 (tight CI) but $\tau = 772 \pm 1126$ for fp8_e4m3 (the rising portion does not saturate within 250 generation steps; sampling noise dominates early).

The mean characteristic time at $T=0$ thus implies: **after ~130 autoregressive decode steps, fp8_e4m3 and bf16 trajectories are essentially uncorrelated.** This is consistent with FDP medians observed in the vLLM-baseline experiment (FDP ~7% of trace ≈ 540 tokens, well past the saturation regime).

### 5.2 The argmax-flip condition

Argmax flips between bf16 and fp8 occur when the quant-induced logit perturbation $\Delta l_t \in \mathbb{R}^{|V|}$ exceeds the bf16 margin $m_t$ at the next-token decision:

$$
\exists\, j \neq i^*\colon \Delta l_t[j] - \Delta l_t[i^*] > m_t.
$$

In TF mode at the FDP position, the actually-observed magnitude of $\Delta l$ averaged over the vocabulary is two orders of magnitude smaller than the margin (KL≈0.008, vocabulary ≈152k, vs median margin = 2.19 nats), so the *average* flip never happens. Flips occur only where the bf16 margin is unusually low.

Indeed, median margin trajectory over 80 problems:

| Position rel. FDP | -50 | -10 | -1 | 0 | +1 | +10 | +50 |
|---|---|---|---|---|---|---|---|
| Median margin | 7.38 | 2.62 | **2.19** | 4.00 | 5.62 | 5.31 | 4.94 |

The minimum is at FDP-1 — exactly the position where the next-token decision is most fragile. This refines a folklore claim about reasoning-model divergence: it is not that quant noise is large at the divergence point, but that the model's own confidence is small there. **The FDP is determined by the model's local uncertainty, not by an unusual quant noise spike.**

---

## 6. Defense Recipes

We now turn from diagnosis to mitigation. We test three policies, all parameterized by $K$ = number of "protected" units.

### 6.1 Per-layer protection (skip-top-K layers)

Keep $K$ of the 28 layers in bf16; quantize the rest with FP8. The top-$K$ are ranked by the logit-impact ablation from §4.3.

| K | Layers protected | Mean KL(bf16 ‖ fp8) | Reduction vs $K=0$ |
|---|---|---|---|
| 0 | – | 0.00757 | – |
| 1 | L3 | 0.00689 | +9% |
| 3 | L3, L15, L0 | 0.00655 | +14% |
| 5 | L3, L15, L0, L5, L9 | 0.00589 | +22% |
| 10 | top-10 | 0.00427 | +44% |

Recovery scales roughly linearly with the fraction of layers protected. **Layer-level skipping is a weak defense lever**: protecting 36% of layers recovers 44% of error.

### 6.2 Per-channel protection (keep top-N outlier channels in bf16)

For each layer independently, identify the top-$N$ K channels by max $|K_\text{pre}[:, h, c]|$ and keep them in bf16; quantize the remaining $1024 - N$ with FP8 e4m3.

| N | %channels | K-err reduction | Attention-KL reduction |
|---|---|---|---|
| 0 | 0% | – | – |
| 1 | 0.10% | 5.4% | 2.1% |
| 5 | 0.49% | 23.7% | 7.8% |
| **10** | **0.98%** | **34.1%** | **14.6%** |
| 25 | 2.44% | 42.7% | 28.3% |
| 50 | 4.88% | 49.2% | 43.6% |
| 100 | 9.77% | 57.0% | 60.3% |

**Per-channel defense is ~30× more efficient than per-layer defense per protected parameter**: 1% of channels recovers 34% of K error, vs 36% of layers for 44% recovery. The same recipe applied to fp8_e5m2 gives nearly identical numbers (K-err -33%, attention-KL -21%), confirming it generalizes within the FP8 family.

### 6.3 Per-channel defense fails on HQQ

For both HQQ INT4 and INT2, the same recipe **increases** the error and the attention shift:

| Quant | N=10 K-err change | Attention-KL change |
|---|---|---|
| fp8_e4m3 | -34% (recovered) | -15% |
| fp8_e5m2 | -33% | -21% |
| hqq_int4 | +6% (worse) | +21% |
| hqq_int2 | +7% | +18% |

The mechanism: HQQ already adapts a separate scale and zero-point per group of 64 elements. Outlier channels do not strain the format. Forcing them into bf16 introduces a mixed-precision inconsistency (the surrounding channels still use HQQ's adapted scales, but the protected channel now sits in a different representation) that disrupts the group-wise calibration HQQ relies on. **Defense recipes are quant-family-specific: outlier-aware per-channel scaling pays for FP8 but not for HQQ.**

### 6.4 Layer-importance ranking is partially universal

Spearman rank correlations between per-layer logit-impact rankings, n=30 problems × 28 layers:

|  | fp8_e4m3 | fp8_e5m2 | hqq_int4 | hqq_int2 |
|---|---|---|---|---|
| fp8_e4m3 | 1.00 | 0.52 | **0.80** | 0.79 |
| fp8_e5m2 | 0.52 | 1.00 | 0.49 | 0.61 |
| hqq_int4 | 0.80 | 0.49 | 1.00 | **0.92** |
| hqq_int2 | 0.79 | 0.61 | 0.92 | 1.00 |

Across-family correlation (FP8 e4m3 ↔ HQQ INT4, ρ=0.80) is **higher** than within-family (FP8 e4m3 ↔ e5m2, ρ=0.52). Reason: fp8_e5m2 trades mantissa for range, so it is a different *kind* of quant from both fp8_e4m3 and HQQ, both of which prioritize precision over range for typical K values (|K| ≲ 400, well within e4m3 range 448).

**Universal worst layers** (top-5 in *all* four formats): L0, L5.  
**Universal safe layers** (bottom-5 in all four): L23, L24, L25, L26, L27.  
**Format-specific layers**: L3, L15 (FP8); L10, L4 (HQQ).

This yields a hybrid recipe: protect L0, L5 always; aggressively quantize L23-L27 always; calibrate the middle of the network per format.

---

## 7. Cross-Architecture: Outlier Dilution at Scale

We replicate the per-channel concentration measurement on Qwen3-4B (10 problems, prompt-only TF capture). With 36 layers and a wider hidden state, outlier concentration is significantly more diluted:

| Metric | Qwen3-1.7B | Qwen3-4B | Change |
|---|---|---|---|
| top-1 channel fraction | 10.3% | 3.6% | ÷3 |
| top-10 fraction | 51.7% | 21.0% | ÷2.5 |
| #ch for 50% noise | 8 | 56 | ×7 |
| #ch for 80% noise | 99 | 216 | ×2 |

**Outlier concentration is model-size dependent.** As the model scales up, the same massive activations get spread over a wider feature space, so each individual channel carries a smaller fraction of total noise. Practical implication: **per-channel defense becomes proportionately less effective on larger models**. SmoothQuant/AWQ-style methods will need to widen the protection set (top-50 or top-100 channels) to maintain comparable recovery on multi-billion-parameter models.

---

## 8. Negative Results

We report the experiments where the data did not support the hypothesis.

**Failure prediction from early KL trajectory.** Hypothesis: a logistic regression on the first 30–200 positions' KL between bf16 and fp8 logits can predict whether the final answer will diverge (binary label from `baseline_boxed ≠ quant_boxed`). LOO-CV AUC:

| Window length $n$ | fp8_e4m3 | fp8_e5m2 |
|---|---|---|
| 30 | 0.424 | 0.350 |
| 50 | 0.394 | 0.467 |
| 100 | 0.506 | 0.467 |
| 200 | 0.573 | 0.283 |

Early KL trajectory is at best weakly predictive (fp8_e4m3 at $n=200$). For fp8_e5m2 the sample is too small (n=17 problems with aligned windows) for a stable estimate. We do not recommend early-window KL as a standalone collapse detector.

**$\tau$ unstable for fp8_e4m3 at $T=0.6$.** The saturating-exponential fit is robust at $T=0$ ($\tau=130$, $R^2=0.80$) but degenerate at $T=0.6$ with 3 seeds ($\tau=772 \pm 1126$, $K_\infty=146 \pm 189$). The likely cause is that sampling noise at $T=0.6$ dominates the KL signal in the first ~50 steps, and the AR generation does not run long enough to clearly enter the saturation regime under the joint quant + sampling noise.

---

## 9. Related Work

**Per-token / per-tensor FP8 KV-cache** is the default in vLLM (Kwon et al., 2023). PyTorch's native FP8 dtypes (`float8_e4m3fn`, `float8_e5m2`) provide IEEE round-to-nearest-even semantics matching the GPU kernels.

**HQQ** (Badri & Shaji, 2023) is the open-source HQQ quantizer used by transformers' `QuantizedCacheConfig(backend="hqq")`. The half-quadratic optimization gives ~1-2% improvement over min-max group quantization on Gaussian-like distributions; on heavy-tailed K distributions the improvement is larger.

**SmoothQuant** (Xiao et al., 2023) and **AWQ** (Lin et al., 2024) both address weight quantization by per-channel scaling that migrates "difficulty" from activations to weights. Our per-channel defense is the spiritual sibling for the KV-cache: we protect rather than rescale, but the underlying insight — outlier channels dominate quant error — is shared.

**Sub-network analyses of language models** (Olah et al., 2020; Templeton et al., 2024) ablate components but generally do not study per-layer KV-cache quantization specifically. Our layer-ablation results align qualitatively with the broader finding that middle layers carry disproportionate computational weight in attention-heavy transformers.

---

## 10. Limitations

1. **One model, four quant formats.** We characterize the mechanism on Qwen3-1.7B and validate that concentration dilutes on Qwen3-4B. We do not claim that L0, L5 are the universal victim layers for all transformer families; the Spearman analysis is internal to Qwen3-1.7B.
2. **CPU simulation, not GPU kernel.** Our FP8 quant uses PyTorch's native cast, which matches the IEEE FP8 specification. Verified to match vLLM's first 10 decode tokens on AIME-24 problem 0 under fp8_e4m3, but we have not bit-matched it across all 80 problems × 250 decode steps.
3. **N=30 for HQQ layer ablation.** HQQ's ~300 ms per quantization call (vs FP8's 0.4 ms) made the 80-problem ablation infeasible on CPU within the time budget. Results may have wider CI than the FP8 n=80 figures.
4. **No HQQ-native FDPs.** We borrow fp8_e4m3 FDPs for HQQ window centering; for failure-prediction labels this means HQQ AUC numbers are correlation with FP8 outcomes, not direct HQQ outcomes.
5. **Greedy and $T=0.6$ only.** No sweep over temperature or repetition_penalty.

---

## 11. Conclusion

KV-cache quantization is far from a "lossy compression" black box. By tracing K-quant noise through attention to logits at per-layer, per-channel, per-position resolution, we have shown:

1. K-noise structure is **highly concentrated for FP8 (top-10 channels = 51.7% noise) and uniform for HQQ (top-10 = 13.1%)**, a direct consequence of per-tensor vs per-group scaling.
2. **Per-channel defense recovers a third of FP8 quant error for 1% of bf16 storage**, generalizes to fp8_e5m2, and *hurts* HQQ.
3. **Layers 0 and 5 are universal "victim" layers**; layers 23-27 are universal "safe" layers; mid-layer policy is format-specific.
4. The argmax-flip condition at FDP is determined by **the bf16 model's own local margin minimum, not by an unusual quant-noise spike**.

The capture pipeline, raw artifacts (~80 GB), and analysis scripts are released. We hope this provides a substrate for future fine-grained quantization studies.

---

## Reproducibility

- Code: `feature/kv-matrix-capture` branch, commits c820d74…1da8149 (full provenance).
- Hardware: AMD Ryzen 5 7640HS, 15.2 GB RAM, 6h–8h wall-clock per full run.
- Software: Python 3.13, PyTorch 2.12 (CPU), transformers 4.51.3, HQQ 0.2.8.
- Data: Qwen3-1.7B (HF revision 70d244cc) on 30 AIME-24 + 50 MATH-500 problems.
- Seed: torch.manual_seed(42) for greedy; seeds 1, 2, 3 for $T=0.6$ runs.
- Acceptance criteria, golden test (`tests/test_capture_fp8_matches_vllm.py`), and per-script reproducible commands in the repository README.

---

## Acknowledgments

[Anonymized for double-blind review]

---

## References

(Skeleton; full reference list pending — replace with conference's BibTeX style.)

- Badri, H. & Shaji, A. (2023). *Half-Quadratic Quantization of Large Machine Learning Models*. Mobius Labs blog.
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP.
- Lin, J. et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. MLSys.
- Micikevicius, P. et al. (2022). *FP8 Formats for Deep Learning*. arXiv:2209.05433.
- Olah, C. et al. (2020). *Zoom In: An Introduction to Circuits*. Distill.
- Templeton, A. et al. (2024). *Scaling Monosemanticity*. Anthropic.
- Xiao, G. et al. (2023). *SmoothQuant*. ICML.
