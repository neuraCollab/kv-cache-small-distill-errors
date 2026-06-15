# Подтверждённые результаты статьи (practical verification)

Этот файл фиксирует claim'ы из `docs/paper/draft.md`, которые **воспроизвелись**
при практической верификации на живой модели — либо через capture-pipeline на
80 задачах (lab JSONs в `outputs/.../analysis/`), либо через свежие пробы на 5
независимых задачах (`scripts/20_paper_validation.py` → `paper_validation_live.json`),
либо через end-to-end AR-mode generation (`scripts/19_defense_validation.py`).

---

## Условия, при которых результаты воспроизводятся

Все цифры ниже получены при **строго следующих условиях**. Изменение любого из
них может изменить численные значения; для FP8-family claim'ов изменение модели
или dataset влияет на абсолютные числа, но не на знак эффекта.

| Параметр | Значение | Где зашито |
|---|---|---|
| **Модель** | `Qwen/Qwen3-1.7B` (HF revision `70d244cc`) | `runner.load_model(...)` |
| **Архитектура** | 28 layers, 8 KV-heads × 128 head_dim → 1024 channels/layer | модель |
| **Хранение KV** | bf16 native cache (`DynamicCache`) | transformers default |
| **Квант FP8** | PyTorch native `torch.float8_e4m3fn` / `torch.float8_e5m2`, **per-tensor** | `src/kvtrace/capture/fp8_sim.py` |
| **Квант HQQ** | `hqq.core.quantize.Quantizer`, `nbits=4/2`, `group_size=64`, `axis=0`, `optimize=True` | same |
| **Hook strategy** | class-level monkey-patch `DynamicCache.update` для перехвата K/V до их сохранения в кеш | `src/kvtrace/capture/attention_hooks.py` |
| **Dataset** | MATH-500 (HuggingFaceH4/MATH-500), задачи 0-79 для lab, 70-74 для live re-check | `dataset_loader.load_math_dataset` |
| **Prompt template** | `apply_chat_template(messages, enable_thinking=True, add_generation_prompt=True)` | `_build_prompt_tokens` |
| **User instruction** | `"Please reason step by step, and put your final answer within \\boxed{}."` | константа `DEFAULT_USER_INSTRUCTION` |
| **Mode capture** | teacher-forced для §3/§4/§6 lab metrics, autoregressive для §5/§19 | manifest mode |
| **Window** | $W=251$ позиций, центр на FDP (для AR) / на конце prompt (для TF) | `WindowSpec` |
| **Decoding** | greedy ($T=0$) для headline; multi-seed $T=0.6$ для variance | `do_sample=False` |
| **Hardware** | CPU (AMD Ryzen 5 7640HS, 15.2 GB RAM) | — |
| **Software** | Python 3.13, PyTorch 2.12, transformers 4.51.3, hqq 0.2.8 | — |

---

## §3.1 Layer-wise relative Frobenius K-error ✓

**Claim**: FP8 e4m3 mean K-error = 2.65%, max = 3.50%; FP8 e5m2 = 5.25% / 7.32%;
HQQ INT4 = 10.3%; HQQ INT2 = 60.7%.

| Quant | Paper mean | Lab data (n=80) | Live (n=5 fresh probs 70-74) |
|---|---|---|---|
| fp8_e4m3 | 2.65% | **2.65%** ✓ | **2.66%** ✓ |
| fp8_e4m3 max | 3.50% | **3.50%** ✓ | 2.87% (n=5 small) |
| fp8_e5m2 | 5.25% | **5.25%** ✓ | **5.28%** ✓ |
| fp8_e5m2 max | 7.32% | **7.32%** ✓ | 5.91% (n=5 small) |
| hqq_int4 | 10.3% | **10.32%** ✓ | 11.10% (slightly off, n=5) |
| hqq_int2 | 60.7% | **60.72%** ✓ | not measured |

The 2× ratio between e4m3 and e5m2 — and the 4× ratio FP8 → HQQ INT4 — воспроизвелись
exactly. Live re-check на 5 свежих задачах (MATH-500 indices 70-74, not in lab
80) подтвердил FP8 numbers с точностью до второго знака.

**Источник**: `layer_head_error_*.npz` (lab), `paper_validation_live.json` (live).

---

## §3.2 K-noise concentration ✓ (headline finding)

**Claim**: для FP8 e4m3 — top-1 канал из 1024 несёт **10.3%** шума, top-10 несут
**51.7%**; для HQQ INT4 — top-10 несут только **13.1%** (4× меньше).

| Quant | Metric | Paper | Lab (n=80, median) | Live (n=5, median) |
|---|---|---|---|---|
| fp8_e4m3 | top-1 frac | 10.3% | **10.27%** ✓ | 10.10% ✓ |
| fp8_e4m3 | top-10 frac | 51.7% | **51.74%** ✓ | 43.70% ≈ ✓ (n=5 noisy) |
| fp8_e4m3 | #ch for 50% | 8 | **8.5** ✓ | n/a |
| fp8_e4m3 | #ch for 80% | 99 | **99** ✓ | n/a |
| fp8_e5m2 | top-1 frac | 10.0% | **9.99%** ✓ | 9.85% ✓ |
| fp8_e5m2 | top-10 frac | 50.4% | **50.45%** ✓ | 43.02% ≈ ✓ |
| hqq_int4 | top-1 frac | 1.5% | **1.46%** ✓ | 1.25% ✓ |
| hqq_int4 | top-10 frac | 13.1% | **13.06%** ✓ | 11.24% ✓ |
| hqq_int4 | #ch for 50% | 53 | **53** ✓ | n/a |
| hqq_int2 | top-1 frac | 1.6% | **1.59%** ✓ | not measured |
| hqq_int2 | top-10 frac | 14.4% | **14.41%** ✓ | not measured |

**Условия, при которых работает**:
- per-tensor FP8 scale (default vLLM kv_cache_dtype) — concentration появляется
  именно потому что один outlier на всю K-матрицу диктует step для inliers.
- HQQ с `group_size=64` — concentration в 4× меньше как раз потому, что каждая
  группа из 64 elements имеет свой scale+zero-point.
- Per-channel scaling (SmoothQuant/AWQ-style) поверх FP8 даст другую concentration.
- При увеличении модели (Qwen3-4B) concentration падает (см. §7 confirmed).

**Источник**: `outlier_channel_impact_*.json`, `paper_validation_live.json`.

---

## §3.3 Reproducibility of outlier set (Jaccard) ✓

**Claim**: median Jaccard overlap top-10 каналов между 3 random seeds @ $T=0.6$ =
**0.879**, IQR [0.78, 1.00].

| Metric | Paper | Lab data |
|---|---|---|
| Jaccard median (fp8_e4m3) | 0.879 | **0.879** ✓ |
| Jaccard Q25 | 0.78 | **0.768** ✓ |
| Jaccard Q75 | 1.00 | **1.00** ✓ |
| top10 frac mean ± std | 52.5% ± 26.6% | **52.46% ± 26.65%** ✓ |

Outlier identity — **архитектурное свойство матриц весов**, а не артефакт
sampling'a. Это важно для production: outliers можно пред-вычислить offline
на одном calibration set и переиспользовать.

**Условия для работы**:
- $T=0.6$ (paper-tested temperature). При $T=0$ identity ещё стабильнее.
- 3 seeds = достаточно для оценки median; больше seeds дадут уже точнее IQR.

**Источник**: `outputs/kv_capture/qwen3-1.7b_multiseed/analysis/variance_summary_fp8_e4m3.json`.

---

## §4.1 Attention-map KL ✓ (exact match)

**Claim**: после softmax(QK/√d) с pre vs post K, KL(att_pre ‖ att_post):

| Quant | Paper global mean | Data | Paper @FDP | Data @FDP |
|---|---|---|---|---|
| fp8_e4m3 | 0.00358 | **0.00358** ✓ | 0.00352 | **0.00352** ✓ |
| fp8_e5m2 | 0.01405 | **0.01405** ✓ | 0.01334 | **0.01334** ✓ |
| hqq_int4 | 0.391 | **0.391** ✓ | 0.381 | **0.381** ✓ |
| hqq_int2 | 4.81 | **4.81** ✓ | 4.78 | **4.78** ✓ |

Softmax усиливает 4× difference в K-error до **108× difference в attention KL**
(0.00358 → 0.391 от FP8 e4m3 к HQQ INT4). Это нелинейное усиление — central
mechanism paper'а.

**Источник**: `attention_shift_summary_*.json`.

---

## §5.1 Saturating-exponential KL trajectory ✓ (partial)

**Claim**: AR logit KL fits $K_\infty(1 - e^{-t/\tau})$ with τ=130 tokens,
R²=0.80 (T=0); τ=47±11 для fp8_e5m2 @ T=0.6.

| Param | Paper | Data |
|---|---|---|
| τ (e4m3, T=0) | 130 | **129.87** ✓ |
| R² (e4m3, T=0) | 0.80 | **0.796** ✓ |
| τ (e5m2, T=0.6) | 47 ± 11 | **46.81 ± 11.23** ✓ |
| τ (e4m3, T=0.6) | 772 ± 1126 | **772.22 ± 1126.04** ✓ (degenerate) |

**Характерное время** τ=130 voxels confirmed. После 130 AR-шагов траектория
fp8_e4m3 и bf16 становятся essentially uncorrelated по KL.

> **Note**: $K_\infty$ paper-claim ≈ 18 nats **не воспроизводится** — actual fit
> gives $K_\infty = 35.36$. См. discrepancies file.

**Источник**: `kl_fit_fp8_e4m3.json`.

---

## §6.2 Per-channel defense, FP8 family ✓ (lab metric)

**Claim**: protect top-10 K outlier channels in bf16, quantize the rest FP8 e4m3.
Result: -34% K-error, -15% attention-shift KL.

| Quant | N | Paper K-err change | Lab (n=80) | Live (n=5) |
|---|---|---|---|---|
| fp8_e4m3 | 1 | -5.4% | -5.4% ✓ | n/a |
| fp8_e4m3 | 5 | -23.7% | -23.7% ✓ | n/a |
| **fp8_e4m3** | **10** | **-34.1%** | **-34.1%** ✓ | **-29.4%** ≈ ✓ |
| fp8_e4m3 | 25 | -42.7% | -42.7% ✓ | n/a |
| fp8_e4m3 | 100 | -57.0% | -57.0% ✓ | n/a |
| fp8_e5m2 | 10 | -33.1% | -33.1% ✓ | -29.5% ≈ ✓ |

**Per-channel defense ~30× efficient than per-layer defense**: 1% каналов
покрывают 34% ошибки, 36% layers покрывают 44% ошибки.

**Условия, при которых работает**:
- FP8 family (e4m3, e5m2) — per-tensor scale, outliers concentrated. ✓
- Lab metric (mean ‖K_pre − K_post‖_F / ‖K_pre‖_F) — direct measurement of K
  reconstruction quality. ✓
- TF mode — single forward pass через captured prompts. ✓
- **Caveat**: эта эффективность в lab metric **НЕ переносится напрямую на
  end-to-end AR generation** (см. discrepancies).

**Источник**: `per_channel_defense_fp8_e4m3.json`, `paper_validation_live.json`.

---

## §6.4 Spearman correlations of per-layer logit impact ✓ (exact)

**Claim**: Spearman ρ between per-layer mean logit-KL rankings:

|  | fp8_e4m3 | fp8_e5m2 | hqq_int4 | hqq_int2 |
|---|---|---|---|---|
| **Paper / Data** | 1.00/**1.00** | 0.52/**0.52** ✓ | 0.80/**0.80** ✓ | 0.79/**0.79** ✓ |
| hqq_int4 | 0.80/**0.80** ✓ | 0.49/**0.49** ✓ | 1.00/**1.00** | 0.92/**0.92** ✓ |

Cross-family correlation (FP8 e4m3 ↔ HQQ INT4, ρ=0.80) > within-family
(FP8 e4m3 ↔ e5m2, ρ=0.52). Это потому что e5m2 trades mantissa for range —
другой *kind* of quant.

**Источник**: computed from `layer_ablation_*.npz`.

---

## §6.4 Universal safe layer L5 ✓

**Claim** (paper): L0, L5 are top-5 worst layers under ALL four quant formats;
L23-L27 are bottom-5 safe under ALL four.

Только **L5** confirmed universally top-5 worst across all 4 formats:

| Quant | Rank of L5 (1=worst) |
|---|---|
| fp8_e4m3 | 4 ✓ |
| fp8_e5m2 | 5 ✓ |
| hqq_int4 | 2 ✓ |
| hqq_int2 | 3 ✓ |

> **Note**: L0 claim **не воспроизводится** для fp8_e5m2 (rank 18, не top-5).
> См. discrepancies. И L23-L27 универсальность тоже частична: только L24, L26
> в bottom-5 для всех 4 форматов.

**Источник**: computed from `layer_ablation_*.npz`.

---

## §6.4 Top-5 worst layers fp8_e4m3 ✓

**Claim** (paper §4.3 + §6.4): logit-impact top-5 for fp8_e4m3 = L3, L15, L0, L5, L9.

| Rank | Paper | Data |
|---|---|---|
| 1 | L3 | **L3** ✓ |
| 2 | L15 | **L15** ✓ |
| 3 | L0 | **L0** ✓ |
| 4 | L5 | **L5** ✓ |
| 5 | L9 | **L9** ✓ |

Exact match. Эта четвёрка/пятёрка — основа per-layer defense recipe (§6.1).

**Источник**: `layer_ablation_fp8_e4m3.npz`.

---

## §7 Cross-architecture outlier dilution ✓ (Qwen3-4B)

**Claim**: на более крупной модели Qwen3-4B (36 layers, 1024 channels) outlier
concentration падает: top-10 fraction = **21.0%** vs 51.7% на 1.7B.

| Metric | Paper 4B | Data 4B (n=10) |
|---|---|---|
| top-1 channel | 3.6% | **3.57%** ✓ |
| top-10 fraction | 21.0% | **20.97%** ✓ |
| #ch for 50% | 56 | **55.5** ✓ |
| #ch for 80% | 216 | **216** ✓ |

**Critical implication for deployment**: per-channel defense **становится
менее эффективным с ростом модели**. Для multi-billion моделей нужно
расширять protection set (top-50 или top-100) чтобы держать recovery.

**Условия**: Qwen3-4B, prompt-only TF capture (без generation), 10 задач AIME-24.

**Источник**: `outputs/kv_capture/qwen3-4b/analysis/outlier_channel_impact_*.json`.

---

## §8 Failure prediction is weak (negative result) ✓

**Claim**: early-window KL trajectory (n=30, 50, 100, 200 first tokens) is at
best weakly predictive of final divergence; LOO-CV AUC < 0.6.

| Quant | n_early | Paper AUC | Data AUC |
|---|---|---|---|
| fp8_e4m3 | 30 | 0.424 | **0.424** ✓ |
| fp8_e4m3 | 50 | 0.394 | **0.394** ✓ |
| fp8_e4m3 | 100 | 0.506 | **0.506** ✓ |
| fp8_e4m3 | 200 | 0.573 | **0.573** ✓ |
| fp8_e5m2 | 30 | 0.350 | **0.350** ✓ |
| fp8_e5m2 | 50 | 0.467 | **0.467** ✓ |
| fp8_e5m2 | 100 | 0.467 | **0.467** ✓ |
| fp8_e5m2 | 200 | 0.283 | **0.283** ✓ |

All numbers exact. Early-window KL — **не рекомендуем как standalone collapse
detector**.

**Источник**: `failure_prediction_*_n*.json`.

---

## Summary: что точно работает в production

1. **Diagnose your FP8 KV-cache for outliers before deploying.** Один проход
   через calibration set, identify top-10 каналов per layer — этих 10/1024
   каналов несут половину шума. Если они защищены в bf16, лабораторный K-error
   падает на 34%. (FP8 e4m3 family only — для HQQ не работает.)
2. **Don't waste effort on per-layer skipping.** 36% layers protected → 44%
   recovery. Per-channel — 1% protection → 34% recovery. 30× efficient.
3. **HQQ format уже несёт outlier-aware quantization "из коробки"** благодаря
   per-group scale. Для HQQ не нужно дополнительной защиты — она вредит.
4. **Larger models нуждаются в более широкой protection set**. Outlier
   concentration падает с масштабом (4B: top-10=21% vs 1.7B: top-10=52%).
5. **Не пытайтесь предсказать failure по early KL** — это не работает.

Условия применимости — см. table в начале файла. **Самое важное**: эти recipes
доказаны на bf16-cache + per-tensor FP8 scale. Если у вас уже SmoothQuant/AWQ
+ FP8, поведение будет другим.
