# Cross-model verification: Qwen3-1.7B vs Qwen2.5-1.5B-Instruct vs SmolLM2-1.7B-Instruct

Запуск `scripts/20_paper_validation.py` на трёх моделях:

1. **Qwen3-1.7B** (paper baseline) — Qwen3 arch (q_norm/k_norm pre-RoPE)
2. **Qwen2.5-1.5B-Instruct** — Qwen2 arch (no q_norm/k_norm), same family
3. **SmolLM2-1.7B-Instruct** — Llama-3-derived arch, **другое семейство**

Все три без изменений в коде pipeline'a.

## Условия запуска

| Параметр | Qwen3-1.7B | Qwen2.5-1.5B | SmolLM2-1.7B |
|---|---|---|---|
| HF repo | `Qwen/Qwen3-1.7B` | `Qwen/Qwen2.5-1.5B-Instruct` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| Family | Qwen3 | Qwen2 | Llama-3-derived |
| Layers | 28 | 28 | **24** |
| Hidden | 2048 | 1536 | 2048 |
| KV-heads × head_dim | 8 × 128 | 2 × 128 (GQA-2) | 8 × 64 (GQA?) |
| Channels/layer | 1024 | 256 | 512 |
| q_norm / k_norm | Yes | No | No |
| Положение RoPE-patch | applied | no-op (Qwen2 module) | no-op (Llama module) |
| Pipeline modifications | none | **none** | **none** |
| Dataset | MATH-500 fresh probs 70-74 | same | same |

---

## Side-by-side результаты

### FP8 e4m3

| Metric | Qwen3-1.7B | Qwen2.5-1.5B | **SmolLM2-1.7B** | Paper Qwen3 lab |
|---|---|---|---|---|
| layer_rel_err mean | 2.66% | 2.65% | **2.61%** | 2.65% |
| layer_rel_err max | 2.87% | 3.22% | 2.69% | 3.50% |
| top-1 ch frac (median) | 10.10% | 12.84% | **2.86%** | 10.3% |
| top-10 ch frac (median) | 43.70% | 39.08% | **17.19%** | 51.7% |
| defense N=10 K-err change | -29.4% | -25.1% | **-9.0%** | -34% |

### FP8 e5m2

| Metric | Qwen3-1.7B | Qwen2.5-1.5B | SmolLM2-1.7B | Paper |
|---|---|---|---|---|
| layer_rel_err mean | 5.28% | 5.29% | **5.16%** | 5.25% |
| top-10 ch frac (median) | 43.02% | 39.29% | **16.82%** | 50.4% |
| defense N=10 | -29.5% | -25.7% | **-8.7%** | -33% |

### HQQ INT4

| Metric | Qwen3-1.7B | Qwen2.5-1.5B | SmolLM2-1.7B | Paper |
|---|---|---|---|---|
| layer_rel_err mean | 11.10% | 11.31% | **10.26%** | 10.32% |
| top-10 ch frac (median) | 11.24% | 8.76% | **3.67%** | 13.1% |
| defense N=10 | -4.1% | -1.9% | **-1.0%** | +6% (HURTS!) |

### HQQ "uniformity ratio" (FP8 top-10 / HQQ top-10)

| Model | Ratio | Interpretation |
|---|---|---|
| Qwen3-1.7B paper lab | 51.7% / 13.1% = **3.9×** | HQQ ~4× более uniform |
| Qwen3-1.7B live | 43.7% / 11.2% = **3.9×** | confirmed |
| Qwen2.5-1.5B | 39.1% / 8.8% = **4.5×** | similar pattern |
| SmolLM2-1.7B | 17.2% / 3.7% = **4.7×** | similar pattern, smaller absolute |

**Ratio = 4-5×** на всех моделях. Это значит, что HQQ universally делает
K-noise distribution в ~4-5× более uniform чем FP8 per-tensor.

---

## Headline finding: concentration НЕ universal, magnitude — universal

### Что **architecture-invariant** (=сохраняется на всех 3 моделях)

1. **FP8 layer-Frobenius K-error magnitude**: 2.61-2.66% для e4m3, 5.16-5.28%
   для e5m2 — точно одинаково на трёх архитектурах. Определяется **только**
   FP8 round-to-nearest при типичных diapazonax K (|K| ≲ 400, well within
   e4m3 range 448).

2. **HQQ K-error magnitude**: ~10-11% INT4 на всех моделях. Определяется
   group_size=64 + 4-bit precision.

3. **Concentration ratio FP8/HQQ = 4-5×**: HQQ всегда даёт более uniform
   distribution за счёт per-group scale.

4. **Direction defense effect**:
   - FP8: defense **всегда помогает** (-9% до -29% на разных моделях)
   - HQQ: defense **по-сути нейтрален** на n=5 (-1% до -4%)

### Что **model-specific**

1. **Абсолютная величина FP8 concentration**:
   - Qwen3-1.7B: top-10 = ~44-52%
   - Qwen2.5-1.5B: top-10 = ~39%
   - SmolLM2-1.7B: top-10 = **17%** (3× меньше!)

2. **Defense efficacy масштабируется с concentration**:
   - high-concentration (Qwen3): defense N=10 recovers ~30% K-err
   - mid (Qwen2.5): recovers ~25%
   - low-concentration (SmolLM2): recovers только ~9%

   **Линейная связь**: примерно `defense_recovery ≈ 0.6 × top_10_fraction`
   во всех случаях.

3. **Layer count**: Qwen3/Qwen2.5 = 28 layers, SmolLM2 = 24 layers. Не влияет
   на per-layer metrics, но влияет на total parameter count, на который масштабируется
   capture cost.

---

## Объяснение: почему SmolLM2 имеет в 3× меньше concentration?

Outlier channels — структурное свойство выученных весов K-projection.
Появляются из обучения когда model развивает "feature detectors" в специфических
каналах, через которые проходит большая часть активации.

Возможные причины меньшей concentration в SmolLM2:

1. **Training data**: SmolLM2 trained на synthetic + curated dataset (SmolLM
   corpus), Qwen3 — на multi-modal/multi-lingual real corpora. Разные
   распределения активаций → разные emergent outlier patterns.

2. **Pre-RoPE normalization**: Qwen3 имеет `q_norm`+`k_norm` RMSNorm-слои
   **перед** RoPE. SmolLM2 (Llama-3-arch) — без них. Эти лишние norm-слои
   могут amplify outlier формирование во время training.

3. **Hidden dim distribution**: SmolLM2 GQA с head_dim=64 (vs 128 на Qwen3).
   Половина головы → outlier "spread" over больше positions внутри head_dim?
   Маловероятно.

4. **Architectural age**: SmolLM2 — современный Llama-3 derivative с
   improved training recipe (better LR schedule, better data filtering).
   Modern training может приводить к менее outlier-heavy K-distributions.

---

## Implications для production deployment

### Для **Qwen-family моделей**

- **Per-channel defense эффективна**: ожидаемое recovery 25-34% K-err при FP8.
- Берите `scripts/20_paper_validation.py` paper's recipe **as-is**.
- Protected fraction (top-10 channels) даёт хорошую compression-quality tradeoff.

### Для **Llama-family / SmolLM2 / других "modern" моделей**

- **Per-channel defense значительно слабее**: только ~9% K-err recovery.
- Скорее всего нужно **расширять protection set** до top-30 или top-50 чтобы
  получить comparable recovery (но цена в bf16 storage растёт).
- Или вообще не использовать per-channel defense, а сразу HQQ или другой
  outlier-adaptive format — он работает универсально (HQQ K-error magnitude
  одинаков на всех 3 моделях).

### Универсальная стратегия

1. **Перед deployment**: запусти script на 5-10 calibration промптах своей
   модели, посмотри top-10 fraction для FP8.
2. Если top-10 ≥ 40% → per-channel defense даст 25-35% recovery, OK.
3. Если top-10 < 25% → per-channel defense даст < 15% recovery, **не стоит**.
   Лучше использовать HQQ INT4 directly (та же magnitude error что FP8 e5m2,
   но без outlier-sensitivity).
4. На любой модели **per-channel defense с calibration via prefill вредит в
   AR** (см. discrepancy D6). Используй только в TF/scoring workloads.

---

## Универсальная Conclusion: какие claim'ы переносятся куда

| Claim | Qwen3 → Qwen2 | Qwen → Llama-3 | Qwen → DeepSeek-V3 (MLA) |
|---|---|---|---|
| FP8 K-err magnitude (2.65%) | ✅ exact | ✅ exact | n/a (MLA changes layout) |
| FP8 concentration top-10 = 51.7% | ⚠ qualitatively | ❌ **drops to 17%** | n/a |
| Per-channel defense -34% K-err | ⚠ proportional, ~25% | ❌ **only ~9%** | n/a |
| HQQ uniform pattern (top-10 ≈ 13%) | ✅ confirmed | ✅ even more uniform (3.7%) | n/a |
| HQQ defense direction | ⚠ fragile | ⚠ same fragile | n/a |
| Layer ranking (L0, L5 worst) | ❓ per-model | ❌ semantics meaningless | n/a |
| KL trajectory τ=130 | ❓ per-model | ❓ per-model | n/a |

**Bottom-line cross-family**: paper's main mechanism (FP8 K-error → softmax
amplification → logit divergence) is universal. The **numerical headline**
("top-10 channels carry 52% of noise; per-channel defense recovers 34%")
is **Qwen-specific** and **does not generalize to Llama-derived families** —
SmolLM2 shows 3× less concentration and 4× less defense efficacy.

The paper's recipe is correct, but its quantitative magnitude was over-stated
as universal. The honest formulation: **"on Qwen3-architecture models with
training data inducing concentrated K-outliers, per-channel defense recovers
30-34% K-error. On other architectures with more uniform K-distributions
(Llama-3 derivatives like SmolLM2), the same recipe recovers <10%."**
