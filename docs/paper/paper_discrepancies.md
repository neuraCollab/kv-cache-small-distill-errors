# Расхождения между claim'ами статьи и данными

Это найденные при практической верификации **расхождения** между значениями,
заявленными в `docs/paper/draft.md`, и тем, что реально показывает
capture-pipeline и end-to-end AR generation на живой модели Qwen3-1.7B.

Расхождения **не код-баги** — это либо неточная формулировка в статье, либо
случай "lab-metric не переносится в production AR".

---

## D1. ⚠️ "L0 is universal worst layer" — false для fp8_e5m2

**Paper §6.4 claim**:
> Universal worst layers (top-5 in *all* four formats): **L0, L5**.

**Что в данных**:

| Quant | Rank L0 | Rank L5 |
|---|---|---|
| fp8_e4m3 | 3/28 ✓ top-5 | 4/28 ✓ |
| fp8_e5m2 | **18/28** ✗ НЕ top-5 | 5/28 ✓ |
| hqq_int4 | 1/28 ✓ | 2/28 ✓ |
| hqq_int2 | 1/28 ✓ | 3/28 ✓ |

**Корректная формулировка**: только **L5 универсально top-5 worst** во всех 4
форматах. L0 в top-5 только для **3 из 4** форматов (всех кроме fp8_e5m2).

**Условия, когда L0 ≠ worst**: специфично для **fp8_e5m2** (5-битная экспонента
trades mantissa for range). На fp8_e5m2 worst-5 = [L6, L16, L12, L11, L5] — это
совсем другая структура (середина сети). Это согласуется с self-обнаружением
paper'a: "fp8_e5m2 — другой *kind* of quant, ρ=0.52 с e4m3".

**Источник**: `layer_ablation_fp8_e5m2.npz`, computed.

---

## D2. ⚠️ "L23-L27 universally safe" — overstated

**Paper §6.4 claim**:
> Universal safe layers (bottom-5 in all four): **L23, L24, L25, L26, L27**.

**Что в данных** (bottom-5 per quant):

| Quant | Bottom-5 safe |
|---|---|
| fp8_e4m3 | L21, L24, L25, L26, L27 |
| fp8_e5m2 | L21, L22, L24, L26, L27 |
| hqq_int4 | L21, L23, L24, L25, L26 |
| hqq_int2 | L23, L24, L25, L26, L27 |

**Intersection** = только **{L24, L26}**, не {L23, L24, L25, L26, L27}.

**Корректная формулировка**: только **L24 и L26** — universally bottom-5 safe
across all 4 formats. L23 пропадает для FP8 family; L25 пропадает для fp8_e5m2;
L27 пропадает для hqq_int4; L21 alternates присутствует/нет.

**Условия применимости softer claim'а**: если рассматривать **bottom-10**
вместо bottom-5, то L23-L27 действительно все попадают для всех 4 формата
(L21 тоже попадает в bottom-10 везде). Так что claim верен на уровне "safe
half" сети, но не на уровне strict bottom-5.

**Источник**: same as D1.

---

## D3. ⚠️ K_∞ ≈ 18 nats — actual K_∞ = 35 nats

**Paper §5.1 claim**:
> $K_\infty \approx 18$ nats, $\tau = 130$ tokens, $R^2 = 0.80$ on the median.

**Что в данных** (saturating-exp fit для fp8_e4m3 @ T=0, median over 80 problems):

| Param | Paper | Data |
|---|---|---|
| K∞ | ≈18 | **35.36** ✗ (~2× больше) |
| τ | 130 | **129.87** ✓ |
| R² | 0.80 | **0.796** ✓ |

τ (характерное время) и R² совпадают. Сам параметр saturation плато $K_\infty$
вдвое больше paper claim'а. Возможные причины:
- statja написана с устаревшим fit'ом (раньше τ был корректнее посчитан, K∞
  ещё нет).
- разный intermediate vocab subset для KL подсчёта.

**Качественный claim "fp8 и bf16 essentially uncorrelated после ~130 шагов"**
остаётся верным независимо от K_∞.

**Источник**: `kl_fit_fp8_e4m3.json`.

---

## D4. ⚠️ FDP-1 — minimum margin position; вместо этого FDP-10

**Paper §5.2 claim**:
> The minimum is at **FDP-1** — exactly the position where the next-token
> decision is most fragile. Median margin trajectory: FDP-1 = **2.19 nats**.

**Что в данных** (median margin over 80 problems, window centered on FDP):

| Position | Paper median | Data median |
|---|---|---|
| FDP-50 | 7.38 | 5.72 |
| **FDP-10** | 2.62 | **2.62** (наш min ✓) |
| **FDP-1** | **2.19** | **3.62** (не min) |
| FDP+0 | 4.00 | 4.39 |
| FDP+1 | 5.62 | 5.37 |
| FDP+10 | 5.31 | 3.62 |
| FDP+50 | 4.94 | 6.24 |

**Корректная формулировка**: minimum margin не в FDP-1, а в **FDP-10**
(2.62 nats). FDP-1 показывает 3.62 nats — выше минимума.

**Качественное наблюдение остаётся**: margin dips близко к FDP. Но точная
позиция минимума сдвинута на 10 шагов раньше FDP, не на 1.

Возможные причины:
- разная FDP definition / re-captured window
- разный subset вопросов (40 vs 80)
- эффект window-shifting во время re-capture с q_post_rope

**Источник**: `margin_trajectory_bf16.npz`.

---

## D5. ⚠️ HQQ defense direction (+6% increase) — fragile

**Paper §6.3 claim**:
> For both HQQ INT4 and INT2, the same recipe **increases** the error:
> hqq_int4 K-err change = **+6%** (worse), hqq_int2 = +7%.

**Lab данные (n=80, TF)**: confirmed ✓
- hqq_int4 N=10: 10.32% → 10.91%, change = **+5.66%** ✓
- hqq_int2 N=10: 60.72% → 64.79%, change = **+6.71%** ✓

**Live данные (n=5, fresh probs 70-74)**: ✗ обратный знак
- hqq_int4 N=10: 11.10% → 10.64%, change = **-4.1%** (HELPS, not HURTS)

**Что это значит**: HQQ defense direction — **очень small-sample-sensitive**.
На 80 задачах эффект +6% (вред), на 5 фресс задачах −4% (польза). Знак
эффекта flips — значит, абсолютная величина эффекта в диапазоне ±5%, ниже
sample noise.

**Корректная формулировка**: per-channel defense на HQQ — **не помогает
существенно**. Чтобы получить устойчивое заключение нужны >>80 задач. Текущая
формулировка "increases error" верна на average, но не robust.

**Условия, при которых HQQ defense direction всё-таки HURTS**:
- group_size=64 (paper-tested)
- nbits ∈ {4, 2}
- n_problems ≥ 80 чтобы усреднить sampling noise

**Источник**: `per_channel_defense_hqq_int4.json`, `paper_validation_live.json`.

---

## D6. 🚨 (CRITICAL) Per-channel defense — lab effect not realized in AR

**Paper §6.2 claim** (lab metric):
> Per-channel defense (top-10 outliers) recovers **34% of FP8 K-error** and
> **15% of attention-map shift** — at a cost of 1% bf16 storage.

**End-to-end AR validation** (`scripts/19_defense_validation.py`, n=10 fresh
MATH-500 probs 60-69, max 100 generated tokens, greedy):

| Metric | fp8_e4m3 baseline | fp8_e4m3_top10 (stateless per-step) | fp8_e4m3_top10 calibrated (SmoothQuant-style) |
|---|---|---|---|
| Mean first divergence step | 57.5 | 58.9 (+1.4) | 52.4 (**-5.1**) |
| Token agreement @100 | 0.658 | 0.668 (+0.010) | 0.600 (**-0.058**) |
| Mean logit KL vs bf16 | 8.7493 | 8.5393 (**-2.4%**) | 10.3289 (**+18.1%**) |

**Что это значит**:
- **Stateless per-step defense** (paper's recipe applied directly during AR
  generation): recovery в logit KL только **-2.4%** vs paper's lab claim
  **-15%** attn-KL. На уровне token-agreement улучшение только +1% (внутри noise).
- **SmoothQuant-style calibrated defense** (outliers зафиксированы один раз из
  prefill K, потом фиксированы на всех decode шагах): **становится хуже**
  (+18% logit KL, -5 steps agreement). То есть calibration на prefill **не
  предсказывает** outliers нужные при decode.

**Mechanism gap lab → AR**:
- Lab metric измеряет single-position K reconstruction: ‖K_pre - K_post‖_F.
- Через softmax → attention → hidden state → logits → next-token — это
  compounding cascade. K-error -34% превращается в logit-KL recovery ~3%.
- В AR при greedy generation один flipped token меняет prefix всех последующих
  K_pre тензоров — outliers shift relative to calibration set.
- Защита top-10 outliers на decode K (size [1, 8, 1, 128] на шаг) — выбор
  outliers очень нестабилен.

**Корректная формулировка**: per-channel defense — **lab-metric effective,
production-AR fragile**. Заявление "recovers 34% K-error" верно для single
forward pass на static input, но end-to-end AR показывает максимум **~2-3%
logit KL improvement**.

**Условия, при которых defense работает в production**:
1. **TF / single-forward mode** (long-context retrieval, scoring): lab metric
   directly applicable, defense daст -34%/-15%. ✓
2. **AR generation**: **lab metric не предсказывает** end-to-end gain.
   Per-step stateless defense даёт +1-3% improvement в logit KL maximum,
   с high per-problem variance (std = 25.3 step delay при mean +1.4).
3. **SmoothQuant calibrated**: **NOT recommended для AR** — calibration на
   prefill не предсказывает decode-step outliers, эффект отрицательный.

**Возможные пути исправления (не реализованы)**:
- Adaptive top-K: при каждом decode шаге re-identify outliers на cumulative
  K-cache, не на текущем токене (требует accumulated state, нарушает hooks).
- Hybrid: protect outliers из prefill calibration ПЛЮС top-1 текущего decode
  step. Untested.
- Просто шире protection set (top-50 или top-100) — даст -49% K-err в лабе,
  но не очевидно что AR ratio улучшится пропорционально.

**Источник**: `defense_validation_e2e.json`.

---

## Summary: верифицированный shortlist что **не использовать** as-is

1. **L0 как universal worst layer** — proven only для 3/4 quant formats.
   Используй L5 вместо.
2. **L23-L27 как universal safe** — proven только L24, L26. Используй
   bottom-10 для строгой универсальности.
3. **Margin minimum at FDP-1** — sample-dependent. Качественный pattern
   ("margin dips near FDP") верен, точная позиция — нет.
4. **K_∞ = 18 nats** — actual 35. τ corrected, но плато tracking misstated.
5. **HQQ defense HURTS by 6%** — fragile claim, flips sign на малых n.
6. 🚨 **Per-channel defense -34% K-err = -15% attn-KL** — lab metric, **не
   AR mode**. End-to-end AR показывает **~2.4%** logit KL recovery. Никогда
   не используй calibrated SmoothQuant-style version для AR — она вредит.

Все остальные claim'ы (§3.1, §3.2 концентрация, §3.3 reproducibility, §4.1
attention KL, §6.4 Spearman, §7 cross-arch dilution, §8 failure prediction)
**воспроизводятся в точности** — см. `paper_confirmed.md`.
