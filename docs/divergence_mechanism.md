# Математический механизм расхождения при KV-квантизации

> **Цель документа.** Объяснить ровно ту цепочку, через которую квантизационный
> шум в K/V-кэше приводит к смене next-token argmax (точка FDP). Каждое
> звено цепочки соответствует количественному анализу из `scripts/07-09_*.py`
> и набору тензоров в `outputs/kv_capture/qwen3-1.7b/`.

## 1. Постановка

Подаём идентичные токены $x_0, x_1, \dots, x_T$ в две инстанции Qwen3-1.7B:

- **bf16**: без KV-квантизации (baseline)
- **fp8** ($e4m3$ или $e5m2$): K, V после вычисления квантуются перед записью
  в кэш; чтение из кэша происходит после деквантизации обратно в bf16.

Оба прогона выполняются в режиме **teacher-forced** на одних и тех же
bf16-токенах. Единственный источник различия — квант-шум $\Delta K_\ell, \Delta V_\ell$
на каждом слое $\ell$.

Обозначения:

| Символ | Что означает |
|---|---|
| $h^{(\ell)}_t$ | hidden state на входе attention блока $\ell$, позиция $t$ |
| $q^{(\ell)}_t, k^{(\ell)}_t, v^{(\ell)}_t$ | post-projection, post-RoPE Q/K/V |
| $K^{(\ell)}_\text{pre}$ | true bf16 K-кэш в слое $\ell$ (вся последовательность) |
| $K^{(\ell)}_\text{post}$ | $\text{quant}(K^{(\ell)}_\text{pre})$ — то, что реально в кэше |
| $\Delta K^{(\ell)}$ | $K^{(\ell)}_\text{post} - K^{(\ell)}_\text{pre}$ — квант-шум |
| $a^{(\ell)}_t$ | attention-веса позиции $t$ в слое $\ell$ |
| $o^{(\ell)}_t$ | attention output позиции $t$ в слое $\ell$ |
| $l_t$ | финальные логиты в позиции $t$ (вектор размера $|\text{vocab}|$) |
| $m_t$ | bf16-margin: $l_t[i^*] - \max_{j \neq i^*} l_t[j]$, где $i^* = \arg\max l_t$ |

## 2. Цепочка распространения ошибки

### 2.1 Layer 0 (исток шума)

После проекций и RoPE имеем чистые $q_0, k_0, v_0$ (одинаковые в bf16 и fp8,
т.к. вход и веса совпадают). Квантизация применяется **только** к K, V перед
записью в кэш:

$$
K_\text{post} = \text{quant}_{fp8}(k_0), \qquad
\Delta K_0 = K_\text{post} - k_0
$$

Attention в bf16 vs fp8:

$$
a_0^{\text{bf16}} = \mathrm{softmax}\!\left(\frac{q_0 k_0^\top}{\sqrt d}\right),
\qquad
a_0^{\text{fp8}} = \mathrm{softmax}\!\left(\frac{q_0 (k_0 + \Delta K_0)^\top}{\sqrt d}\right)
$$

Разложение по Тейлору первого порядка вокруг $\Delta K_0 \to 0$:

$$
\Delta a_0 = a_0^{\text{fp8}} - a_0^{\text{bf16}}
\;\approx\;
a_0 \odot
\left(\frac{q_0 \Delta K_0^\top}{\sqrt d}
       - \sum_j a_{0,j} \cdot \frac{q_0 \Delta K_{0,j}^\top}{\sqrt d}\right)
$$

(второй член — центрирование от свойства softmax).

Output слоя 0:

$$
o_0^{\text{bf16}} = a_0^{\text{bf16}} \cdot v_0,
\qquad
o_0^{\text{fp8}} = a_0^{\text{fp8}} \cdot (v_0 + \Delta V_0)
$$

$$
\boxed{
\Delta o_0
= \underbrace{\Delta a_0 \cdot v_0}_{\text{attention shift}}
+ \underbrace{a_0^{\text{bf16}} \cdot \Delta V_0}_{\text{value shift}}
+ \underbrace{\Delta a_0 \cdot \Delta V_0}_{\text{cross} \,\sim\, O(\Delta^2)}
}
$$

В практике на Qwen3-1.7B + FP8: $\|\Delta K\|_F / \|K\|_F \sim 2{-}5\%$
(см. `summary.json`). Cross-term пренебрежимо мал, основные вклады —
attention shift и value shift в сопоставимых пропорциях.

### 2.2 Layer $\ell > 0$ (накопление)

Hidden state на входе слоя $\ell$:

$$
h^{(\ell)} = h^{(\ell-1)} + o^{(\ell-1)} \quad \text{(residual)}
$$

(в Qwen3 ещё применяется LayerNorm/RMSNorm, опускаем для ясности).

Шум по слоям складывается:

$$
\Delta h^{(\ell)} = \sum_{j=0}^{\ell-1} \Delta o^{(j)}
$$

Каждый слой $\ell$ добавляет два эффекта:

1. **Trickle-down**: $\Delta h^{(\ell)}$ распространяется на $q^{(\ell)}, k^{(\ell)}, v^{(\ell)}$,
   значит attention слоя $\ell$ возмущено даже без своей квант-ошибки.
2. **Own quant**: $\Delta K^{(\ell)}$, $\Delta V^{(\ell)}$ добавляют новую
   ошибку в этом слое.

Финальный hidden state:

$$
h^{(L)}_t = h^{(0)}_t + \sum_{\ell=0}^{L-1} o^{(\ell)}_t
\quad\Rightarrow\quad
\Delta h^{(L)}_t = \sum_{\ell=0}^{L-1} \Delta o^{(\ell)}_t
$$

В Qwen3-1.7B $L = 28$. Линейное накопление шума через 28 уровней
объясняет, почему даже 2–5% ошибка на каждом слое выливается в нетривиальное
$\Delta l_t$ на выходе.

### 2.3 Логиты и порог переключения argmax

$$
l_t = W_{\text{lm\_head}} \cdot h^{(L)}_t
\quad\Rightarrow\quad
\Delta l_t = W_{\text{lm\_head}} \cdot \Delta h^{(L)}_t
$$

bf16 выбирает токен $i^* = \arg\max l_t^{\text{bf16}}$ с margin'ом

$$
m_t = l_t^{\text{bf16}}[i^*] - \max_{j \neq i^*} l_t^{\text{bf16}}[j]
$$

fp8 выберет другой токен (т.е. **позиция $t$ становится FDP**)
тогда и только тогда, когда:

$$
\boxed{
\exists\, j \neq i^* :\;
\Delta l_t[j] - \Delta l_t[i^*] \;>\; m_t
}
$$

То есть **квант-шум перекидывает порядок top-1 vs top-2 логитов**
именно когда margin модели в этой позиции меньше суммарного логит-шума.

## 3. Что можно вычислить из сохранённых тензоров

Capture-файлы содержат: $q^{(\ell)}_{\text{pre-RoPE}}$, $K^{(\ell)}_\text{pre}$, $K^{(\ell)}_\text{post}$,
$V^{(\ell)}_\text{pre}$, $V^{(\ell)}_\text{post}$, $l_t$ для $t \in$ window
$[\text{FDP}-150, \text{FDP}+100]$.

### Прямые величины (без RoPE-реконструкции)

| Что | Формула | Скрипт |
|---|---|---|
| Per-(layer, head) Frobenius relative error $\|\Delta K\|_F/\|K\|_F$ | `compute_relative_quant_error` | `07_analyze_captures.py` |
| Logit KL trajectory $\mathrm{KL}(p^{\text{bf16}}_t \,\|\, p^{\text{fp8}}_t)$ по всем $t$ окна | `logit_kl_trajectory` | `08_divergence_mechanism.py` |
| bf16 margin trajectory $m_t$ по окну | `bf16_margin_trajectory` | `08_divergence_mechanism.py` |
| Per-layer K-noise по позициям $\|\Delta K^{(\ell)}_t\|_2$ | `per_position_kv_quant_noise` | `08_divergence_mechanism.py` |
| Outlier-channels: каналы с $|K| > 448$ или $> 57344$ | `top_outlier_channels` | `09_outliers.py` |

### Что нужно для attention shift (требует RoPE-реконструкции)

$Q$ в capture — **pre-RoPE** (выход `q_proj` до применения rotary embedding).
Чтобы посчитать $a^{(\ell)}_t = \mathrm{softmax}(q^{(\ell)}_t K^{(\ell)\top}/\sqrt d)$,
нужно сначала применить RoPE к $q^{(\ell)}_t$ с правильными позициями.
Это не сделано в первой итерации; см. `src/kvtrace/capture/analysis.py::apply_rope_q`
как точку расширения.

## 4. Ожидаемые наблюдения

Гипотезы для проверки в `08_divergence_mechanism.py`:

**H1: KL(bf16||fp8) растёт с позицией.**
Так как $\Delta h^{(L)}_t$ накапливается через все предыдущие токены (через
self-attention), ожидается монотонный рост или поступательная аккумуляция
KL вдоль окна $[FDP-150, FDP+100]$. Острый пик не ожидается до самого FDP
(где margin кончился).

**H2: На позиции FDP margin $m_{\text{FDP}}$ мал.**
По определению FDP, в этой позиции квант-шум перекинул argmax. Значит
margin был меньше шума. Ожидается, что $m_{\text{FDP}} < m_{t < \text{FDP}}$
в среднем.

**H3: Per-layer K-noise монотонно почти константен.**
Каждый слой квантуется независимо, ошибка $\|\Delta K^{(\ell)}\|/\|K^{(\ell)}\|$
определяется свойствами K на этом слое, а не предыдущими слоями.
Ожидается, что распределение K-noise по слоям — относительно плоское, с возможными
outlier-слоями (где K имеет особенно тяжёлые хвосты).

**H4: Outlier-каналы у K/V концентрируются в небольшом подмножестве каналов.**
В литературе по LLM-квантизации известно: 1–5% каналов содержат
massive activations (значения >> остальных). FP8 e4m3 их обрезает (max ±448),
порождая локальные spikes в $\Delta K$. Ожидается, что pct outliers по слоям
неоднороден.

## 5. Следующие шаги (вне scope текущего ран'а)

1. **Реализовать RoPE-реконструкцию Q** → корректные attention maps →
   реальная декомпозиция $\Delta o = \Delta a \cdot V + a \cdot \Delta V$.
2. **Per-channel outlier-induced error**: маскировать top-k каналов в $\Delta K$,
   проверить, какая доля общего шума объясняется этими каналами.
3. **Causal vs anticausal**: разделить $\Delta K$ на «K в past positions» (читает
   будущая attention) и «K в new position» (читает только она сама).
   Помогает локализовать, какие позиции K больнее всего портят attention.
