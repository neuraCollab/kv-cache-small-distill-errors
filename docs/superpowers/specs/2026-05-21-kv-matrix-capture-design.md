# KV-matrix capture для Qwen3-1.7B на CPU

**Status:** Draft, awaiting user review
**Author:** ...
**Date:** 2026-05-21
**Related:** `outputs/paper/report.pdf` (основной эксперимент)

## Цель

Сохранить полные тензоры Q, K, V и logits во время forward-pass'а Qwen3-1.7B на CPU под тремя конфигурациями KV-кэша — `bf16`, `fp8_e4m3`, `fp8_e5m2` — в окне вокруг точки первого расхождения (FDP) с bf16-трассой. Деривативные метрики (attention KL, layer-wise diff, статистика выбросов и т.д.) считаются офлайн на сохранённых тензорах вне scope этой работы.

Исследовательский вопрос: какие структурные различия в матрицах K и V возникают между bf16- и квантованным forward'ами в момент, когда модель выбирает разный следующий токен.

## Scope

### Что входит
- **Модель**: `Qwen/Qwen3-1.7B`, bf16, `enable_thinking=true`, `trust_remote_code=true`
- **Квантизации**: `bf16` (baseline), `fp8_e4m3`, `fp8_e5m2`
- **Задачи**: те же 80 задач (30 AIME-24 + 50 MATH-500), тот же seed, что в основном эксперименте
- **Окно захвата**: позиции `[FDP − 150, FDP + 100]` включительно с обеих сторон → `W ≤ 251` позиций (Python slice `[fdp_idx − 150 : fdp_idx + 101]`)
- **Режимы захвата**: teacher-forced (TF) для всех квантизаций; autoregressive (AR) дополнительно для FP8-конфигов (для bf16 AR ≡ TF)
- **Платформа**: CPU only (HF transformers; vLLM не используется)

### Что НЕ входит
- HQQ-кванты (нет существующих autoregressive-трасс; генерация на CPU нереалистична)
- DeepSeek-distilled модели (трассы коллапсируют в 100% Repetition/loop — механизм первого расхождения там тонет в петле)
- Qwen3-8B (нет существующих трасс; генерация на CPU нереалистична)
- Расчёт деривативных метрик (attention KL, divergence dynamics, plotting и т.п.) — выполняется пользователем офлайн
- Repetition-penalty sweep, multi-seed, cross-architecture — это `future work` из §7 PDF

## Архитектура

### Новый модуль
```
src/kvtrace/capture/
  __init__.py
  fp8_sim.py          # fp8_e4m3, fp8_e5m2 quant→dequant
  attention_hooks.py  # forward-hook installer + KV-cache подмена
  cpu_runner.py       # orchestrator: load Qwen3-1.7B на CPU, TF/AR
  storage.py          # safetensors writer + loader helpers

scripts/
  06_capture_kv.py    # main entry point
```

### Data flow per (problem, quant, mode)

```
existing artifacts                   new pipeline                       output
─────────────────                    ────────────                       ──────
traces/qwen3-1.7b_bf16.jsonl     ─┐
fdps/qwen3-1.7b_fp8_*.jsonl      ─┤
                                  │
                                  ▼
                       load Qwen3-1.7B (bf16, CPU)
                       install hooks (Q, K_pre, V_pre, K_post, V_post, logits)
                       configure quant_sim (None | fp8_e4m3 | fp8_e5m2)
                                  │
                           ┌──────┴──────┐
                           ▼             ▼
                  TF: bf16-tokens   AR: prefix + 250 generated steps
                  [W positions]     (KV-quant в loop)
                  1 forward pass     greedy decode
                           │             │
                           └──────┬──────┘
                                  ▼
                   slice к [FDP−150, FDP+100]
                                  ▼
              outputs/kv_capture/qwen3-1.7b/<quant>_<mode>/<problem_id>.safetensors
```

### Tensor layout файла

Формат: `safetensors` (atomic write, mmap-friendly), fp16 precision.

```
meta (JSON в header):
  model         = "qwen3-1.7b"
  quant         ∈ {"bf16", "fp8_e4m3", "fp8_e5m2"}
  mode          ∈ {"tf", "ar"}
  problem_id    = int
  fdp_token_idx = int (из существующего FDP-файла; для bf16 — referenced fdp_idx от fp8_e4m3)
  window_start  = int
  window_end    = int
  W             = window_end − window_start
  input_token_ids  = [W ints]
  gen_token_ids    = [W ints]   # AR only; для TF идентичен input_token_ids
  truncated_left   = bool
  truncated_right  = bool
  early_eos        = bool        # AR only
  pytorch_version, transformers_version, model_revision_hash, run_timestamp

per layer ℓ ∈ [0..27]:
  q_ℓ              [W, 16, 128]   # 16 attention heads, post-RoPE
  k_pre_quant_ℓ    [W,  8, 128]   # 8 KV heads (GQA), bf16 reference
  v_pre_quant_ℓ    [W,  8, 128]
  k_post_quant_ℓ   [W,  8, 128]   # = k_pre для bf16; quant→dequant для fp8
  v_post_quant_ℓ   [W,  8, 128]

logits             [W, 151936]    # vocab Qwen3
```

Для `bf16` поля `k_post == k_pre` (для единообразия loader'а).

### Use of HF transformers, не vLLM
- vLLM CPU-build не поддерживает Qwen3 и не отдаёт K/V с per-layer/per-token гранулярностью
- HF transformers даёт прямой доступ к attention modules и `DynamicCache` через хуки

### Use of хуков, не custom attention module
- Qwen3 имеет специфичный rotary embedding и sliding-window attention; переписывать forward целиком — источник багов
- Forward hooks + targeted замена K/V в `DynamicCache.update()` = минимум кода, максимум compatibility
- Конкретный механизм: `forward_hook` на каждом `Qwen3Attention` block перехватывает (Q, K, V, attn_output) пост-attention; pre-cache подмена K/V реализована через monkey-patch метода `DynamicCache.update` — оригинальный `update` вызывается с `key_states = quant_fn(key_states)` и `value_states = quant_fn(value_states)`

## FP8 симуляторы

### Реализация
```python
# src/kvtrace/capture/fp8_sim.py
import torch

def fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """bf16 -> fp8_e4m3 -> bf16. Идентично vLLM/HF KV-quant в e4m3."""
    return x.to(torch.float8_e4m3fn).to(x.dtype)

def fp8_e5m2(x: torch.Tensor) -> torch.Tensor:
    """bf16 -> fp8_e5m2 -> bf16."""
    return x.to(torch.float8_e5m2).to(x.dtype)

QUANT_FNS = {
    "bf16":     lambda x: x,
    "fp8_e4m3": fp8_e4m3,
    "fp8_e5m2": fp8_e5m2,
}
```

PyTorch 2.1+ имеет нативные `torch.float8_e4m3fn` и `torch.float8_e5m2`. CPU cast реализует IEEE FP8 round-to-nearest-even — ту же спецификацию, что vLLM-кёрнел.

### Явные негативные решения
- Не используем per-channel/per-token scaling — vLLM 0.7.3 default использует per-tensor scaling = 1 для FP8 KV; воспроизводим default
- Не реализуем свою bit-manipulation — PyTorch native достаточен
- Не реализуем stochastic rounding — vLLM и PyTorch оба используют round-to-nearest-even

### Верификация (golden test)
```python
# tests/test_fp8_sim_matches_vllm.py
def test_fp8_e4m3_matches_existing_vllm_trace():
    """CPU FP8 sim первого decode-токена должен совпасть с vLLM из исходного рана."""
    prompt_tokens, expected_first_token = load_existing_artifact(
        "outputs/traces/qwen3-1.7b_fp8_e4m3.jsonl", problem_id=0
    )
    out = cpu_runner.forward_with_quant(prompt_tokens, quant="fp8_e4m3")
    assert out.first_decode_argmax == expected_first_token, (
        "CPU FP8 sim расходится с vLLM на первом decode-токене — симулятор неверен"
    )
```

Если тест fails — симулятор имеет арифметический баг, K/V дампы методологически невалидны до фикса. Запускается **до** full run.

## Capture modes

### Teacher-Forced (TF) — для всех квантизаций

1. `T = len(bf16_trace_tokens)`; `we = min(T, fdp_idx + 101)`; `ws = max(0, fdp_idx − 150)`
2. `input_tokens = bf16_trace[0 : we]`
3. Один forward pass через Qwen3-1.7B с hook'ами и активным `quant_sim`:
   - На каждом attention-слое: K_pre, V_pre захвачены до квантизации
   - K_post = `quant_fn(K_pre)`, V_post = `quant_fn(V_pre)` — записываются в `DynamicCache`
   - Q (после RoPE), K_pre, V_pre, K_post, V_post → hook собирает все
   - На выходе: logits per position
4. Slice все собранные тензоры к `[ws : we]` → `W = we − ws ≤ 251`
4. Save: `outputs/kv_capture/qwen3-1.7b/<quant>_tf/<problem>.safetensors`

Input одинаков для всех 3 квантизаций — bf16-trace tokens. Это и есть смысл teacher-forcing: изолировать "квант на тех же токенах считает другой K/V" от "квант видит другие токены".

Для `bf16` baseline `fdp_idx` берётся от fp8_e4m3 (репрезентативный quant) — это координата slice'а; bf16 capture даёт ground-truth K/V в той же позиции, где FP8 расходится.

### Autoregressive (AR) — только FP8 (e4m3, e5m2)

Для `bf16` AR ≡ TF, отдельно не делаем.

1. `prefix_tokens = bf16_trace[0 : fdp_idx − 150]`
2. Prefill: forward pass на prefix → `DynamicCache` заполнен квантованными K/V
3. Generate: `max_new_tokens = 250`, greedy (`do_sample=False`, `temperature=0`), `repetition_penalty=1.05`
   - На каждом decode-шаге hook захватывает Q, K_pre, V_pre, K_post, V_post, logits для нового токена
4. Window = последние 150 позиций prefix + 250 generated tokens, итого `W ≤ 250`
5. Save: `outputs/kv_capture/qwen3-1.7b/<quant>_ar/<problem>.safetensors`

CPU AR может декодировать токены, отличные от GPU-trace из основного эксперимента (CPU fp8 sim ≠ GPU fp8 kernel bit-by-bit на ранних позициях). `gen_token_ids` сохраняем явно; не предполагаем совпадения с `outputs/traces/qwen3-1.7b_<quant>.jsonl`.

### Edge cases

| Случай | Поведение |
|---|---|
| `fdp_idx < 150` | `ws = 0`; `W < 251`; `truncated_left = true` |
| `fdp_idx + 101 > T` | `we = T`; `truncated_right = true` |
| FDP отсутствует (`both_correct`/`both_wrong`/`no_boxed`) | Skip; запись в `_skipped.jsonl` с reason |
| AR: `<eos>` до 250 шагов | Stop; `early_eos = true` |
| AR: prefix > `max_position_embeddings` | Skip; reason = "prefix_too_long" |
| TF: trace tokens > context window | Truncate input к `[max(0, we − max_ctx) : we]`; `prefix_truncated = true` |

### Sliding window attention test

Qwen3 использует sliding-window attention. Hook должен корректно работать когда:
- Кеш меньше window size → full attention
- Кеш ≥ window size → sliding window
- Pre-fill vs decode — два разных code path в HF Qwen3

Тест:
```python
def test_hooks_capture_under_sliding_window():
    long_prefix = torch.tensor([...]).repeat(...)  # ≥ window_size
    capture = cpu_runner.run_tf(long_prefix, quant="fp8_e4m3")
    assert capture.q[0].shape[0] == W
    assert torch.equal(capture.k_post[0], fp8_sim.fp8_e4m3(capture.k_pre[0]))
```

## Storage budget

Qwen3-1.7B: 28 слоёв, 16 attention heads, 8 KV heads (GQA), head_dim = 128, vocab = 151936.

Размер на позицию (fp16):

| Тензор | Shape | Bytes/pos |
|---|---|---|
| Q | [16, 128] | 4 096 |
| K_pre | [8, 128] | 2 048 |
| V_pre | [8, 128] | 2 048 |
| K_post | [8, 128] | 2 048 |
| V_post | [8, 128] | 2 048 |
| На слой | — | 12 288 |
| × 28 слоёв | — | 344 064 |
| logits | [151936] | 303 872 |
| **Per-position total** | — | **~648 KB** |

`W = 251` → один capture-файл ≈ **160 MB**

### Полный бюджет диска

| Конфигурация | Файлов | Размер |
|---|---|---|
| bf16, TF | 80 | 12.6 GB |
| fp8_e4m3, TF | 80 | 12.6 GB |
| fp8_e4m3, AR | 80 | 12.6 GB |
| fp8_e5m2, TF | 80 | 12.6 GB |
| fp8_e5m2, AR | 80 | 12.6 GB |
| **Всего** | **400** | **~63 GB** |

Минус skipped: по Table 5 PDF FDP вычислен для всех 80 задач × оба fp8 конфига → ожидаемая skip rate < 5% (только trace-tokens shorter than 251 → truncated, но не skipped). Realistic: **~60 GB**.

Опциональные рычаги сокращения (флаги CLI, default off):
- `--logits-topk 200` — экономит ~95% места на logits → итог ~25 GB
- `--window-pre 50 --window-post 50` — экономит ~60% → итог ~25 GB

Default — full storage.

## CPU wall-clock budget

Ориентир: Qwen3-1.7B bf16 на современном desktop CPU (8-12 ядер).

| Этап | Per (problem, quant) | Total |
|---|---|---|
| TF capture (bf16) | ~45 сек × 80 | ~1 ч |
| TF capture (fp8 × 2) | ~45 сек × 80 × 2 | ~2 ч |
| AR capture (fp8 × 2) | (prefill ~30c + 250 × 200мс) × 80 × 2 | ~3.5 ч |
| **Total** | — | **~6-8 ч** |

Полный пайплайн помещается в один рабочий день. Smoke-тесты на 1-2 problem перед full run — ~30 минут.

## Integration

### Phase 6 в `scripts/run_all.sh`

```
Phase 1 (generate)   -> outputs/traces/...
Phase 2 (find_fdp)   -> outputs/fdps/...
Phase 3 (judge)      -> outputs/judgments/...
Phase 4 (analyze)    -> outputs/report.md, outputs/plots/
Phase 5 (post-hoc)   -> outputs/paper/...
Phase 6 (kv_capture) -> outputs/kv_capture/qwen3-1.7b/<quant>_<mode>/<problem>.safetensors   [NEW]
```

Phase 6 — независимая фаза, требует только готовых артефактов Phase 1 + Phase 2.

### CLI

```
python scripts/06_capture_kv.py \
  --model qwen3-1.7b \
  --quants bf16 fp8_e4m3 fp8_e5m2 \
  --modes tf ar \
  --problems all \
  --window-pre 150 --window-post 100 \
  --output-dir outputs/kv_capture/
```

Дополнительные флаги:
- `--problems 5,8,14` — debug subset
- `--dry-run` — печать план без запуска
- `--resume` — пропустить уже существующие capture-файлы
- `--logits-topk N` — top-k compression logits
- `--smoke` — 2 problems × 1 quant × 1 mode, для CI

### CI и тесты

Новые тестовые файлы:

```
tests/test_fp8_sim.py                  # идемпотентность, NaN handling
tests/test_fp8_sim_matches_vllm.py     # golden против существующего vLLM-артефакта
tests/test_attention_hooks.py          # хук capture'ит правильные тензоры на toy model
tests/test_capture_window.py           # edge cases для slicing окна
tests/test_capture_sliding_window.py   # Qwen3 sliding window
tests/test_storage_roundtrip.py        # safetensors save/load preserves shapes/dtypes
tests/test_capture_smoke.py            # end-to-end 1 problem на маленькой модели
```

Цель покрытия: ≥ 85% (тот же стандарт, что в §8.4 PDF).

Маркеры pytest:
- Все новые тесты — `@pytest.mark.cpu_capture`, прогоняются в CI
- Smoke-тест с Qwen3-1.7B — `@pytest.mark.slow`, в CI `skip` (требует 3GB модель), запускается локально перед release

### Артефакты для воспроизводимости

- `outputs/kv_capture/qwen3-1.7b/_run_metadata.json` — версии PyTorch, transformers, hash модели Qwen3-1.7B, hash коммита, run timestamp
- `outputs/kv_capture/qwen3-1.7b/_skipped.jsonl` — список skipped (problem, quant, mode, reason)
- Дампы НЕ выгружаются на HuggingFace Hub автоматически (~60GB > soft limit). Опциональная загрузка через отдельный script с явным флагом.

## Ограничения (попадут в раздел Limitations будущего отчёта)

1. **CPU FP8 cast vs vLLM GPU FP8 kernel** — арифметически идентично (тот же IEEE FP8 стандарт), но порядок операций в attention compute может слегка отличаться → начиная с некоторого decode-шага AR-trajectory на CPU может разойтись с GPU-trajectory из основного эксперимента. В FDP-окне (медианно ~500-700 токенов от старта по Table 5 PDF) расхождение должно быть минимальным; валидируется `tests/test_fp8_sim_matches_vllm.py` (первый decode-токен должен совпасть).
2. **Greedy decoding** в AR — нет оценки variance.
3. **Один seed** (тот же, что в основном эксперименте) — нет sampling variance.
4. **Окно ±150/+100** может не покрыть длиннодистанционное накопление ошибки — расширение требует увеличения диска и/или top-k logits.
5. **`finish_reason` HF-генератор** ограничения те же, что в §3.4 PDF — `always "stop"` для AR. Прокси-метрика — `generated_tokens == max_new_tokens`.
6. **Sliding window** Qwen3 — проверяется тестом, но при выходе за `max_position_embeddings` capture skipped.

## Acceptance criteria

Этот spec считается выполненным когда:
- [ ] `tests/test_fp8_sim_matches_vllm.py` проходит (CPU FP8 sim воспроизводит vLLM первый decode-токен на existing artifact)
- [ ] `python scripts/06_capture_kv.py --smoke` отрабатывает end-to-end за ≤ 5 минут
- [ ] Full run собирает ≥ 380 capture-файлов (400 минус < 5% skipped по `_skipped.jsonl`) за ≤ 10 часов CPU
- [ ] Все 7 новых тест-файлов проходят
- [ ] Coverage нового кода ≥ 85%
- [ ] `outputs/kv_capture/qwen3-1.7b/_run_metadata.json` корректно фиксирует версии и hash коммита
- [ ] Loader helper `kvtrace.capture.storage.load(path)` восстанавливает все тензоры с правильными shapes/dtypes на independent Python session
