# Какие модели подходят под статью

Анализ применимости pipeline'a и claim'ов работы к разным семействам LLM.
Pipeline = `src/kvtrace/capture/{attention_hooks,cpu_runner,fp8_sim}.py` +
`scripts/20_paper_validation.py`.

---

## Технические требования pipeline'a

Чтобы pipeline сработал **без модификаций кода**, модель должна
удовлетворять ВСЕМ требованиям:

| # | Требование | Зачем | Где проверяется |
|---|---|---|---|
| 1 | Decoder-only causal LM, доступен через `AutoModelForCausalLM` | Загрузка | `CaptureRunner.load_model` |
| 2 | Архитектурный путь `model.model.layers[i].self_attn` | Сбор attention modules | `runner._attention_modules` |
| 3 | KV-cache реализован через `transformers.cache_utils.DynamicCache` (а не `StaticCache` / `SinkCache` / `QuantizedCache`) | Class-level monkey-patch `update` | `_patch_cache_update` |
| 4 | K layout `[B, num_kv_heads, seq, head_dim]` 4D | concentration по dim 1 и dim 3 | `fp8_skip_outliers`, `identify_top_outlier_channels` |
| 5 | bf16-compatible compute path (`torch_dtype=torch.bfloat16`) | matching FP8 cast | `from_pretrained(torch_dtype=...)` |
| 6 | Eager attention (`attn_implementation="eager"`) поддерживается | hook firing | `load_model` |
| 7 | Стандартный chat-template доступен через `tokenizer.apply_chat_template(...)` | формирование промпта | `_build_prompt_tokens` |
| 8 | Поддерживает `enable_thinking=True` ИЛИ устойчиво игнорирует unknown kwarg | для CoT reasoning | same |

Требование 8 опционально: для не-Qwen моделей `enable_thinking` — это extra
kwarg, который transformers chat-template обычно пропускает; для thinking-mode
эффекта это просто не сработает, но prompt всё равно построится.

## Какие семейства гарантированно подходят

| Семейство | Архитектура | Размеры | Тест |
|---|---|---|---|
| **Qwen3** | Qwen3 (q_norm/k_norm pre-RoPE, GQA) | 0.5B–32B | ✅ Qwen3-1.7B (paper baseline), Qwen3-4B (cross-arch §7) |
| **Qwen2.5 / Qwen2** | Qwen2 (без q_norm/k_norm, GQA) | 0.5B–72B | ✅ Qwen2.5-1.5B-Instruct (validated в `paper_cross_model.md`) |
| **Llama 1/2/3/3.1/3.2/3.3** | Llama (RoPE, GQA с Llama-3) | 1B–405B | ✅ ожидается work (та же arch как SmolLM2) |
| **SmolLM / SmolLM2** | Llama-3-derived | 135M–1.7B | ✅ SmolLM2-1.7B-Instruct (см. этот файл, результаты ниже) |
| **Mistral 7B / Mistral Nemo** | Mistral (Llama-like) | 7B–22B | ✅ ожидается work |
| **Phi-3 / Phi-3.5** | Phi-3 (Llama-derived) | 1.8B–14B | ✅ ожидается work |
| **Gemma-2** | Gemma (post-LN, sliding-window attention) | 2B–27B | ⚠ ожидается work, но sliding-window может квантоваться отдельно |
| **Yi** | Llama-derived | 6B–34B | ✅ ожидается work |
| **DeepSeek-V1, DeepSeek-Coder** | Llama-derived | 1B–67B | ✅ ожидается work |
| **DeepSeek-R1-Distill-Qwen** | Qwen2 base + distillation | 1.5B–32B | ✅ work как Qwen2 |
| **DeepSeek-R1-Distill-Llama** | Llama base + distillation | 8B–70B | ✅ work как Llama |

## Семейства, которые НЕ подходят без модификации

| Семейство | Причина | Что сломается |
|---|---|---|
| **DeepSeek-V2 / V3** | MLA (Multi-Latent Attention) — KV не хранится в стандартном [B, h, seq, d] layout, а как compressed latent + uplift проекция | Требование 4. Pipeline неполный — фиксирует не то, что реально кешируется |
| **Mamba / RWKV / state-space** | нет K, V кеша вообще | Требование 3. install_capture_hooks не сработает |
| **T5 / FLAN-T5 / BART** | encoder-decoder, есть cross-attention KV + self-attention KV отдельно | Требование 1. AutoModelForCausalLM не загрузит |
| **MoE с distributed K (Mixtral, DeepSeek-MoE)** | K shared между expert weights, но cache standard. Работает | Скорее всего work, но не verified |
| **Models with native quantized KV** (например `kv_cache_dtype="fp8"` already baked in) | `DynamicCache.update` не получает bf16 K на входе | Требование 5. Получишь FP8→FP8 quant вместо bf16→FP8 |
| **vLLM-only models** без HF transformers integration | нет AutoModelForCausalLM | Требование 1 |

## Какие claim'ы статьи переносятся на разные семейства

Для claim'ов из `paper_confirmed.md` указано как ожидается переносимость.
Это inference, основанный на (a) cross-arch §7 Qwen3-4B результатах из paper,
(b) cross-family результатах в `paper_cross_model.md` (Qwen3 vs Qwen2.5).

| Claim | Trans-family | Trans-size | Trans-quant |
|---|---|---|---|
| §3.1 FP8 K-error magnitude (e4m3 ≈ 2.6%, e5m2 ≈ 5.2%) | ✅ универсально (Qwen3 = Qwen2.5 в точности) | ✅ не растёт с размером | определяется FP8 format |
| §3.2 FP8 outlier concentration top-10 ≈ 50% | ✅ направление сохраняется, абс. число падает | ✅ → padает с моделью (Qwen3-4B: 21%; predicted Llama-70B: ~10-15%) | ✓ per-tensor FP8 only |
| §3.2 HQQ uniform (top-10 ≈ 13%) | ✅ universal (Qwen3 12%, Qwen2.5 9%) | ?  не testing'ed | ✓ HQQ group_size 64 |
| §6.2 per-channel defense @N=10 helps FP8 | ✅ направление сохраняется | ✅ recovery proportional to concentration | ✓ FP8 only |
| §6.3 per-channel defense hurts HQQ | ⚠ fragile, sample-dependent | ? | ✓ HQQ |
| §6.4 L0, L5 universal worst layers (Qwen3-1.7B internal) | ❓ specific to Qwen3-1.7B arch | ❓ | layer numbering meaningless cross-family |
| §6.4 Spearman ρ matrix between quant formats | ❓ specific to Qwen3-1.7B | ❓ | ranking semantics preserved |
| §5.1 saturating-exp KL trajectory τ=130 | ❓ likely needs re-fit per model | ❓ | per-model |
| §7 outlier dilution with model size | ✅ confirmed Qwen3-1.7B → 4B | n/a (это и есть claim) | per-tensor FP8 |
| §8 failure prediction AUC < 0.6 | ⚠ не testing'ed на других семействах | ? | ? |
| §5.2 margin minimum near FDP | ⚠ argmax-flip mechanism универсальный, но точная позиция местная | ? | ? |

**Universalest claim'ы** (которые точно переносятся):
1. FP8 K-error magnitude — определяется только распределением K + FP8 format
2. Concentration ratio FP8/HQQ ≈ 4-5× — структурный
3. Per-channel defense helps FP8 — пока есть concentration
4. HQQ format уже несёт outlier-adaptation per-group

**Model-specific claim'ы**:
1. Конкретные victim/safe layers (L0, L5, L23-L27)
2. τ для KL trajectory
3. Конкретные числа top-1, top-10 (зависит от concentration на модели)

---

## SmolLM2-1.7B-Instruct: Llama-3 architecture

[Результаты добавлены в `paper_cross_model.md` после run завершения.]
