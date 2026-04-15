# PLAN.MD — Labus Sales RAG

Roadmap и статус реализации. Актуально на **2026-04-09** (ветка `main`, последний коммит `33eee65`, версия 1.1.0).

Production RAG-система для labus.pro: подбор товаров/услуг, детерминированная оценка смет через SmetaEngine, генерация КП на базе истории Bitrix24. Развёрнута на `https://ai.labus.pro`.

---

## 1. История реализации (готово)

### P1–P6 — База RAG
| # | Этап | Коммит-якорь |
|---|---|---|
| P1 | Аудит данных + knowledge docs ingest | — (исторический) |
| P2 | BGE-M3 + Qdrant + BM25 hybrid retrieval | — |
| P3 | CrossEncoder reranker + heuristic boosts | — |
| P4 | PricingResolver (auto/guided/manual) | — |
| P5 | Deepseek API генерация + prompts | — |
| P6 | FastAPI + CLI + eval suite (46 кейсов) | — |

### P6.5–P7 — Production hardening
| # | Этап | Коммит-якорь |
|---|---|---|
| | FastPanel deploy ai.labus.pro | `5806805` |
| | User registration approval + VPS sync | `f3af973` |
| | Admin UI (users, chats, feedback) | `d187261`, `f239b4e`, `453c7d1` |
| | Qdrant native hybrid search (замена кастомного RRF) | `770b9ab` |
| | Multimodal: Gemini Vision + site scraper | `1e28389` |
| | Architecture improvements (manager feedback) | `cec5167` |
| | Vision API → artemox/gemini-2.0-flash | `e49fc3d` |
| | Vision: structured prompt + offers/goods cross-ref | `7990715` |
| | Vision: local Ollama pipeline | `cb0bc40` |
| | Enriched roadmaps metadata + Gemini 3 | `8a9a52c` |
| | Segmented references (deals/offers/products) + photo enrichment + ROI anchors | `4ee8165` |
| | Clarification parser + retriever/reranker tuning | `75ce890` |

### P7 — SmetaEngine (deterministic deal estimation)
| # | Этап | Коммит-якорь |
|---|---|---|
| P7.0 | `buildSmetaTemplates.mjs` + `smeta_engine.py` + embed script | `07788eb` |
| P7.1 | Runtime hotfixes + P7.5 template rebuild | `865f160` |
| P7.6 | SEED_DEALS pattern для «Логотип» + deploy strategy doc | `03bb134` |
| P7.6 | Пересобранные артефакты (templates + embeddings) | `ec11eae` |
| P7.7-B+C | Enrichment pool для SEED + notAny фильтр «Титульных вывесок» | `3c447b6` |
| P7.7-A | Keyword override safety net для «Логотип» | `33eee65` |

### P7.6+ — Deploy infrastructure
- `/opt/rag/artifacts/` буфер создан
- Git-pull workflow заменил scp для всего кроме heavy artifacts
- README.md переписан под реальную раскладку контейнера (`/RAG_ANALYTICS/output/` абсолютный)
- reference_prod_deploy_paths.md в memory

---

## 2. Текущее состояние компонентов

| Компонент | Состояние | Примечание |
|---|---|---|
| HybridRetriever (Qdrant native) | ✅ prod | RRF α=0.7 dense |
| CrossEncoderReranker | ✅ prod | heuristic boosts |
| PricingResolver (auto/guided/manual) | ✅ prod | детерминированный |
| ParametricCalculator | ✅ prod | формулы из parametric_rules.csv |
| SmetaEngine | ✅ prod | P7.7, SEED для «Логотип» |
| QueryParser + QueryDecomposer | ✅ prod | letter_count, height, m.p., area |
| Clarification parser | ✅ prod | describe/provided-smeta intent |
| DealLookup | ✅ prod | real line items из normalized |
| PhotoIndex | ✅ prod | segmented references visual enrichment |
| VisionAnalyzer (Gemini + Ollama) | ✅ prod | offline ingest |
| FeedbackStore (SQLite) | ✅ prod | ratings + comments, RLHF-base |
| DeepseekGenerator | ✅ prod | graceful degradation |
| Admin UI | ✅ prod | users, chats, feedback |
| FastPanel + nginx 180s timeout | ✅ prod | ai.labus.pro |

---

## 3. Известные проблемы

| # | Проблема | Приоритет | Где |
|---|---|---|---|
| B1 | **«Титульные вывески» canonical picker bug** — каноническая смета содержит 11 раздутых позиций (Шоколад, Табличка). Picker выбирает не ту сделку. | P1 | `buildSmetaTemplates.mjs` |
| B2 | **SEED_DEALS покрытие узкое** — только «Логотип». «Фирменный стиль», «Брендбук», «Айдентика» страдают от picker'а. | P1 | `buildSmetaTemplates.mjs` + `_KEYWORD_OVERRIDES` |
| B3 | **Eval suite не покрывает SmetaEngine** — `test_cases.json` (46 кейсов) не проверяет quality thresholds, SEED категории, parametric scaling. | P1 | `eval/test_cases.json` + `scripts/eval.py` |
| B4 | **Deepseek dependency** — нет fallback на локальный LLM для offline-режима. | P3 | `app/core/generator.py` |
| B5 | **Нет monitoring feedback_store** — собираем, но не анализируем. | P2 | новый скрипт `scripts/feedback_report.py` |
| B6 | **Canonical picker overfit на малочисленные категории** — small_penalty 0.85 помогает, но не везде. | P2 | `smeta_engine.py` SIM_* thresholds |
| B7 | **Manual tier 12,130+ позиций** — для них цена не выдаётся; closest analogs не всегда релевантны. | P3 | `pricing_resolver.py` |
| B8 | **Vision pipeline внешняя зависимость** — Gemini через artemox; Ollama-вариант медленнее. | P3 | — |

---

## 4. Roadmap кандидатов (P8+)

Приоритизированный пул. Порядок внутри каждой группы — предпочтение.

### P8 — SmetaEngine расширение (наиболее ценное)

**P8.1 — SEED_DEALS на бренд-категории**
- Добавить SEED для «Фирменный стиль», «Брендбук», «Айдентика»
- Выбрать canonical_id вручную из чистых сделок
- Настроить `enrich_regex` / `enrich_exclude` / keyword override
- Тесты: «разработайте фирменный стиль», «брендбук с нуля», «айдентика под ключ»
- ROI: закрывает дыру дизайн-направления; высокий частотный диапазон в истории чатов

**P8.2 — Fix canonical picker в buildSmetaTemplates.mjs**
- Debug почему «Титульные вывески» получает 11 раздутых позиций
- Ввести sanity checks: max positions count, max total deviation от median, позиции должны быть из одного b24_section
- ROI: устраняет регрессию на короткие запросы вывесок

**P8.3 — Eval suite расширение под SmetaEngine**
- Новые категории в `test_cases.json`: `smeta_seed`, `smeta_parametric`, `smeta_quality_threshold`
- Метрики: category precision@1, total deviation от канонического, quality classification accuracy
- Regression guard: «Сколько стоит логотип?» должен всегда давать «Логотип», никогда «Титульные вывески»
- ROI: защита от будущих регрессий + honest baseline

### P9 — Data quality

**P9.1 — Feedback loop analysis**
- Скрипт `scripts/feedback_report.py`: агрегация ratings/comments из `labus_rag.db` по категориям, confidence tiers, SmetaEngine hit rate
- Еженедельный отчёт в `reports/feedback_weekly.md`
- ROI: видимость что менеджеры реально отвечают «плохо»/«хорошо»

**P9.2 — Manual tier reduction**
- Повторно прогнать classifyProductFact() с relaxed parameters
- Для manual позиций с ≥5 сделками в orders — повысить до guided
- ROI: сократить 12,130 manual-позиций на ~20%

**P9.3 — Roadmap docs пересборка**
- Актуализировать `RAG_DATA/roadmaps/*.md` под текущие процедуры
- Пересобрать roadmap_docs.jsonl
- ROI: лучшие ответы на «как оформить безнал», «какая скидка на повтор»

### P10 — Infrastructure

**P10.1 — Local LLM fallback**
- Поднять Qwen2.5-7B-Instruct или DeepSeek-R1-Distill через vLLM на сервере
- Настроить auto-switch если Deepseek API > 10s latency
- ROI: устраняет external dependency; latency ↓

**P10.2 — Incremental indexing**
- Вместо `build_index.py --recreate` — инкрементальный upsert изменённых docs
- Требует хэш-сравнения product_docs.jsonl
- ROI: быстрый deploy после обновления каталога

**P10.3 — Prod monitoring**
- Prometheus metrics endpoint (`/metrics`)
- Grafana dashboard: query latency, SmetaEngine hit rate, Deepseek error rate, Qdrant query time
- ROI: сейчас летаем вслепую

### P11 — UX / product

**P11.1 — Streaming smeta до LLM**
- Вернуть structured smeta в stream перед тем как Deepseek допишет summary
- Требует изменения frontend + FastAPI streaming response
- ROI: perceived latency 60ms вместо 870ms для hit-case

**P11.2 — Clarification inline в UI**
- Когда SmetaEngine minimal/none — показать кнопки «Уточнить параметры» (буквы, высота, материал)
- ROI: меньше ручной работы менеджера

**P11.3 — Управление SEED через админку**
- UI для добавления/редактирования SEED_DEALS без редактирования .mjs
- ROI: менеджеры могут калибровать без разработчика

---

## 5. Предложение по следующей итерации

**Минимальный осмысленный следующий шаг** — P8.3 (eval suite под SmetaEngine) **до** P8.1/P8.2. Почему:
- Сейчас любое изменение buildSmetaTemplates.mjs — игра вслепую. Нужен regression guard.
- P8.1 и P8.2 после eval — можно будет объективно мерить улучшения.
- Feedback loop (P9.1) даёт внешний сигнал, eval — внутренний. Оба нужны.

**Предлагаемая короткая цепочка:** P8.3 (eval) → P8.2 (fix picker) → P8.1 (SEED расширение) → P9.1 (feedback report).

Альтернативно — если острая боль сейчас в чём-то одном (например, «Титульные вывески» мешают прод-запросам), можно начать с P8.2 изолированно.

---

## 6. Deploy процедуры

См. [README.md](README.md) §3, [reference_prod_deploy_paths.md](../../.claude/projects/d--SALES-RAG/memory/reference_prod_deploy_paths.md) в memory, [RUNBOOK.md](RUNBOOK.md) для локальной разработки.
