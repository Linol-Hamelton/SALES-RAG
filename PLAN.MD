# PLAN.MD — Labus Sales RAG

Roadmap и статус реализации. Актуально на **2026-04-15** (ветка `main`, последний коммит `df0d1df`, версия 1.1.0).

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

### P8 — Feedback gates + bundle retrieval refinement
| # | Этап | Коммит-якорь | Описание |
|---|---|---|---|
| P8.8 | Feedback-replay fixes (Tier 0/2/3/4/5 gates) | `2dbdda9` | Иерархичные gates для разных confidence-тиров в feedback |
| P8.7-C | Dedicated bundle retrieval для bundle-intent queries | `41e052b` | Специализированный путь поиска для запросов о пакетах |
| P8.7-D | Out-of-scope + financial modifier hard-overrides | `ef42606` | Блокировка нерелевантных ответов вне scope |
| P8.6-B | Force manual confidence для complexity markers | `df3d04f` | Снижение confidence для сложных многопараметрических запросов |
| P8.6-A | Tighten keyword override excludes | `4e42606` | Сужение keyword override для precision |

### P7.6+ — Deploy infrastructure
- `/opt/rag/artifacts/` буфер создан
- Git-pull workflow заменил scp для всего кроме heavy artifacts
- README.md переписан под реальную раскладку контейнера (`/RAG_ANALYTICS/output/` абсолютный)

### P10.6 — Data pipeline consolidation ✅ (задеплоено 2026-04-15, commit `df0d1df`)

| Пункт | Файл | Описание |
|---|---|---|
| A2 | `buildSmetaTemplates.mjs` | `product_to_category_map` в `smeta_templates.json` |
| B1 | `ingest_offer_compositions.py` | Снят KEY_OFFER_IDS whitelist: 22 docs → **2136** |
| B2 | `ingest_knowledge.py` | Удалён дубль ROADMAPS-файла из knowledge |
| B3/B4/B6/B7 | `ingest.py` | `linked_smeta_category_ids`, `offer_id`, `related_roadmap_slugs`, `good_ids` в payload |
| B5 | `ingest_roadmaps.py` | Fuzzy-link roadmap→product/category (13/65 файлов с категориями, 76 threshold) |
| C1 | `build_index.py` | Удалён `upload_to_qdrant.py`, добавлен `--from-npy` |
| C2 | `build_index.py` | Расширены payload-индексы для новых полей |
| C3 | `build_index.py` | `SKIP_FILES` + `bridge_unresolved.jsonl` |
| D2 | `generator.py` | Удалён regex-хак в offer_profile генерации |
| D3 | `retriever.py` | Roadmap boost +0.25 при trigger-словах (POST-retrieval) |
| E1 | `refreshSalesRagData.mjs` | Единый pipeline: JS аналитика + Python ingest + reindex |
| E2 | `verifySalesRag.mjs` | Ассерты на jsonl-объёмы; default Python → `.venv` |

**Prod после P10.6 — 51 531 точек в Qdrant:**

| doc_type | Кол-во | Δ |
|---|---|---|
| product_docs | 12 574 | |
| bundle_docs | 12 560 | |
| deal_profile_docs | 9 530 | |
| photo_analysis_docs | 11 105 | |
| offer_composition_docs | **2 136** | +2114 (было 22) |
| offer_profile_docs | 2 136 | |
| faq_docs | **365** | +365 (было 0) |
| roadmap_docs | 706 | |
| knowledge_docs | 157 | |
| timeline_docs | 217 | |
| bridge_docs + прочие | 45 | |

---

## 2. Текущее состояние компонентов

| Компонент | Состояние | Примечание |
|---|---|---|
| HybridRetriever (Qdrant native) | ✅ prod | RRF α=0.7 dense; P11-R1 forced roadmap injection |
| CrossEncoderReranker | ✅ prod | heuristic boosts; feedback gates (P8.8) |
| PricingResolver (auto/guided/manual) | ✅ prod | детерминированный; financial modifiers (P8.7-D) |
| ParametricCalculator | ✅ prod | формулы из `parametric_rules.csv` |
| SmetaEngine | ✅ prod | P7.7, SEED для «Логотип»; `product_to_category_map` (P10.6-A2) |
| QueryParser + QueryDecomposer | ✅ prod | letter_count, height, m.p., area; bundle-intent path (P8.7-C) |
| Clarification parser | ✅ prod | describe/provided-smeta intent |
| DealLookup | ✅ prod | real line items из normalized |
| PhotoIndex | ✅ prod | segmented references visual enrichment; `good_ids` в payload (P10.6-B7) |
| VisionAnalyzer (Gemini + Ollama) | ✅ prod | offline ingest |
| FeedbackStore (SQLite) | ✅ prod | ratings + comments; replay gates (P8.8) |
| DeepseekGenerator | ✅ prod | graceful degradation |
| Admin UI | ✅ prod | users, chats, feedback |
| FastPanel + nginx 180s timeout | ✅ prod | ai.labus.pro |
| Offer Composition Pipeline | ✅ prod | P10.6-B1: 2136 docs вместо 22 |
| Consolidated Refresh Pipeline | ✅ prod | P10.6-E1: `node refreshSalesRagData.mjs` — единый entry point |

---

## 3. Известные проблемы

| # | Проблема | Приоритет | Где |
|---|---|---|---|
| B1 | **«Титульные вывески» canonical picker bug** — каноническая смета содержит 11 раздутых позиций (Шоколад, Табличка). Picker выбирает не ту сделку. | P2 | `buildSmetaTemplates.mjs` |
| B2 | **SEED_DEALS покрытие узкое** — только «Логотип». «Фирменный стиль», «Брендбук», «Айдентика» страдают от picker'а. | P2 | `buildSmetaTemplates.mjs` + `_KEYWORD_OVERRIDES` |
| B3 | **Eval suite не покрывает SmetaEngine** — `test_cases.json` (46 кейсов) не проверяет quality thresholds, SEED категории, parametric scaling. | P2 | `eval/test_cases.json` + `scripts/eval.py` |
| B4 | **Roadmap linking частичный** — B5 резолвит 13/65 файлов по категориям, 2/65 по продуктам. Брендированный мерч не линкуется (разрыв именования `макет кружки` vs `Брендированные кружки`). | P2 | `ingest_roadmaps.py` |
| B5 | **Deepseek dependency** — нет fallback на локальный LLM для offline-режима. | P3 | `generator.py` |
| B6 | **Нет monitoring feedback_store** — собираем, но не анализируем. | P3 | `scripts/feedback_report.py` |
| B7 | **Canonical picker overfit на малочисленные категории** — `small_penalty` 0.85 помогает, но не везде. | P3 | `smeta_engine.py` SIM_* thresholds |
| B8 | **Manual tier 12 130+ позиций** — для них цена не выдаётся; closest analogs не всегда релевантны. | P3 | `pricing_resolver.py` |
| B9 | **Vision pipeline внешняя зависимость** — Gemini через artemox; Ollama-вариант медленнее. | P3 | — |

---

## 4. Roadmap (P11+)

### P11 — Roadmap integration (в процессе)

**P11-R1 — Forced roadmap injection** ✅ ВЫПОЛНЕН (локально, ещё не задеплоен)
- `retriever.py`: новый метод `retrieve_roadmap()` — изолированный hybrid-запрос только в roadmap namespace
- `query.py`: `_force_inject_roadmap()` — при детекции trigger-слов («этапы», «как делаете», «процесс», «сроки»…) вставляет top-3 roadmap-docs в reranked с синтетически высоким score
- Root cause: POST-retrieval boost (D3) не работал, т.к. roadmap не попадал в initial top-K из-за embedding distance

**P11-R2 — Roadmap special formatting в generator.py** ✅ ВЫПОЛНЕН (локально)
- `elif doc_type == "roadmap"` блок в `_format_context_block`: структурированный формат `[Регламент: …] Этапы: …`
- LLM видит чёткую структуру «это процесс, не товар»

**P11-R3 — Eval roadmap invariant** ✅ ВЫПОЛНЕН (2026-04-15)
- `eval.py`: `check_roadmap_trigger()` — при `"roadmap_trigger": true` в кейсе ассертирует `retrieved_doc_types.roadmap ≥ 1`
- `eval.py`: новая метрика `roadmap_trigger_accuracy` в отчёте
- `eval/test_cases.json`: 6 новых кейсов категории `roadmap_process` (roadmap_01–05 с trigger, roadmap_06 negative)

**P11-R4 — Feedback-replay 23 пары** ✅ ВЫПОЛНЕН (2026-04-15)
- `eval/feedback_replay/prepare_r4_pairs.py`: отбор 23 пар из feedback_pairs.json (6 roadmap-релевантных по expert comment + 17 baseline)
- `eval/feedback_replay/measure_cite_rate.py`: новый скрипт замера cite-rate регламент/этап/сроки
- `eval/feedback_replay/analyze.py`: добавлена критика `cites_roadmap`
- **Результат R4**: cite-rate 0% → 18.2% (+18.2%), 4 improved, 0 regressed — P11 работает
- Verdict по 23 парам: 5 improved, 9 same, 5 regressed (регрессии по wrong_catalog/wrong_math, не по роадмапу)

### P8 — SmetaEngine расширение

**P8.1 — SEED_DEALS на бренд-категории**
- Добавить SEED для «Фирменный стиль», «Брендбук», «Айдентика»
- Выбрать `canonical_id` вручную из чистых сделок; настроить `enrich_regex` / `enrich_exclude`
- ROI: закрывает дыру дизайн-направления; высокий частотный диапазон в истории чатов

**P8.2 — Fix canonical picker (Титульные вывески)**
- Debug почему получает 11 раздутых позиций
- Sanity checks: max positions, max deviation от median, позиции из одного b24_section
- ROI: устраняет регрессию на короткие запросы вывесок

**P8.3 — Eval suite расширение под SmetaEngine**
- Новые категории в `test_cases.json`: `smeta_seed`, `smeta_parametric`, `smeta_quality_threshold`
- Метрики: category precision@1, total deviation от канонического
- Regression guard: «Сколько стоит логотип?» → всегда «Логотип», никогда «Титульные вывески»

### P9 — Data quality

**P9.1 — Feedback loop analysis**
- `scripts/feedback_report.py`: агрегация ratings из `labus_rag.db` по категориям, confidence tiers
- Еженедельный отчёт в `reports/feedback_weekly.md`

**P9.2 — Manual tier reduction**
- Relaxed parameters для позиций с ≥5 сделками → guided
- ROI: сократить 12 130 manual-позиций на ~20%

**P9.3 — Roadmap docs пересборка**
- Актуализировать `RAG_DATA/ROADMAPS/*.md` под текущие процедуры
- ROI: лучшие ответы на «как оформить безнал», «скидка на повтор»

### P10 — Infrastructure

**P10.1 — Local LLM fallback**
- Qwen2.5-7B-Instruct или DeepSeek-R1-Distill через vLLM
- Auto-switch если Deepseek API > 10s latency

**P10.2 — Incremental indexing**
- Инкрементальный upsert вместо `--recreate`; хэш-сравнение jsonl

**P10.3 — Prod monitoring**
- Prometheus `/metrics` + Grafana: query latency, SmetaEngine hit rate, Qdrant query time

### P12 — UX / product

**P12.1 — Streaming smeta до LLM**
- Structured smeta в stream перед Deepseek summary; ROI: perceived latency 60ms vs 870ms

**P12.2 — Clarification inline в UI**
- Кнопки «Уточнить параметры» при SmetaEngine minimal/none

**P12.3 — Управление SEED через админку**
- UI для SEED_DEALS без редактирования .mjs

---

## 5. Deploy процедуры

```bash
# Полный refresh локально (данные + аналитика + реиндекс):
node refreshSalesRagData.mjs   # использует .venv по умолчанию

# Только аналитика (без реиндекса):
node refreshSalesRagData.mjs --skip-index

# Деплой на прод — см. reference_prod_deploy_paths.md:
# 1. git push origin main
# 2. scp -P 22 -r RAG_RUNTIME/data root@62.217.178.117:/opt/rag/artifacts/
# 3. На сервере: git pull + docker cp + build_index --from-npy --qdrant-url http://labus_qdrant:6333
# 4. docker restart labus_api
```

Python venv с CUDA: `D:/SALES_RAG/RAG_RUNTIME/.venv/Scripts/python.exe` (torch 2.5.1+cu121).
