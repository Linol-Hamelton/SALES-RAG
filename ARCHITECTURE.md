# Labus Sales RAG — Architecture

Production RAG-система для labus.pro. Локальная обработка retrieval + детерминированный pricing; Deepseek API только для суммаризации и рассуждений. Multimodal enrichment через Gemini/Ollama Vision.

**Version:** 1.1.0 (P7.7) · **Stack:** FastAPI · Qdrant · BGE-M3 · BGE-Reranker-v2-m3 · Deepseek API · SQLite · Node.js analytics

---

## 1. Data flow (high level)

```
                                Client (ai.labus.pro/chat)
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  FastAPI (app.main) · workers=1 · lifespan loads all models              │
│  Routers: auth · chats · query · admin · eval · health                   │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ /query_structured
                                 ▼
              ┌─────────────────────────────────────────┐
              │  QueryParser  →  QueryDecomposer         │
              │  (direction, intent, budget)             │
              │  (letter_count, height_cm, linear_m, …)  │
              └──────────┬─────────────────────┬─────────┘
                         │                     │
              ┌──────────▼──────────┐  ┌───────▼──────────────┐
              │  Clarification path │  │  SmetaEngine          │
              │  (describe/provided │  │  (template-based      │
              │   smeta → LLM)      │  │   deterministic       │
              └─────────────────────┘  │   estimator)          │
                                       └───────┬──────────────┘
                                               │ match quality
                                               │
                        ┌──────────────────────┼──────────────────────┐
                        │                      │                      │
                        ▼                      ▼                      ▼
                  maximal / significant    minimal / none       junk / under-key
                  → use template           → try next           → fallback to
                                                                   HybridRetriever
                                                                   pipeline (legacy)
                                               │
                                               ▼
              ┌─────────────────────────────────────────────────┐
              │  HybridRetriever (Qdrant native RRF)             │
              │  • BGE-M3 dense (1024d) + BM25 sparse            │
              │  • Payload filters: doc_type, direction,         │
              │    price_mode, confidence_tier                   │
              └──────────────────────┬──────────────────────────┘
                                     ▼
              ┌─────────────────────────────────────────────────┐
              │  CrossEncoderReranker (bge-reranker-v2-m3)       │
              │  + heuristic boosts (direction, auto-tier, …)    │
              └──────────────────────┬──────────────────────────┘
                                     ▼
              ┌─────────────────────────────────────────────────┐
              │  PricingResolver (auto / guided / manual)        │
              │  • ParametricCalculator (площадь, буквы, м.п.)   │
              │  • DealLookup (real line items from normalized)  │
              │  • PhotoIndex (photo enrichment)                 │
              └──────────────────────┬──────────────────────────┘
                                     ▼
              ┌─────────────────────────────────────────────────┐
              │  DeepseekGenerator — summary + reasoning         │
              │  (graceful degradation если API упал)            │
              └──────────────────────┬──────────────────────────┘
                                     ▼
              ┌─────────────────────────────────────────────────┐
              │  StructuredResponse: pricing, bundle, refs,      │
              │  smeta, flags, risks, segmented references,      │
              │  source_distinction                              │
              └─────────────────────────────────────────────────┘

                Side channels:
                • FeedbackStore (SQLite) — ratings, comments
                • Chat history (SQLite labus_rag.db)
                • Logs (structured JSON)
```

---

## 2. Компоненты

### 2.1 QueryParser ([app/core/query_parser.py](RAG_RUNTIME/app/core/query_parser.py))

Regex + словари направлений. Без LLM. Извлекает: `direction`, `intent` (product/bundle/policy/general), `budget`, `quantity`, `is_financial_modifier`.

### 2.2 QueryDecomposer ([app/core/query_decomposer.py](RAG_RUNTIME/app/core/query_decomposer.py))

Извлекает численные параметры для scaling канонических смет:
- `letter_count` — число букв/символов
- `height_cm` — высота букв / баннера
- `linear_meters` — погонаж (плинтус, кабель)
- `area_sqm` — площадь (баннер, пленка)

Результат `decomp={...}` передаётся в SmetaEngine → `_scale_quantity()` домножает per-letter позиции.

### 2.3 SmetaEngine ([app/core/smeta_engine.py](RAG_RUNTIME/app/core/smeta_engine.py)) — **P7 cornerstone**

Detremрминированный estimator поверх офлайн-построенных шаблонов категорий.

**Офлайн (на build-host):**
1. `RAG_ANALYTICS/buildSmetaTemplates.mjs` кластеризует сделки в категории labus-таксономии, строит `canonical_smeta` (SEED_DEALS или p60 picker), агрегирует `price_stats` (mean/std/confidence), извлекает `keywords_text` из enrichment pool. Output: `smeta_templates.json`.
2. `RAG_RUNTIME/scripts/embed_smeta_categories.py` — BGE-M3 embed для `keywords_text` каждой категории. Output: `smeta_category_embeddings.npy` (N×1024, L2-нормализованные).

**Онлайн:**
1. **Keyword override** (pre-semantic safety net, P7.7-A) — regex include/exclude форсит конкретную категорию минуя cosine-ранжирование. Нужно для коротких запросов («Сколько стоит логотип?»), где семантика перебивается bias log10(n).
2. **Cosine similarity** (если override не сработал) — `_embeddings @ q.T` с биасами: `(1 + 0.15·log10(n))` и small-penalty `0.85` для категорий с n<10.
3. **Quality classification** по raw cosine:
   - `maximal` ≥ 0.55 → high confidence
   - `significant` ≥ 0.45 → manager review flag
   - `minimal` ≥ 0.38 → low-confidence, возможно нерелевантно
   - < 0.38 → `none`, fallback в легаси HybridRetriever
4. **Template application** — позиции из `canonical_smeta` конвертируются в `DealItem` с `quantity` × `unit_price` (mean финальной статистики). Per-letter позиции скейлятся на `decomp.letter_count`.
5. **Price band** — `total ± max(aggregated_std, 10%)`.

**SEED_DEALS** — hand-curated canonical (не picker): `{ dataset, canonical_id, seed_ids, enrich_regex, enrich_exclude }`. Использован для «Логотип» (canonical=21208, 4 позиции, 23186₽, источник — реальная сделка). Pattern расширяется на «Фирменный стиль» / «Брендбук» при необходимости.

#### Stability playbook (L1+L2+L3) — паттерн для стабильных категорий

Категории Логотип / Объёмные буквы / Световые вывески выдают стабильные ответы благодаря трёхуровневой защите. При добавлении новой «стабильной» категории применяй все три слоя:

**L1 — Taxonomy split** ([buildSmetaTemplates.mjs `LABUS_CATEGORIES`](RAG_ANALYTICS/buildSmetaTemplates.mjs)):
- Специфичная категория идёт **раньше** общей в массиве (первое совпадение побеждает).
- `allOf` — обязательные regex; `anyOf` — хотя бы один; `notAny` — контаминационные фильтры.
- Пример: `Объёмные буквы` раньше `Световые короба` раньше `Световые вывески`. `Титульные вывески` с `notAny:[/логотип/i]` чтобы не поймать TITLE «Логотип на титульной вывеске».

**L2 — SEED_DEALS canonical** (для категорий, где p60-picker даёт вырожденный шаблон):
- Ручной `canonical_id` + `seed_ids` пул для обогащения `price_stats`.
- Отдельный `enrich_regex` + `enrich_exclude` пул для `keywords_text` — широкий семантический сигнал БЕЗ загрязнения цен.
- Сейчас только «Логотип»; рекомендовано для «Фирменный стиль», «Брендбук», «Логотип под ключ».

**L3 — Runtime keyword override** ([smeta_engine.py `_KEYWORD_OVERRIDES`](RAG_RUNTIME/app/core/smeta_engine.py)):
- Pre-semantic safety net — обгоняет cosine-ранжирование для коротких запросов.
- Поля: `target`, `include` (все должны match), `exclude` (ни один), `strong` (bool).
- **`strong=True`** → `forced_sim` floor = `SIM_MAXIMAL` (0.55), результат попадает в `auto`/`high`.
- **`strong=False`** → floor = `SIM_SIGNIFICANT` (0.45), результат `guided`/`medium`.
- Сейчас включён для: Логотип, Объёмные буквы, Световые вывески (П8.4).

**Типовые exclude-паттерны для L3:**
- `под\s*ключ|пакет\s+услуг|полный\s+комплект|комплекс|набор` — bundle/комплекс-запросы уходят в bundle-флоу, а не в canonical.
- `ремонт|починк|восстановл` — ремонтные запросы идут в manual-флоу.
- Мерч-носители: `ручк|кружк|футболк|шоппер|значок|брелок|бейдж…`
- Печатные носители: `листовк|баннер|этикет|визитк|табличк`.

### 2.4 HybridRetriever ([app/core/retriever.py](RAG_RUNTIME/app/core/retriever.py))

**Qdrant native hybrid search** (ранее был BM25Okapi+кастомный RRF, рефакторен на нативный Qdrant RRF — см. коммит `770b9ab`). Dense через BGE-M3 (1024d, cosine), sparse через BM25 на уровне Qdrant, RRF fusion, payload фильтры (`doc_type`, `direction`, `price_mode`, `confidence_tier`).

Типы документов в коллекции `labus_docs`: `product`, `bundle`, `pricing_policy`, `retrieval_support`, `roadmap`, `offer_profile`.

### 2.5 CrossEncoderReranker ([app/core/reranker.py](RAG_RUNTIME/app/core/reranker.py))

`BAAI/bge-reranker-v2-m3` cross-encoder на топ-20 → топ-8. Heuristic boosts поверх CE-score:

| Сигнал | Множитель |
|---|---|
| `auto` confidence | ×1.3 |
| `guided` confidence | ×1.1 |
| `order_rows ≥ 100` | ×1.15 |
| `doc_type` match | ×1.2 |
| `direction` match | ×1.25 |
| Безнал / финансовый модификатор | ×0.3 |

### 2.6 PricingResolver ([app/core/pricing_resolver.py](RAG_RUNTIME/app/core/pricing_resolver.py))

Детерминированный определитель цены без LLM:

```
confidence_tier  ──►  логика
─────────────────────────────────────────
auto (23)       ──►  price = MODE ± 5%
guided (473)    ──►  [p25, p75] исторических, flag manager_review
manual (12130+) ──►  НЕТ цены, review_reasons + closest analogs
```

Источники приоритетом: orders (исторические) → goods.BASE_PRICE → offers (структура).

### 2.7 ParametricCalculator ([app/core/parametric_calculator.py](RAG_RUNTIME/app/core/parametric_calculator.py))

Формульный расчёт для параметрических товаров (баннер по м², буквы по высоте, кабель по м.п.) на базе `pricing/parametric_rules.csv`. Используется PricingResolver когда матчится параметрический паттерн.

### 2.8 DealLookup ([app/core/deal_lookup.py](RAG_RUNTIME/app/core/deal_lookup.py))

Загружает `RAG_ANALYTICS/output/normalized/*.csv` в память, предоставляет быстрый lookup реальных line items по deal_id. Используется при сегментации references (deals/offers/products), чтобы показывать конкретные позиции из исторической сделки, а не только агрегаты.

### 2.9 PhotoIndex ([app/core/photo_index.py](RAG_RUNTIME/app/core/photo_index.py))

Индекс результатов vision analysis (`data/photo_analysis_raw.jsonl`). Привязывает фото к deal_id и предоставляет визуальное enrichment в ответах API (segmented references).

### 2.10 VisionAnalyzer ([app/core/vision.py](RAG_RUNTIME/app/core/vision.py))

Multimodal pipeline:
- `scripts/fetch_site_images.py` — скрейпит изображения с labus.pro
- `scripts/vision_analysis.py` — Gemini API (через artemox, `gemini-2.0-flash` / `gemini-3`) для структурированного описания
- `scripts/vision_analysis_local.py` — альтернатива через локальный Ollama
- Output: `data/photo_analysis_raw.jsonl` с описаниями, категориями, cross-reference к offers/goods

### 2.11 FeedbackStore ([app/core/feedback_store.py](RAG_RUNTIME/app/core/feedback_store.py))

SQLite (`labus_rag.db`) для пользовательского feedback: rating, comment, привязка к message_id. Основа для будущего RLHF-подхода: анализ паттернов отрицательных фидбеков для калибровки ретривера.

### 2.12 DeepseekGenerator ([app/core/generator.py](RAG_RUNTIME/app/core/generator.py))

LLM через Deepseek API (OpenAI-compat). `generate()` для `/query`, `generate_structured()` для `/query_structured`. Prompts в `configs/prompts.yaml`. **Graceful degradation:** при ошибке API возвращаем structured response без summary/reasoning — retrieval+pricing+smeta работают независимо.

### 2.13 ClarificationParser (в [app/routers/query.py](RAG_RUNTIME/app/routers/query.py))

Распознаёт intent «опиши мне эту смету»/«клиент прислал такие цифры» — эти запросы идут прямо в LLM, минуя SmetaEngine и under-key фильтр. Нужно чтобы не матчились шумовые категории вроде «Вывески 6000₽».

---

## 3. Данные

### 3.1 Источники (read-only, локально)

```
RAG_DATA/
├── goods.csv           — каталог ~13,000+ позиций
├── offers.csv          — шаблонные КП
└── orders.csv          — исторические сделки Bitrix24
```

### 3.2 RAG_ANALYTICS/output/ (Node build pipeline)

```
├── normalized/*.csv            — нормализованные таблицы (DealLookup источник)
├── facts/
│   ├── product_facts.csv       — ~12,626 продуктов с ценовой статистикой
│   ├── bundle_facts.csv        — bundle сигнатуры
│   └── template_match_facts.csv
├── pricing/
│   ├── pricing_recommendations.csv  — auto/guided/manual тиры
│   └── parametric_rules.csv    — формулы для ParametricCalculator
├── kpis/*.csv                  — KPI агрегаты
├── smeta_templates.json        — P7: категории + canonical smeta + price_stats (~1.5MB)
└── smeta_category_embeddings.npy — BGE-M3 embeddings категорий (N × 1024)
```

### 3.3 RAG_RUNTIME/data/processed/ (Python ingest)

| Файл | Источник | Содержимое |
|---|---|---|
| `product_docs.jsonl` | goods.normalized + product_facts + pricing_recs | Карточки товаров с ценами |
| `bundle_docs.jsonl` | bundle_facts + deal_facts + template_match | Наборы с составом |
| `pricing_policy_docs.jsonl` | pricing_recs + qa_report + kpi_summary | Правила auto/guided/manual |
| `retrieval_support_docs.jsonl` | domain knowledge | Синонимы, глоссарий, паттерны |
| `roadmap_docs.jsonl` | roadmaps/*.md | Roadmap-инструкции (FAQ, процедуры) |
| `offer_profiles.jsonl` | offers.csv + template_match | ROI anchors, структура типичных КП |
| `photo_analysis_raw.jsonl` | vision pipeline | Описания изображений |

### 3.4 Runtime persistence

- `RAG_RUNTIME/indexes/` — BM25 state (если используется legacy), артефакты индексации
- `RAG_RUNTIME/models/` — BGE-M3, BGE-Reranker (~3GB)
- Qdrant data volume (`/opt/rag/qdrant_data/` на проде)
- `labus_rag.db` (SQLite) — чаты, сообщения, feedback, пользователи

---

## 4. Response schema (`/query_structured`)

```json
{
  "query": "string",
  "summary": "string",
  "confidence": "auto|guided|manual",
  "direction": "string",
  "detected_direction": "string",
  "estimated_price": {
    "min": 0, "max": 0, "currency": "RUB",
    "mode": "auto|guided|manual|template",
    "basis": "historical_orders|catalog|analog|template:<category>"
  },
  "bundle": [ { "product_name": "...", "quantity": 1, "unit": "шт",
                 "unit_price": 0, "total": 0, "b24_section": "", "notes": "" } ],
  "smeta": {
    "category_name": "Логотип",
    "match_quality": "maximal|significant|minimal|none",
    "match_similarity": 0.61,
    "match_reason": "...",
    "total": 23186.21,
    "price_band": { "min": 20867.59, "max": 25504.83 },
    "confidence": "high|medium|low",
    "flags": [ "..." ],
    "source_deal_id": "21208",
    "deals_in_category": 47
  },
  "flags": [ "manager_confirmation_required", "wide_price_range", ... ],
  "risks": [ "string" ],
  "reasoning": "string",
  "references": [
    { "doc_id": "...", "source_type": "product|bundle|pricing_policy",
      "dataset": "orders|offers|goods", "title": "...", "score": 0.0 }
  ],
  "segmented_references": {
    "deals": [...], "offers": [...], "products": [...]
  },
  "source_distinction": {
    "current_catalog": [...], "historical": [...], "template": [...]
  }
}
```

---

## 5. Деградация

| Ситуация | Поведение |
|---|---|
| Deepseek API недоступен | Response без summary/reasoning, retrieval+pricing+smeta работают |
| Qdrant недоступен | Сервер помечает `/health` unhealthy, SmetaEngine продолжает работу (ему Qdrant не нужен) |
| SmetaEngine артефакты missing | `is_ready=False`, всё идёт в legacy pipeline |
| Vision API недоступен | Photo analysis пропускается при ингесте, PhotoIndex пуст |
| Категория SmetaEngine < 0.38 | Fallback в HybridRetriever |
| `describe`/`provided-smeta` intent | Прямо в LLM, минуя SmetaEngine |
| Junk category / under-key total | Blocked, fallback в LLM |
| Продукт не в каталоге | `manual` + closest analogs |
| Безнал / финансовый модификатор | Flag, penalty ×0.3 в rerank, нет цены |

---

## 6. Производительность (RTX 3090 типичная)

| Компонент | Время (CUDA) | Время (CPU) |
|---|---|---|
| QueryParser + Decomposer | <2 ms | <2 ms |
| SmetaEngine (embed + match + build) | ~15 ms | ~160 ms |
| BGE-M3 query embed | ~10 ms | ~150 ms |
| Qdrant hybrid search | ~8 ms | ~8 ms |
| CrossEncoder rerank (20 pairs) | ~30 ms | ~300 ms |
| PricingResolver | <5 ms | <5 ms |
| Deepseek API | ~800 ms | ~800 ms |
| **Total (SmetaEngine hit, с LLM)** | **~870 ms** | **~1,000 ms** |
| **Total (SmetaEngine hit, без LLM)** | **~60 ms** | **~180 ms** |
| **Total (HybridRetriever fallback, с LLM)** | **~870 ms** | **~1,300 ms** |

SmetaEngine выдаёт детерминированный ответ раньше LLM — в UI можно стримить smeta до того, как Deepseek допишет summary.

---

## 7. Ограничения и будущие работы

- **Качество данных:** ~12,130 позиций manual/non-catalog — для них цена не выдаётся
- **SEED_DEALS покрытие:** пока только «Логотип». Расширение на «Фирменный стиль», «Брендбук» запланировано.
- **Canonical picker bug:** «Титульные вывески» имеет раздутый canonical из 11 позиций (Шоколад, Табличка) — отдельный баг picker'а в buildSmetaTemplates.mjs.
- **Нет real-time обновлений:** изменение данных → пересборка fact tables → ingest → `build_index.py --recreate`.
- **Deepseek зависимость:** generation требует API key; замена на локальную LLM (Qwen2.5-7B или Llama-3.1-8B через vLLM) — опциональный P8+ трек.
- **Vision pipeline:** Gemini через artemox — внешняя зависимость; `vision_analysis_local.py` как Ollama-альтернатива.
- **Eval под SmetaEngine:** текущий `eval/test_cases.json` (46 кейсов) не покрывает SmetaEngine quality thresholds и SEED_DEALS categories — требуется расширение.
