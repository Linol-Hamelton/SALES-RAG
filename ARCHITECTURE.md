# Labus Sales RAG — Architecture

## Обзор системы

Локальная RAG-система без облачных зависимостей (кроме Deepseek API для генерации).

```
Client Request
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                       FastAPI (port 8000)                    │
│  /health  /query  /query_structured  /rebuild_index          │
└────────────────────────────┬────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   QueryParser         PricingResolver    DeepseekGenerator
   (direction,         (auto/guided/      (summarization,
    intent, budget)     manual logic)      LLM via API)
          │
          ▼
   HybridRetriever
   ┌──────┴──────┐
   │             │
Qdrant BM25    BM25Okapi
(BGE-M3       (lexical,
 dense)        rank-bm25)
   └──────┬──────┘
          │ RRF fusion (α=0.7 dense)
          ▼
   CrossEncoderReranker
   (bge-reranker-v2-m3
    + heuristic boosts)
          │
          ▼
   Top-N results
   + PricingResolver
   + StructuredResponse
```

---

## Компоненты

### 1. QueryParser (`app/core/query_parser.py`)

Разбирает входящий текст на структурированные сигналы:

| Поле | Описание | Пример |
|------|----------|--------|
| `direction` | Бизнес-направление | `"Цех"`, `"Сольвент"`, `"Дизайн"` |
| `intent` | Тип запроса | `product`, `bundle`, `policy`, `general` |
| `budget` | Упомянутый бюджет | `50000.0` |
| `quantity` | Количество | `3.0` |
| `is_financial_modifier` | Безнал / скидка | `True/False` |

Работает на regex + словарях направлений. Без LLM, мгновенно.

---

### 2. HybridRetriever (`app/core/retriever.py`)

**Dense:** BGE-M3 (`BAAI/bge-m3`) → Qdrant collection `labus_docs`
- Multilingual (RU/EN), 1024-dim, cosine similarity
- Payload filters: `doc_type`, `direction`, `price_mode`, `confidence_tier`

**Lexical:** BM25Okapi (`rank-bm25`)
- Токенизация через `app/utils/text.tokenize_ru`
- Сохраняется в `indexes/bm25_state.pkl`

**Fusion:** Reciprocal Rank Fusion
```python
score = α * rank_dense + (1-α) * rank_bm25    # α=0.7
score *= (1 + direction_boost) if direction matches
```

**Типы документов** в коллекции:
- `product` — товары/услуги из каталога
- `bundle` — шаблонные и исторические наборы
- `pricing_policy` — логика auto/guided/manual
- `retrieval_support` — синонимы, глоссарий

---

### 3. CrossEncoderReranker (`app/core/reranker.py`)

**Модель:** `BAAI/bge-reranker-v2-m3`
- Cross-encoder: оценивает пару (query, doc) совместно
- Значительно точнее bi-encoder при top-20 → top-8

**Heuristic boosts** поверх CE-score:
| Сигнал | Множитель |
|--------|-----------|
| `auto` confidence | ×1.3 |
| `guided` confidence | ×1.1 |
| order_rows ≥ 100 | ×1.15 |
| doc_type match | ×1.2 |
| direction match | ×1.25 |
| Безнал / финансовый модификатор | ×0.3 |

---

### 4. PricingResolver (`app/core/pricing_resolver.py`)

Определяет цену независимо от LLM на основе структурированных данных:

```
confidence_tier из product_facts / pricing_recommendations
    │
    ├── "auto" (23 продукта)
    │   └── price = MODE ± 5%
    │       confidence = "auto"
    │
    ├── "guided" (473 продукта)
    │   └── price = [p25, p75] исторических цен
    │       confidence = "guided", requires manager confirmation
    │
    └── "manual" (12,130+ позиций)
        └── НЕ выдаёт цену
            confidence = "manual"
            review_reasons + closest analogs
```

Приоритет источников цен:
1. `orders` — исторические реальные сделки (главный источник)
2. `goods.BASE_PRICE` — текущий reference anchor
3. `offers` — шаблонная структура и quantity ladder

---

### 5. DeepseekGenerator (`app/core/generator.py`)

LLM-генерация через Deepseek API (OpenAI-совместимый):
- `generate()` — human-readable текст для `/query`
- `generate_structured()` — JSON для `/query_structured`

Prompts хранятся в `configs/prompts.yaml`.

**Graceful degradation:** если API недоступен, возвращается структурированный ответ без LLM-части (retrieval + pricing работают независимо).

---

## Слои данных

### Источники (read-only)

```
RAG_DATA/
├── goods.csv           — каталог товаров/услуг
├── offers.csv          — шаблонные комплекты
└── orders.csv          — исторические сделки

RAG_ANALYTICS/output/
├── normalized/*.csv    — нормализованные данные
├── facts/*.csv         — product_facts, bundle_facts, template_match_facts
├── pricing/*.csv       — pricing_recommendations
└── kpis/*.csv          — KPI агрегаты
```

### Производные (RAG_RUNTIME/data/processed/)

| Файл | Источник | Содержимое |
|------|----------|-----------|
| `product_docs.jsonl` | goods.normalized.csv + product_facts + pricing_recs | Карточки товаров с ценами |
| `bundle_docs.jsonl` | bundle_facts + deal_facts + template_match_facts | Наборы с составом и ценами |
| `pricing_policy_docs.jsonl` | pricing_recs + qa_report + kpi_summary | Правила авто/гайд/мануал |
| `retrieval_support_docs.jsonl` | Domain knowledge | Синонимы, глоссарий, паттерны |

---

## Схема ответа

```json
{
  "query": "string",
  "summary": "string",           // LLM-generated
  "confidence": "auto|guided|manual",
  "direction": "string",
  "detected_direction": "string",
  "estimated_price": {
    "min": 0,
    "max": 0,
    "currency": "RUB",
    "mode": "auto|guided|manual",
    "basis": "historical_orders|catalog|analog"
  },
  "bundle": [
    {"product_name": "...", "qty": 1, "unit_price": 0, "source": "..."}
  ],
  "flags": ["manager_confirmation_required", "wide_price_range", ...],
  "risks": ["string"],
  "reasoning": "string",         // LLM-generated
  "references": [
    {
      "doc_id": "string",
      "source_type": "product|bundle|pricing_policy",
      "dataset": "orders|offers|goods",
      "title": "string",
      "score": 0.0
    }
  ],
  "source_distinction": {
    "current_catalog": [...],
    "historical": [...],
    "template": [...]
  }
}
```

---

## Ранжирование кандидатов

Порядок сигналов по умолчанию:
1. **Product overlap** — пересечение упомянутых товаров с составом дока
2. **Quantity similarity** — близость количеств
3. **Template similarity** — совпадение с шаблонными офферами
4. **Client context** — если доступен (direction, бюджет)
5. **Direction match** — совпадение направления
6. **Pricing confidence** — предпочтение auto > guided > manual

TITLE не является основным сигналом (исторические заголовки слишком разнородны).

---

## Деградация

| Ситуация | Поведение |
|----------|----------|
| Deepseek API недоступен | Ответ без summary/reasoning, retrieval + pricing работают |
| Qdrant недоступен | BM25-only режим, предупреждение в /health |
| Продукт не найден в каталоге | manual + closest analogs |
| Безнал / финансовый модификатор | Флаг, penalty ×0.3 в rerank, нет цены |
| Слишком мало истории | guided → уменьшенный confidence |
| Несвязный запрос | Политика docs + clarification suggestion |

---

## Производительность (типичная конфигурация)

| Компонент | Время (CUDA) | Время (CPU) |
|-----------|-------------|------------|
| Query parsing | <1 ms | <1 ms |
| BGE-M3 query embed | ~10 ms | ~150 ms |
| Qdrant top-20 | ~5 ms | ~5 ms |
| BM25 | ~2 ms | ~2 ms |
| CrossEncoder rerank (20 pairs) | ~30 ms | ~300 ms |
| Pricing resolve | <5 ms | <5 ms |
| Deepseek API | ~800 ms | ~800 ms |
| **Total (с LLM)** | **~850 ms** | **~1,300 ms** |
| **Total (без LLM)** | **~50 ms** | **~470 ms** |

---

## Ограничения

- **Качество данных:** ~12,130 позиций manual/non-catalog — для них цена не выдаётся
- **Fine-tuning не выполнен:** RAG работает без дообучения LLM; при необходимости следующий шаг — LoRA на Deepseek или Qwen
- **Нет real-time обновлений:** при изменении данных требуется пересборка индекса
- **Deepseek зависимость:** generation требует API key; в будущем — замена на локальную LLM (Qwen2.5-7B или Llama-3.1-8B через vLLM)
