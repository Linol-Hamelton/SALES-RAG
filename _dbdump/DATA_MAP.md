# DATA_MAP — карта данных и связности слоёв RAG

**Статус:** живой документ. Обновлять при добавлении doc_types, изменении payload-схемы,
или правке `scripts/ingest*.py`. Baseline зафиксирован 2026-04-14 в рамках P10.5.

**Назначение:** единая точка правды о том, какие данные у нас есть, в каких слоях они
лежат, через какие id связаны между собой, и где связность теряется при прохождении
в prompt LLM.

---

## 1. Слои данных (сверху вниз)

```
RAG_DATA/              ← сырьё (CSV, MD, DOCX, JSON)
   ↓   build pipeline (RAG_ANALYTICS/scripts)
RAG_ANALYTICS/output/  ← нормализованные + агрегированные факты/KPI/pricing
   ↓   ingestion (RAG_RUNTIME/scripts/ingest*.py)
RAG_RUNTIME/data/      ← jsonl-документы по doc_type + bridge_docs.jsonl
   ↓   build_index.py (embed BGE-M3 + sparse BM25)
Qdrant collection      ← ~75 000 points, hybrid dense+sparse
   ↓   retriever.py (intent-filter → hybrid search → rerank)
generator._format_context_block → prompt → LLM
```

Каждая стрелка — потенциальная точка потери связности. В этом документе
перечислены все известные разрывы (секция 5).

---

## 2. Источники сырых данных (RAG_DATA/)

| Источник | Объём | Ключи | Назначение |
|----------|-------|-------|-----------|
| [RAG_DATA/goods.csv](../RAG_DATA/goods.csv) | 9 995 строк | **PRODUCT_ID** (GOOD_ID пустой!), PRODUCT_NAME, BASE_PRICE, SECTION_NAME, DIRECTION | Каталог услуг/товаров. Цена в `BASE_PRICE`, не `PRICE` |
| [RAG_DATA/offers.csv](../RAG_DATA/offers.csv) | 11 778 строк | **ID (offer_id)**, **GOOD_ID** (line-item), PRODUCT_ID (каталог), **OPPORTUNITY (deal_id)**, PRICE, QUANTITY | Коммерческие предложения клиентам |
| [RAG_DATA/orders.csv](../RAG_DATA/orders.csv) | 50 408 строк | **ID (order_id)**, **GOOD_ID** (line-item), PRODUCT_ID (каталог), **COMPANY_ID**, PRICE | Фактические заказы |

> **Критично — ID-семантика:** `PRODUCT_ID` — ключ каталога (9 995 значений в goods.csv).
> `GOOD_ID` — **транзакционный** line-item id, существует только в offers/orders (в goods.csv всегда пустой).
> В структурированном ответе API поле `deal_items[].good_id` по смыслу — каталожный
> PRODUCT_ID (именование поля — историческое). LLM следует ссылаться на PRODUCT_ID из
> `product`-документов и из `product_catalog_refs` в bridge-пакетах.
| [RAG_DATA/ROADMAPS/](../RAG_DATA/ROADMAPS/) | 65 `.md` | service_name (stem файла) | Регламенты процессов. Только **13** содержат паттерн `Пакет «…» (N – M руб.)`; остальные — чек-листы без фиксированных пакетов |

**Важно:** bridge_docs.jsonl собирается вручную из BRIDGE_DEFS в
[scripts/ingest_bridges.py](../RAG_RUNTIME/scripts/ingest_bridges.py), а не из ROADMAPS.
Никакой авто-парсинг ROADMAPS пакетов не даст новых bridge'ей без ручной разметки —
13 уже размеченных roadmap'ов совпадают с 4 существующими + candidates без packages.

---

## 3. Агрегированные факты (RAG_ANALYTICS/output/)

| Артефакт | Объём | Ключи | Роль |
|----------|-------|-------|------|
| [normalized/{goods,offers,orders}.csv](../RAG_ANALYTICS/output/normalized/) | совпадает с RAG_DATA + 15 колонок (CLIENT_KEY, NORMALIZED_DIRECTION, PRICE_NUM, DEAL_DURATION_DAYS) | — | Обогащение сырья контекстом клиента и направления |
| [facts/product_facts.csv](../RAG_ANALYTICS/output/facts/product_facts.csv) | 12 567 | PRODUCT_ID, PRODUCT_KEY | Статистика цен P25/P50/P75 + коэффициент наценки |
| [facts/bundle_facts.csv](../RAG_ANALYTICS/output/facts/bundle_facts.csv) | 12 535 | **BUNDLE_KEY** (составной `PRODUCT_ID\|…`), DEAL_COUNT | Типовые наборы услуг из реальных сделок |
| [facts/deal_facts.csv](../RAG_ANALYTICS/output/facts/deal_facts.csv) | 18 150 | DEAL_ID, CLIENT_KEY, BUNDLE_KEY | Профиль сделки: состав, длительность, направления |
| [facts/deal_profiles.csv](../RAG_ANALYTICS/output/facts/deal_profiles.csv) | 9 494 | DEAL_ID | Углублённый профиль (orders + offers aggregated) |
| [facts/offer_profiles.csv](../RAG_ANALYTICS/output/facts/offer_profiles.csv) | 2 135 | DEAL_ID | Матрица offer↔product↔direction (только offers-only deals) |
| [facts/client_facts.csv](../RAG_ANALYTICS/output/facts/client_facts.csv) | 3 149 | CLIENT_KEY | Профиль клиента: повторяемость, доминирующее направление |
| [facts/service_composition.csv](../RAG_ANALYTICS/output/facts/service_composition.csv) | 12 | DIRECTION | Типовые составы (core/optional products) по направлениям |
| [facts/template_match_facts.csv](../RAG_ANALYTICS/output/facts/template_match_facts.csv) | 4 416 | TEMPLATE_BUNDLE_KEY | Согласованность сделок с шаблонами |
| [facts/timeline_facts.csv](../RAG_ANALYTICS/output/facts/timeline_facts.csv) | 216 | GROUP_KEY | Временные профили сделок |
| kpis/* | — | PRODUCT_ID, CLIENT_KEY, DIRECTION | Показатели (catalog utilization, client concentration, monthly trends) |
| [pricing/pricing_recommendations.csv](../RAG_ANALYTICS/output/pricing/pricing_recommendations.csv) | 12 567 | PRODUCT_ID, PRODUCT_KEY | CONFIDENCE_TIER (auto/guided/manual) + рекомендованная цена |
| [smeta_templates.json](../RAG_ANALYTICS/output/smeta_templates.json) | 124 категории | **category_id** (`labus:…`), canonical_smeta.positions[].good_id, all_goods_in_category | Каноническая смета по категории + список GOOD_ID в категории |

---

## 4. Индексированные документы (RAG_RUNTIME/data/*.jsonl)

16 doc_types, ~75 000 документов в Qdrant. Для каждого doc_type указаны:
- **Payload cross-ref**: какие id-поля в payload могут служить ссылками на другие doc_types
- **В prompt попадает?** — селекция в [generator.py `_format_context_block`](../RAG_RUNTIME/app/core/generator.py)

| doc_type | Источник | #docs | Payload cross-ref id | В prompt? |
|----------|----------|-------|----------------------|-----------|
| `product` | product_facts + pricing_recommendations | 12 568 | **GOOD_ID, product_id, product_key, nearest_analogs[]** | Имя+цена ✅; **GOOD_ID ❌ (G1)** |
| `bundle` | bundle_facts + template_match_facts | 12 536 | **bundle_key, sample_deal_ids[]** | Состав+медиана ✅; **id ❌** |
| `deal_profile` | deal_profiles + deals.json + photo_analysis | 9 495 | **deal_id** | Header+состав ✅; **deal_id как ref ❌ (G5)** |
| `offer_profile` | offer_profiles + deals.json | 2 136 | **deal_id, offer_id** (из title/text) | Состав ✅; **offer_id ❌** |
| `offer_composition` | offers.normalized (ключевые) | 22 | **offer_id, products[].product_id** | Имена товаров ✅; **product_id в items ❌ (G4)** |
| `service_pricing_bridge` | BRIDGE_DEFS (manual) | 4 | **service, packages[].offer_ids, packages[].products[]** | Пакеты+цены ✅; **offer_ids ❌ (G2)**; **good_ids в packages не зарезолвлены (G3)** |
| `photo_analysis` | photo_analysis_raw.jsonl | 11 105 | deal_id, good_ids (в searchable_text, не в payload) | Текст ✅; **good_ids ❌ (G6)** |
| `roadmap` | ROADMAPS/*.md chunks | 706 | — | Чанки текста ✅ |
| `knowledge` | FAQ + MD + DOCX | 183 | — | Контент ✅ |
| `pricing_policy` | pricing_summary + qa_report | 11 | — | Правило ✅ |
| `service_composition` | service_composition.csv | 12 | DIRECTION | Состав ✅ |
| `timeline` | timeline_facts.csv | 217 | GROUP_KEY | Сроки ✅ |
| `retrieval_support` | domain_knowledge | 11 | — | Только фильтрация (не в context) |
| `roi_anchor` | manual | 7 | — | Смешаны в knowledge |
| `faq` | labus_faqs_final.json | **0** | — | Пусто — FAQ исходник удалён |

---

## 5. Матрица связности (из источника → doc_type → ключи)

```
goods.csv (GOOD_ID, PRODUCT_ID)
    ├→ product_facts.csv (PRODUCT_ID) ─────┐
    ├→ bundle_facts.csv (BUNDLE_KEY)       │
    ├→ offers.csv (GOOD_ID, OPPORTUNITY)   │     ingest.py
    └→ orders.csv (GOOD_ID, COMPANY_ID)    │        ↓
                                           ├→ doc_type=product
                                           │    payload: {good_id, product_id, ...}
                                           │
smeta_templates.json                       │
    canonical_smeta.positions[].good_id ←──┘
    all_goods_in_category[] ──────────────→ doc_type=smeta_template (implicit)

offers.csv (ID, GOOD_ID, OPPORTUNITY)
    ├→ offer_profiles.csv (DEAL_ID)
    ├→ deal_profiles.csv (DEAL_ID)
    └→ normalized/offers.csv
            ↓  ingest.py (ключевые)
       doc_type=offer_composition (22 docs)
          payload: {offer_id, products:[{product_id, name}]}

BRIDGE_DEFS (scripts/ingest_bridges.py)
    packages[].offer_ids ────────→ (ref на offers.csv.ID / offer_composition.offer_id) ⚠ НЕ резолвится runtime
    packages[].products[] (name)  ────────→ (должно ref на goods.PRODUCT_NAME → GOOD_ID) ⚠ НЕ резолвится ingest
            ↓
       doc_type=service_pricing_bridge (4 docs)
```

**Ключевые «мостовые» поля** (должны работать для link-following):
- `offer_id` ↔ `offers.csv.ID` ↔ `offer_composition.offer_id` ↔ `offer_profile.offer_id`
- `deal_id` ↔ `deal_profile.deal_id` ↔ `offer_profile.deal_id` ↔ `photo_analysis.deal_id` ↔ `offers.csv.OPPORTUNITY`
- `good_id` ↔ `goods.csv.GOOD_ID` ↔ `product.good_id` ↔ `smeta_templates.positions[].good_id`
- `product_id` ↔ `goods.csv.PRODUCT_ID` ↔ `product.product_id` ↔ `offer_composition.products[].product_id`
- `bundle_key` ↔ `bundle_facts.csv` ↔ `bundle.bundle_key` ↔ `deal_facts.csv.BUNDLE_KEY`

---

## 6. Разрывы связности (G1–G7)

Разрыв = информация есть в источнике, но теряется по пути к LLM.

| # | Разрыв | Где теряется | Последствие | Фикс |
|---|--------|-------------|-------------|------|
| **G1** | `product.PRODUCT_ID` не прозрачно помечен в prompt (было `Арт.{id}` — неоднозначно) | [generator.py `_format_context_block` ветка `doc_type=="product"`](../RAG_RUNTIME/app/core/generator.py) | LLM путается, какой id класть в `deal_items[].good_id` структурированного ответа | P10.5-II.1 (переименовано в `PRODUCT_ID=`) |
| **G2** | `bridge.offer_ids` не в prompt **и** не триггерит retrieval | [generator.py ветка `doc_type=="service_pricing_bridge"`](../RAG_RUNTIME/app/core/generator.py); [retriever.py `retrieve_by_intent`](../RAG_RUNTIME/app/core/retriever.py) | LLM видит пакет «27 500–48 500 ₽», но не видит реальные КП #21208/21210/21214 с составом | P10.5-II.2 + P10.5-IV |
| **G3** | `bridge.packages[].products` — строки без каталожного ID | [scripts/ingest_bridges.py `BRIDGE_DEFS`](../RAG_RUNTIME/scripts/ingest_bridges.py) | В goods.csv `Отрисовка ручная…` имеет PRODUCT_ID и BASE_PRICE, но bridge хранит только имя. LLM не может заполнить `deal_items[].good_id` | P10.5-III (резолв в `product_catalog_refs`) |
| **G4** | `offer_composition.products[].product_id` не форматируется | [generator.py ветка `doc_type=="offer_composition"`](../RAG_RUNTIME/app/core/generator.py) | Имена есть, артикулы теряются | P10.5-II.3 |
| **G5** | `deal_profile.deal_id` ↔ `bundle.sample_deal_ids` — только в тексте | Нет link-following | LLM видит кейс и шаблон, не понимает что они о той же сделке | P10.5-II.4 |
| **G6** | `photo_analysis.good_ids` не нормализованы в payload | [scripts/ingest.py (photo_analysis)](../RAG_RUNTIME/scripts/ingest.py) | Vision извлекает артикулы, но они только в `searchable_text` | Out-of-scope (Phase C) |
| **G7** | `smeta_templates.all_goods_in_category → product docs` — слабая связь | Runtime (SmetaEngine bridge-path) | Category «Логотип» в smeta_templates имеет 20+ GOOD_ID, но при переключении в bridge-block эти GOOD_ID не подмешиваются | Out-of-scope (Phase C) |

---

## 7. Покрытие bridges (baseline 2026-04-14)

Из [category_coverage.summary.txt](../RAG_RUNTIME/data/category_coverage.summary.txt):

- Всего категорий: **124**
- Всего сделок: **5 406**
- Revenue proxy: **150 204 300 ₽**
- Уже покрыто bridges: **4 020 914 ₽ (2.7%)**
- Top-25 non-bridged = **88.7%** revenue потенциала

Текущие bridges (4): `Дизайн логотипа`, `Дизайн брендбука`, `Дизайн фирменного стиля`, `Дизайн упаковки`.

**Top-5 некрытых кандидатов** (но все без packages в roadmap — требуют manual BRIDGE_DEFS):
1. Футболки — 25 143 413 ₽ (69 сделок)
2. Кружки — 13 472 447 ₽ (216 сделок)
3. Бейсболки — 11 939 634 ₽ (70 сделок)
4. Этикетки и стикеры — 6 920 716 ₽ (516 сделок)
5. Брошюры — 6 618 010 ₽ (250 сделок)

**Решение:** расширение coverage (Phase B) отложено до завершения P10.5 I–V.
Новые bridges унаследуют те же разрывы связности — нет смысла добавлять их до фикса G1–G5.

---

## 8. Регенерация инвентаризации

### 8.1 Пересчитать category coverage

```bash
cd d:/SALES_RAG
python RAG_RUNTIME/scripts/rank_categories_by_volume.py
# → RAG_RUNTIME/data/category_coverage.csv
# → RAG_RUNTIME/data/category_coverage.summary.txt
```

### 8.2 Посмотреть реальный payload-schema в Qdrant (docker exec)

```bash
docker exec labus_api python - <<'PY'
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.config import settings

c = QdrantClient(url=settings.qdrant_url)
for dt in ["product", "service_pricing_bridge", "offer_composition",
           "offer_profile", "deal_profile", "bundle", "photo_analysis"]:
    res, _ = c.scroll(
        settings.qdrant_collection,
        scroll_filter=Filter(must=[FieldCondition(
            key="doc_type", match=MatchValue(value=dt))]),
        limit=1, with_payload=True, with_vectors=False,
    )
    if res:
        keys = sorted(res[0].payload.keys())
        print(f"{dt}: {keys}")
    else:
        print(f"{dt}: <no docs>")
PY
```

Этот выхлоп показывает **актуальные** поля payload'а (после всех ingest'ов).
Сравнивать с таблицей в §4 при изменениях.

### 8.3 Пересчитать количество документов по doc_type

```bash
docker exec labus_api python - <<'PY'
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.config import settings

c = QdrantClient(url=settings.qdrant_url)
total = c.count(settings.qdrant_collection).count
print(f"Total: {total}")
for dt in ["product", "bundle", "deal_profile", "offer_profile",
           "offer_composition", "service_pricing_bridge", "photo_analysis",
           "roadmap", "knowledge", "pricing_policy", "service_composition",
           "timeline", "retrieval_support", "roi_anchor", "faq"]:
    n = c.count(settings.qdrant_collection, count_filter=Filter(must=[
        FieldCondition(key="doc_type", match=MatchValue(value=dt))])).count
    print(f"  {dt}: {n}")
PY
```

---

## 9. История ревизий

| Дата | Что изменилось | Commit/Phase |
|------|----------------|--------------|
| 2026-04-14 | Первичная фиксация после P10-A (A1–A7 deployed) | P10.5-I |

При каждом изменении ingest/payload — допиши строку сюда и обнови §4.
