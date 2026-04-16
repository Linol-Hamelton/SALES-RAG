# Data gaps — P12 audit (2026-04-15)

Read-only roadmap для P13+. Зафиксировано в рамках P12 Phase 4 после разбора
54 feedback-пар (2026-04-15 dump) и анализа 4 направлений улучшений
(catalog grounding, parametric pricing, manager-script routing, Мерч bridges).

Секции:
- **(a) Accessible-now** — уже закрыто в P12 Phase 3, сохранено как roadmap-чекпойнт.
- **(b) Accessible-deferred** — решается на текущих источниках, отложено на P13+.
- **(c) Requires external access** — требует доступа, не входящего в scope P12.

---

## (a) Accessible-now — закрыто в P12

| Gap | How it closed | Ref |
|---|---|---|
| Нет каталожных ссылок в reasoning | `labus_url` в product-payload + prompt-инструкция цитировать `labus.pro/product/XXXX` | P12.3.A1–A3 |
| Тираж/формат/материал не извлекаются | 4 новых экстрактора в `query_decomposer.py` + поля в `QueryDecomposition` + scaling в `SmetaEngine._scale_quantity` | P12.3.B1–B2 |
| Нет сигнала для тиражных запросов без количества | Флаг `need_parametrization` + вопрос в summary | P12.3.B3 |
| Manager-script запросы уходят в ценовую оценку | `is_macro` + `macro_type` в knowledge-payload, `_force_inject_macro` + intent hint | P12.3.C1–C3 |
| Нет bridge'ей для топ-3 мерча | +3 BRIDGE_DEFS (Футболки / Кружки / Бейсболки) с bucketing по тиражу | P12.3.D1 |
| Taxonomy-пробелы в feedback-анализе | +5 CRITIQUE_TYPES + cluster-report | P12.2 |

---

## (b) Accessible-deferred — можно сделать на текущих источниках

| Gap | Current state | Impact | Effort | Plan |
|---|---|---|---|---|
| `photo_analysis.good_ids` не structured | good_ids сидят в `searchable_text`, не в payload-поле → нельзя фильтровать | Средний: vision-подбор товаров слабее, чем мог бы быть | S (regex extract + re-ingest) | P13.A |
| Deal timeline milestones | Таймштампы есть в orders/deals CSV, но не индексированы по этапам | Roadmap-вопросы про сроки отвечают шаблонно, без статистики | M (новый ingest из orders.csv) | P13.B |
| SEED_DEALS для Мерч (Футб/Круж/Бейсб) | Bridges есть (P12.3.D1), SEED_DEALS нет — confidence не авто | Низкий: bridge-путь уже даёт корректный ответ | M (нужны clean offer_ids из dataset) | P13.C |
| Bridge coverage остальной Печатной | 2.7% revenue до P12.D1 → ~10% после | Высокий для top-2 Печатной (Этикетки, Брошюры) | M (SEED_DEALS + BRIDGE_DEF) | P13.D |
| Response macros не покрывают отраслевые брифы | Есть 3 brief-anchor'а (мерч/print/signage) | Средний: отраслевые скрипты (ресторан, медцентр) не маршрутизируются | continuous (пополнение через user input) | P13.E |
| FAQ ↔ product cross-link | FAQ-документы не знают про PRODUCT_ID | Низкий: faq-ответы без каталожной ссылки | S (текстовый post-link через goods.csv) | P13.F |
| Intent classifier на macro/script | Сейчас regex (_MACRO_INTENT_RE) | Низкий: regex ceiling ~80–85% precision | M (ML-классификатор после сбора baseline) | P13.G |

---

## (c) Requires external access — вне scope P12

| Gap | Why blocked | Requires |
|---|---|---|
| CRM attachments (PDF брифы, ТЗ от клиентов) | Нет доступа к Bitrix24 API в текущем pipeline | Bitrix API интеграция |
| Vision re-run для полного покрытия good_ids | Subset проанализирован, полный прогон не запускался | Vision API (Gemini) + бюджет на 20K+ фото |
| Внешние прайсы / конкурентный бенчмарк | Нет access | Vendor PDFs, scraping агрегаторов |
| Live feedback stream | Сейчас периодический dump БД с prod | API-эндпоинт для feedback push / webhook |

---

## Методика audit'а

1. `extract_feedback_pairs.py --since 2026-04-01` → 54 пары (22 positive / 32 negative).
2. `analyze.py --cluster-report` → таксономия из 14 critique_types, top кластеры:
   - wrong_catalog (закрывается P12.3.A)
   - parametric_mismatch (P12.3.B)
   - manager_script_intent (P12.3.C)
   - missing_article_link (P12.3.A1)
3. Bridge coverage проверен по fact-rollup: Мерч (3 направления) даёт ~50M₽ выручки —
   максимальный ROI для новых bridges (P12.3.D).
4. Этот артефакт read-only: не триггерит изменений кода в P12, только фиксирует
   приоритеты для P13+.

---

## Out of scope (подтверждено с пользователем 2026-04-15)

- Bitrix API / CRM attachments — вне scope P12.
- Vision re-run для полного good_ids покрытия — вне scope P12.
- Новые внешние источники данных — вне scope P12; работаем только с `RAG_DATA/`
  + канонические макросы из `Брифы.md`.
