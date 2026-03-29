# PLAN.MD — Labus Sales RAG

Production-ready локальная RAG-система для подбора товаров/услуг, оценки стоимости сделок и формирования коммерческих предложений на базе исторических данных.

---

## Статус реализации

| # | Этап | Статус | Артефакты |
|---|------|--------|-----------|
| 1 | Аудит данных | ✅ Готов | `scripts/data_report.py` → `reports/data_readiness_report.*` |
| 2 | RAG knowledge objects | ✅ Готов | `scripts/ingest.py` → `data/processed/*.jsonl` |
| 3 | Стек под машину | ✅ Выбран | BGE-M3 + Qdrant + Deepseek API |
| 4 | Pipeline | ✅ Готов | `app/core/` |
| 5 | Логика ранжирования | ✅ Готов | `app/core/reranker.py` |
| 6 | Eval suite | ✅ Готов | `eval/test_cases.json`, `scripts/eval.py` |
| 7 | API + CLI | ✅ Готов | `app/main.py`, `scripts/query_cli.py` |
| 8 | Документация | ✅ Готова | `README.md`, `RUNBOOK.md`, `ARCHITECTURE.md` |
| 9 | Финальная проверка | ⬜ Ожидает запуска | — |

---

## Команды запуска (всё с нуля)

```bash
# 1. Окружение
cd RAG_RUNTIME
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Модели
huggingface-cli download BAAI/bge-m3 --local-dir models/embeddings/BAAI_bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/reranker/BAAI_bge-reranker-v2-m3

# 3. Конфиг (отредактировать DEEPSEEK_API_KEY)
# configs/.env уже создан — вставить ключ

# 4. Qdrant
docker compose up qdrant -d

# 5. Данные → Индекс
python scripts/check_gpu.py
python scripts/data_report.py
python scripts/ingest.py
python scripts/build_index.py

# 6. Сервер
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# 7. Eval
python scripts/eval.py --verbose
```

---

## Стек

| Слой | Компонент | Обоснование |
|------|-----------|-------------|
| Embeddings | `BAAI/bge-m3` | Лучшее качество RU/EN, 1024-dim, многоязычный |
| Vector store | Qdrant (local Docker) | Payload фильтры, production-ready, персистентность |
| Lexical | BM25Okapi | Дополняет dense для точных терминов |
| Fusion | RRF (α=0.7) | Стандарт hybrid retrieval |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder, качественнее bi-encoder |
| Generation | Deepseek API | RU/EN business prompts, без локального GPU для LLM |
| API | FastAPI + uvicorn | Async, production-grade, OpenAPI docs |

---

## Данные

### Источники (read-only, не изменялись)

| Файл | Назначение |
|------|-----------|
| `RAG_DATA/goods.csv` | Текущий каталог ~13,000+ позиций |
| `RAG_DATA/offers.csv` | Шаблонные комплекты |
| `RAG_DATA/orders.csv` | Исторические сделки |
| `RAG_ANALYTICS/output/facts/product_facts.csv` | ~12,626 продуктовых фактов |
| `RAG_ANALYTICS/output/facts/bundle_facts.csv` | Bundle сигнатуры |
| `RAG_ANALYTICS/output/pricing/pricing_recommendations.csv` | auto/guided/manual тиры |

### Производные knowledge documents

| Тип | Кол-во | Описание |
|-----|--------|---------|
| `product` | ~12,626 | Карточки товаров с ценами и confidence |
| `bundle` | N | Наборы из шаблонов и реальных заказов |
| `pricing_policy` | ~10+ | Правила auto/guided/manual + QA caveats |
| `retrieval_support` | ~11 | Синонимы, глоссарий направлений, паттерны |

---

## Ключевые числа из данных

| Параметр | Значение |
|----------|---------|
| Auto-pricing продуктов | 23 |
| Guided-pricing продуктов | 473 |
| Manual/non-catalog позиций | ~12,130 |
| Направлений | 8 (Цех, Сольвент, Дизайн, Печатная, РИК, Безнал, …) |
| Тест-кейсов в eval | 46 (6 категорий) |

---

## Архитектурные решения

### 1. Deepseek API вместо локальной LLM
**Причина:** Retrieval + pricing не требуют LLM. LLM нужен только для summarization. Deepseek дешевле и проще, чем содержать локальный 7B на GPU.
**Когда менять:** Когда нужна полная offline работа → Qwen2.5-7B или Llama-3.1-8B через vLLM.

### 2. Qdrant вместо чистого FAISS
**Причина:** Payload фильтры (`direction`, `doc_type`, `price_mode`) критичны для точного retrieval. FAISS не поддерживает фильтрацию без post-processing.

### 3. BGE-M3 vs multilingual-e5-large
**Причина выбора BGE-M3:** Лучше на коротких бизнес-запросах RU, поддерживает colbert-style reranking, активно поддерживается BAAI.

### 4. workers=1 для uvicorn
**Причина:** BGE-M3 (~2.2 GB) загружается в память процесса. При нескольких воркерах каждый загружает модель отдельно.

### 5. Pricing без LLM
**Причина:** LLM склонен галлюцинировать цены. PricingResolver детерминированно следует той же логике что и `classifyProductFact()` из Node.js аналитики.

---

## Следующие шаги (рекомендации)

### Приоритет 1 — Запуск и проверка
- [ ] Загрузить модели (`huggingface-cli download`)
- [ ] Запустить `build_index.py` и проверить Qdrant коллекцию
- [ ] Запустить `eval.py` и получить baseline метрики

### Приоритет 2 — Качество retrieval
- [ ] Добавить больше синонимов в `retrieval_support_docs` на основе реальных запросов менеджеров
- [ ] Настроить `direction_boost` и `RRF_ALPHA` на основе eval результатов
- [ ] Добавить примеры real queries менеджеров в `test_cases.json`

### Приоритет 3 — Локальная LLM (опционально)
- [ ] Тестировать Qwen2.5-7B-Instruct через vLLM как замену Deepseek
- [ ] Сравнить latency и качество RU ответов

### Приоритет 4 — UI
- [ ] Простой Streamlit / Gradio интерфейс для менеджеров
- [ ] История запросов в SQLite

---

## Файловая структура RAG_RUNTIME

```
RAG_RUNTIME/
├── app/                    # FastAPI приложение
│   ├── core/               # retriever, reranker, generator, pricing, query_parser
│   ├── routers/            # health, query, admin, eval
│   ├── schemas/            # Pydantic models
│   ├── utils/              # logging, text
│   ├── config.py
│   └── main.py
├── configs/
│   ├── .env                # секреты (не в git)
│   ├── settings.yaml       # гиперпараметры
│   └── prompts.yaml        # LLM prompts
├── data/processed/         # JSONL knowledge docs
├── eval/                   # test_cases.json, results.json
├── indexes/                # bm25_state.pkl
├── logs/
├── models/                 # BGE-M3, BGE-Reranker
├── reports/                # data_readiness, model_readiness
├── scripts/                # check_gpu, data_report, ingest, build_index, query_cli, eval
├── README.md
├── RUNBOOK.md
├── ARCHITECTURE.md
├── requirements.txt
├── docker-compose.yml
└── Dockerfile
```
