# Labus Sales RAG — Runtime

Локальная production-ready RAG-система для подбора товаров/услуг, оценки стоимости сделок и формирования коммерческих предложений.

**Стек:** Python 3.11 · FastAPI · FAISS/Qdrant · BGE-M3 (embeddings) · BGE-Reranker-v2-m3 · Deepseek API (generation)

---

## Быстрый старт

```bash
# 1. Перейти в RAG_RUNTIME
cd RAG_RUNTIME

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # Linux/Mac

# 3. Установить зависимости
pip install -r requirements.txt
# GPU PyTorch (если CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Создать configs/.env
cp configs/.env configs/.env
# Отредактировать: вставить DEEPSEEK_API_KEY, проверить пути

# 5. Проверить окружение
python scripts/check_gpu.py

# 6. Загрузить модели
huggingface-cli download BAAI/bge-m3 --local-dir models/embeddings/BAAI_bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/reranker/BAAI_bge-reranker-v2-m3

# 7. Запустить Qdrant (в отдельном терминале)
docker compose up qdrant -d

# 8. Инgestировать данные и построить индекс
python scripts/data_report.py         # проверка данных
python scripts/ingest.py              # создать knowledge docs
python scripts/build_index.py         # векторный индекс + BM25

# 9. Запустить сервер
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# 10. Тестовый запрос
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "световая вывеска кофейня под ключ"}'
```

---

## Структура папок

```
RAG_RUNTIME/
├── app/
│   ├── core/               # retriever, reranker, generator, pricing_resolver, query_parser
│   ├── routers/            # health, query, admin, eval
│   ├── schemas/            # Pydantic models
│   ├── utils/              # logging, text utils
│   ├── config.py           # Settings (Pydantic BaseSettings)
│   └── main.py             # FastAPI factory + lifespan
├── configs/
│   ├── .env                # секреты и пути (не в git)
│   ├── settings.yaml       # гиперпараметры retrieval/rerank/gen
│   └── prompts.yaml        # LLM prompt templates
├── data/
│   ├── processed/          # JSONL knowledge docs (product, bundle, policy, support)
│   └── ...
├── eval/
│   ├── test_cases.json     # 46 тест-кейсов по 6 категориям
│   └── results.json        # результаты последнего прогона eval
├── indexes/
│   ├── faiss/              # FAISS index (если включён fallback)
│   └── bm25_state.pkl      # BM25 index
├── logs/                   # структурированные JSON-логи
├── models/
│   ├── embeddings/         # BGE-M3
│   └── reranker/           # BGE-Reranker-v2-m3
├── reports/
│   ├── data_readiness_report.md
│   ├── data_readiness_report.json
│   ├── model_readiness_report.md
│   └── model_readiness_report.json
├── scripts/
│   ├── check_gpu.py        # проверка окружения
│   ├── data_report.py      # аудит данных
│   ├── ingest.py           # ingest pipeline
│   ├── build_index.py      # embedding + vector index
│   ├── query_cli.py        # CLI query
│   └── eval.py             # eval runner
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
├── RUNBOOK.md
└── ARCHITECTURE.md
```

---

## Конфигурация

Все параметры через `configs/.env` (переменные окружения) и `configs/settings.yaml` (гиперпараметры).

### Ключевые переменные в .env

| Переменная | Описание | По умолчанию |
|-----------|----------|-------------|
| `DEEPSEEK_API_KEY` | API ключ Deepseek | *обязательно* |
| `DEEPSEEK_BASE_URL` | URL Deepseek API | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | Модель | `deepseek-chat` |
| `QDRANT_URL` | URL Qdrant | `http://localhost:6333` |
| `EMBEDDING_MODEL_PATH` | Путь к BGE-M3 | `models/embeddings/BAAI_bge-m3` |
| `RERANKER_MODEL_PATH` | Путь к reranker | `models/reranker/BAAI_bge-reranker-v2-m3` |
| `RETRIEVAL_TOP_K` | Кандидатов из retrieval | `20` |
| `RERANK_TOP_N` | Финальных после rerank | `8` |

### Режим без GPU

Если GPU недоступен, система работает на CPU — только медленнее при embed/rerank. Генерация через Deepseek API не требует GPU.

---

## API Endpoints

| Endpoint | Метод | Описание |
|----------|-------|---------|
| `/health` | GET | Статус компонентов системы |
| `/query` | POST | Human-readable ответ |
| `/query_structured` | POST | JSON ответ с pricing, bundle, references |
| `/rebuild_index` | POST | Перестройка индекса в фоне |
| `/eval_summary` | GET | Последние результаты eval |

### Пример запроса `/query_structured`

```json
POST /query_structured
{
  "query": "световая вывеска аптека состав и стоимость с монтажом под ключ",
  "max_results": 5
}
```

### Пример ответа

```json
{
  "query": "световая вывеска аптека...",
  "summary": "Найдены 3 подходящих шаблона...",
  "confidence": "guided",
  "direction": "Цех",
  "estimated_price": {
    "min": 45000,
    "max": 85000,
    "currency": "RUB",
    "mode": "guided",
    "basis": "historical_orders"
  },
  "bundle": [
    {"product_name": "Световые объёмные буквы", "qty": 1, "unit_price": 35000},
    {"product_name": "Монтаж наружной конструкции", "qty": 1, "unit_price": 8000}
  ],
  "flags": ["manager_confirmation_required"],
  "risks": ["wide_price_range"],
  "references": [...],
  "reasoning": "Guided pricing на основе 47 похожих заказов..."
}
```

---

## CLI

```bash
# Запрос без сервера (прямой вызов компонентов)
python scripts/query_cli.py --query "баннер 3х6 метров цена"
python scripts/query_cli.py --query "безнал 10%" --mode structured
python scripts/query_cli.py --query "световая вывеска" --show-context

# Eval
python scripts/eval.py --verbose
python scripts/eval.py --categories auto_priced guided_priced

# Инфра
python scripts/check_gpu.py
python scripts/data_report.py
python scripts/ingest.py --verbose
python scripts/build_index.py --recreate
```

---

## Перенос на новую машину

Переносить:
- `RAG_RUNTIME/models/` — embedding и reranker модели (большие, ~2.5 GB)
- `RAG_RUNTIME/indexes/` — FAISS индекс и BM25 state
- `RAG_RUNTIME/data/processed/` — canonical knowledge docs (JSONL)
- `RAG_RUNTIME/configs/.env` — конфиг с API ключом

**Не переносить** (пересобирается):
- `RAG_RUNTIME/.venv/`
- `RAG_RUNTIME/logs/`
- `RAG_RUNTIME/qdrant_data/` — Qdrant пересобирается через `build_index.py`

После переноса выполнить:
```bash
pip install -r requirements.txt
docker compose up qdrant -d
python scripts/build_index.py --recreate   # переиндексация в Qdrant
uvicorn app.main:app --port 8000 --workers 1
```

---

## Known Issues

- `workers=1` обязательно для uvicorn — BGE-M3 не должен дублироваться в памяти
- Qdrant должен быть запущен до старта сервера
- На Windows кодировка консоли: данные в UTF-8, при отображении кракозябр — проверять `chcp 65001`
- Deepseek API key обязателен для генерации; retrieval и pricing работают без него
