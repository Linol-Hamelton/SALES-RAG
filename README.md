# Labus Sales RAG

Полная RAG-система для подбора товаров/услуг, оценки стоимости сделок и формирования коммерческих предложений компании Лабус (labus.pro).

**Стек:** Python 3.11 · FastAPI · Qdrant · BGE-M3 (embeddings) · BGE-Reranker-v2-m3 · Deepseek API (generation)

---

## Структура проекта

```
SALES_RAG/
├── RAG_DATA/           # Исходные данные: CSV из Bitrix24, markdown-документы, roadmaps
│   ├── goods.csv       # Каталог товаров
│   ├── offers.csv      # Коммерческие предложения
│   ├── orders.csv      # Заказы
│   ├── ROADMAPS/       # 65 дорожных карт услуг
│   └── *.md            # Базы знаний, FAQ, брифы
├── RAG_ANALYTICS/      # Node.js аналитика: факты, бандлы, профили сделок
│   ├── lib/            # Общие утилиты
│   ├── build*.mjs      # Скрипты генерации аналитических CSV
│   └── output/         # Сгенерированные CSV-файлы
├── RAG_RUNTIME/        # Python RAG-приложение (FastAPI)
│   ├── app/            # Основной код: retriever, reranker, generator, routers
│   ├── configs/        # prompts.yaml, settings.yaml, .env
│   ├── scripts/        # ingest.py, build_index.py, vision_analysis.py
│   ├── data/           # Сгенерированные JSONL + embeddings.npy
│   └── models/         # BGE-M3 + BGE-Reranker (локальные)
├── refreshSalesRagData.mjs   # Полный refresh из Bitrix24
├── verifySalesRag.mjs        # Проверка целостности данных
└── package.json
```

---

## Быстрый старт (RAG Runtime)

```bash
cd RAG_RUNTIME

python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt

# GPU PyTorch (если CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Конфигурация
cp configs/.env.example configs/.env
# Вставить DEEPSEEK_API_KEY

# Модели
huggingface-cli download BAAI/bge-m3 --local-dir models/embeddings/BAAI_bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/reranker/BAAI_bge-reranker-v2-m3

# Qdrant + индексация
docker compose up qdrant -d
python scripts/ingest.py
python scripts/ingest_knowledge.py
python scripts/ingest_roadmaps.py
python scripts/build_index.py --recreate

# Запуск
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## Обновление данных из Bitrix24

```bash
# Из корня SALES_RAG:
node refreshSalesRagData.mjs
# или: npm run refresh
```

Этот сценарий последовательно:
1. Скачивает свежие данные из Bitrix24
2. Исправляет направления в orders.csv
3. Запускает полную аналитику
4. Проверяет целостность

---

## API Endpoints

| Endpoint | Метод | Описание |
|----------|-------|---------|
| `/health` | GET | Статус компонентов |
| `/query` | POST | Human-readable ответ |
| `/query_structured` | POST | JSON с pricing, bundle, references |
| `/rebuild_index` | POST | Перестройка индекса |

---

## Деплой (VPS)

```bash
docker compose -f docker-compose.prod.yml up -d
```

Контейнеры: `labus_api` (FastAPI) + `labus_qdrant` (vector DB).

Обновление кода на VPS:
```bash
cd /opt/rag && git pull origin main
docker cp /opt/rag/app labus_api:/app/
docker cp /opt/rag/configs labus_api:/app/
docker restart labus_api
```

---

## Конфигурация

Основные переменные в `configs/.env`:

| Переменная | Описание |
|-----------|----------|
| `DEEPSEEK_API_KEY` | API ключ Deepseek (обязательно) |
| `QDRANT_URL` | URL Qdrant (`http://localhost:6333`) |
| `EMBEDDING_MODEL_PATH` | Путь к BGE-M3 |
| `RERANKER_MODEL_PATH` | Путь к reranker |

Гиперпараметры в `configs/settings.yaml`: top_k, boost multipliers, temperature.

---

## Known Issues

- `workers=1` обязательно — BGE-M3 не должен дублироваться в памяти
- Qdrant должен быть запущен до старта сервера
- Windows: `chcp 65001` для корректного UTF-8
