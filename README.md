# Labus Sales RAG

RAG-система для подбора товаров/услуг, оценки стоимости сделок и формирования
коммерческих предложений компании Лабус (labus.pro).

**Стек:** Python 3.11 · FastAPI · Qdrant · BGE-M3 · BGE-Reranker-v2-m3 ·
Deepseek API · Node.js (analytics) · Docker

---

## 1. Структура репозитория (локально)

```
SALES_RAG/
├── RAG_DATA/                   # CSV из Bitrix24, markdown, roadmaps (НЕ в git, большие)
├── RAG_ANALYTICS/              # Node.js: сборка fact tables, smeta-шаблонов
│   ├── buildSmetaTemplates.mjs
│   └── output/
│       ├── smeta_templates.json              ← ARTIFACT (scp)
│       └── smeta_category_embeddings.npy     ← ARTIFACT (scp)
├── RAG_RUNTIME/                # FastAPI приложение
│   ├── app/                    # код, пушится через git
│   │   ├── core/smeta_engine.py
│   │   ├── routers/query.py
│   │   └── main.py
│   ├── configs/                # prompts.yaml, settings.yaml (.env — ручной, НЕ в git)
│   ├── scripts/                # ingest*.py, build_index.py, embed_smeta_categories.py
│   ├── data/                   # jsonl + embeddings.npy (ARTIFACTS)
│   └── models/                 # BGE-M3, Reranker (ARTIFACTS)
├── _dbdump/                    # снапшоты SQLite чатов (локально, НЕ в git)
└── refreshSalesRagData.mjs
```

---

## 2. Стратегия деплоя (долгосрочная)

### 2.1 Принципы

1. **GitHub = источник правды для кода.** Весь Python/Node/YAML пушится через
   `git push`. Сервер подтягивает через `git pull` в `/opt/rag/`.
2. **Тяжёлые/конфиденциальные артефакты идут напрямую.** Модели, эмбеддинги,
   сгенерированные JSONL, `smeta_templates.json`, CSV Bitrix24 — **не в git**,
   копируются `scp` с локальной машины в `/opt/rag/artifacts/` на сервере.
3. **Локальная машина = build host.** Всё тяжёлое (BGE-M3, Node-аналитика,
   индексация) собирается локально на CUDA. Сервер только раскатывает готовое.
4. **Промежуточный буфер на сервере — `/opt/rag/artifacts/`** (создаётся один
   раз). Оттуда `docker cp` раскладывает файлы в контейнер.
5. **Коммит ВСЕГДА идёт перед деплоем.** Артефакты в коммите упоминаются по
   имени + SHA сборки, чтобы прод был воспроизводим.

### 2.2 Фактическая раскладка на сервере

Хост `/opt/rag/` (исторически плоский, не совпадает с локальной структурой):

```
/opt/rag/
├── .git/                      # clone этого репо (main)
├── app/                       # устаревший mirror, НЕ используется для деплоя кода
├── RAG_RUNTIME/               # частичный mirror, тоже stale
├── RAG_ANALYTICS/output/      # источник scp для smeta_templates.json
├── artifacts/                 # ← СОЗДАТЬ: рабочий буфер для scp (П7.6+)
│   ├── query.py
│   ├── smeta_templates.json
│   └── smeta_category_embeddings.npy
├── configs/                   # .env хранится здесь (НЕ в git)
├── data/, indexes/, models/   # stale копии volume содержимого
└── docker-compose.prod.yml
```

Контейнер `labus_api` (образ `rag-api`):

```
/app/
├── main.py                    # uvicorn entrypoint (PROJECT_ROOT = /app)
├── app/                       # реальный пакет Python (НЕ /app!)
│   ├── main.py
│   ├── core/smeta_engine.py   # читает smeta_templates.json из /RAG_ANALYTICS/output/
│   ├── routers/query.py       # ← ЦЕЛЬ деплоя Python hotfix
│   └── ...
├── configs/.env
├── data/                      # jsonl, embeddings.npy, labus_rag.db (SQLite чатов)
└── models/embeddings|reranker

/RAG_ANALYTICS/output/         # ← ЦЕЛЬ деплоя smeta-артефактов (НЕ под /app!)
├── smeta_templates.json
└── smeta_category_embeddings.npy
```

**Важно:** `smeta_engine.py` ищет шаблоны по пути `/RAG_ANALYTICS/output/`
относительно корня контейнера, а не `/app/RAG_ANALYTICS/`. Это фиксированная
особенность образа `rag-api`.

### 2.3 Таксономия файлов

| Категория | Где | Как в прод |
|-----------|-----|-----------|
| Python код (`app/`, `scripts/`) | git | `git pull` → `docker cp /opt/rag/app/... labus_api:/app/app/` |
| Конфиги (`.yaml`, prompts) | git | `git pull` → `docker cp` в `/app/configs/` |
| `.env`, API ключи | `/opt/rag/configs/.env` | вручную, **НЕ в git** |
| `smeta_templates.json` | локально + `/opt/rag/artifacts/` | `scp` → `docker cp` в `/RAG_ANALYTICS/output/` |
| `smeta_category_embeddings.npy` | локально + `/opt/rag/artifacts/` | `scp` → `docker cp` в `/RAG_ANALYTICS/output/` |
| Модели BGE-M3, Reranker | локально + `/opt/rag/artifacts/models/` | `scp` → `docker cp` в `/app/models/` (редко) |
| Qdrant data | `/opt/rag/qdrant_data/` (docker volume) | не копируется |
| SQLite чатов `labus_rag.db` | docker volume | pull через `.backup` snapshot |
| CSV Bitrix24 | только локально | на сервер не нужны |
| Node.js `.mjs` | git | **не запускается на сервере** (build host only) |

---

## 3. Процедуры

### 3.1 Первичная настройка (один раз)

```bash
# [сервер]
mkdir -p /opt/rag/artifacts
cd /opt/rag && git remote -v   # убедиться, что origin указывает на нужный GitHub
```

### 3.2 Деплой Python hotfix (самый частый путь)

Применяется когда меняются `.py` / `.yaml` / `prompts.yaml`.

```bash
# [локально]
git add RAG_RUNTIME/app/routers/query.py
git commit -m "fix(query): <summary>"
git push origin main

# [локально → сервер] — буферизируем через artifacts, не /tmp
scp -P 22 RAG_RUNTIME/app/routers/query.py \
  root@62.217.178.117:/opt/rag/artifacts/query.py

# [сервер]
cd /opt/rag && git pull origin main
docker cp /opt/rag/artifacts/query.py labus_api:/app/app/routers/query.py
docker restart labus_api
sleep 30
docker logs --tail 50 labus_api | grep -E "Application ready|Smeta engine|ERROR"
```

### 3.3 Деплой smeta-шаблонов (после Node-пересборки)

```bash
# [локально]
node RAG_ANALYTICS/buildSmetaTemplates.mjs
PYTHONIOENCODING=utf-8 python RAG_RUNTIME/scripts/embed_smeta_categories.py

git add RAG_ANALYTICS/buildSmetaTemplates.mjs
git commit -m "feat(smeta): <summary>"
git push origin main

# [локально → сервер]
scp -P 22 RAG_ANALYTICS/output/smeta_templates.json \
  root@62.217.178.117:/opt/rag/artifacts/smeta_templates.json
scp -P 22 RAG_ANALYTICS/output/smeta_category_embeddings.npy \
  root@62.217.178.117:/opt/rag/artifacts/smeta_category_embeddings.npy

# [сервер]
cd /opt/rag && git pull origin main
docker cp /opt/rag/artifacts/smeta_templates.json \
  labus_api:/RAG_ANALYTICS/output/smeta_templates.json
docker cp /opt/rag/artifacts/smeta_category_embeddings.npy \
  labus_api:/RAG_ANALYTICS/output/smeta_category_embeddings.npy
docker restart labus_api
sleep 30
docker logs --tail 50 labus_api | grep -E "Application ready|Smeta engine|SEED|ERROR"
```

### 3.4 Комбинированный деплой (код + артефакты)

Последовательно: сначала commit+push+scp всех файлов, потом один `docker restart`.

```bash
# [локально]
git add ... && git commit -m "..." && git push origin main
scp -P 22 RAG_RUNTIME/app/routers/query.py                   root@62.217.178.117:/opt/rag/artifacts/
scp -P 22 RAG_ANALYTICS/output/smeta_templates.json          root@62.217.178.117:/opt/rag/artifacts/
scp -P 22 RAG_ANALYTICS/output/smeta_category_embeddings.npy root@62.217.178.117:/opt/rag/artifacts/

# [сервер]
cd /opt/rag && git pull origin main
docker cp /opt/rag/artifacts/query.py                        labus_api:/app/app/routers/query.py
docker cp /opt/rag/artifacts/smeta_templates.json            labus_api:/RAG_ANALYTICS/output/smeta_templates.json
docker cp /opt/rag/artifacts/smeta_category_embeddings.npy   labus_api:/RAG_ANALYTICS/output/smeta_category_embeddings.npy
docker restart labus_api && sleep 30
docker logs --tail 50 labus_api | grep -E "Application ready|Smeta engine|SEED|ERROR"
```

### 3.5 Полный reindex (редко)

```bash
# [локально]
python RAG_RUNTIME/scripts/ingest.py
python RAG_RUNTIME/scripts/ingest_knowledge.py
python RAG_RUNTIME/scripts/ingest_roadmaps.py
python RAG_RUNTIME/scripts/build_index.py --recreate   # --batch-size 8 на GPU

scp -P 22 -r RAG_RUNTIME/data/*.jsonl RAG_RUNTIME/data/embeddings.npy \
  root@62.217.178.117:/opt/rag/artifacts/data/

# [сервер]
docker cp /opt/rag/artifacts/data/. labus_api:/app/data/
docker restart labus_api
```

### 3.6 Дамп БД чатов для анализа фидбэка

```bash
# [сервер]
docker exec labus_api sqlite3 /app/data/labus_rag.db ".backup /tmp/labus_rag_snapshot.db"
docker cp labus_api:/tmp/labus_rag_snapshot.db /tmp/labus_rag_snapshot.db

# [локально]
scp -P 22 root@62.217.178.117:/tmp/labus_rag_snapshot.db \
  d:/SALES_RAG/_dbdump/$(date +%Y-%m-%d)/labus_rag.db
```

---

## 4. Smoke-тесты после деплоя

```bash
# describe intent — должен пойти в LLM, НЕ «Вывески 6000₽»
time curl -sS -X POST https://ai.labus.pro/query_structured -H "Content-Type: application/json" \
  -d '{"query":"У меня есть смета для вывески: 1 Макет 3000₽, 2 Монтаж 3780₽..."}' | head -c 400

# under-key объёмные буквы — «Световые вывески» / «Объёмные буквы»
time curl -sS -X POST https://ai.labus.pro/query_structured -H "Content-Type: application/json" \
  -d '{"query":"Кафе КУРУШ. Объёмные буквы 45 см, подсветка..."}' | head -c 400

# SEED-based Логотип — 4 позиции, ~23к
time curl -sS -X POST https://ai.labus.pro/query_structured -H "Content-Type: application/json" \
  -d '{"query":"Сколько стоит логотип?"}' | head -c 400

docker logs --tail 80 labus_api 2>&1 | grep -E "describe/provided-smeta|junk category|under min total|hit pre-LLM|blocked by under-key|SEED|Логотип"
```

---

## 5. Локальная разработка

```bash
cd RAG_RUNTIME
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

cp configs/.env.example configs/.env   # вставить DEEPSEEK_API_KEY

huggingface-cli download BAAI/bge-m3 --local-dir models/embeddings/BAAI_bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/reranker/BAAI_bge-reranker-v2-m3

docker compose up qdrant -d
python scripts/ingest.py && python scripts/ingest_knowledge.py && python scripts/ingest_roadmaps.py
python scripts/build_index.py --recreate

uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## 6. API

| Endpoint | Метод | Описание |
|----------|-------|---------|
| `/health` | GET | Статус компонентов |
| `/query` | POST | Human-readable ответ |
| `/query_structured` | POST | JSON: pricing, bundle, references, smeta |
| `/rebuild_index` | POST | Перестройка индекса (защищено) |

Прод: `https://ai.labus.pro` (nginx `proxy_*_timeout 180s`)

---

## 7. Operational notes

- `workers=1` обязательно — BGE-M3 держится в памяти воркера.
- Qdrant стартует до API (compose dependency).
- Windows: `PYTHONIOENCODING=utf-8` для скриптов с Юникодом.
- SmetaEngine LRU cache: 256 слотов; ключ = (normalized_query, letter_count, height_cm, linear_meters).
- `.env` на сервере не перезаписывается `git pull` — хранится вне git-tree в `/opt/rag/configs/.env`.
- `/opt/rag/app/` и `/opt/rag/RAG_RUNTIME/` на сервере — устаревшие mirror'ы, оставлены как бэкап, **не использовать** для деплоя.
- Git на сервере может быть отстающим от `main` — при `git pull` проверяй вывод на merge-конфликты с старым scp-дрейфом.
