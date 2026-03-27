# Labus Sales RAG — Runbook

Операционный справочник: установка, запуск, пересборка, диагностика.

---

## 1. Первоначальная установка

### 1.1 Требования

- Python 3.11+
- Docker Desktop (для Qdrant)
- CUDA 12.x + GPU с ≥6 GB VRAM (рекомендуется; CPU-режим тоже работает)
- HuggingFace CLI: `pip install huggingface_hub`
- Deepseek API key (для LLM-генерации)

### 1.2 Создать окружение

```bash
cd RAG_RUNTIME
python -m venv .venv
source .venv/Scripts/activate        # Windows
# source .venv/bin/activate          # Linux/Mac

# Основные зависимости
pip install -r requirements.txt

# GPU PyTorch (выбрать нужную версию CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# pip install torch torchvision  # CPU-only
```

### 1.3 Настроить конфиг

```bash
# Отредактировать configs/.env:
DEEPSEEK_API_KEY=sk-YOUR-KEY-HERE
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=labus_docs

EMBEDDING_MODEL_PATH=models/embeddings/BAAI_bge-m3
RERANKER_MODEL_PATH=models/reranker/BAAI_bge-reranker-v2-m3
```

### 1.4 Загрузить локальные модели

```bash
# BGE-M3 embedding model (~2.2 GB)
huggingface-cli download BAAI/bge-m3 \
  --local-dir models/embeddings/BAAI_bge-m3

# BGE-Reranker-v2-m3 (~0.6 GB)
huggingface-cli download BAAI/bge-reranker-v2-m3 \
  --local-dir models/reranker/BAAI_bge-reranker-v2-m3
```

Если HuggingFace недоступен из-за сети:
```bash
# Через прокси или зеркало:
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-m3 \
  --local-dir models/embeddings/BAAI_bge-m3
```

### 1.5 Проверить окружение

```bash
python scripts/check_gpu.py
```

Ожидаемый вывод:
```
============================================================
SALES_RAG Environment Check
Platform: Windows 11
============================================================
✓ Python 3.11.x
✓ PyTorch 2.x.x
  GPU 0: NVIDIA GeForce RTX XXXX (N.N GB VRAM)
✓ sentence-transformers 3.x.x
✓ qdrant-client 1.x.x
✓ openai 1.x.x

Model test:
  Loading BGE-M3 on cuda...
  ✓ BGE-M3 loaded! Embedding shape: (1, 1024), device: cuda
  VRAM used: 2.20 GB
============================================================
✓ Environment ready!
```

---

## 2. Запуск инфраструктуры

### 2.1 Запустить Qdrant

```bash
# В отдельном терминале или в фоне
docker compose up qdrant -d

# Проверить что работает
curl http://localhost:6333/
```

### 2.2 Первичная инgestация и индексация

```bash
# Шаг 1: Аудит данных
python scripts/data_report.py

# Шаг 2: Построить knowledge docs (JSONL)
python scripts/ingest.py --verbose

# Шаг 3: Embeddings + Qdrant + BM25
python scripts/build_index.py --verbose

# Шаг 3a: Только BM25 (быстро, без GPU)
python scripts/build_index.py --skip-qdrant

# Шаг 3b: Пересоздать коллекцию Qdrant с нуля
python scripts/build_index.py --recreate
```

---

## 3. Запуск сервера

```bash
# ВАЖНО: workers=1 обязательно (embedding model в памяти процесса)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# С логами в файл
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 \
  2>&1 | tee logs/server.log
```

Проверить:
```bash
curl http://localhost:8000/health
```

---

## 4. Тестовые запросы

### CLI (без сервера)

```bash
python scripts/query_cli.py --query "световая вывеска кофейня монтаж под ключ"
python scripts/query_cli.py --query "баннер 3х6 цена" --mode structured
python scripts/query_cli.py --query "безнал 10%" --show-context
```

### HTTP API

```bash
# Human-readable
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "объёмные буквы для магазина"}' | python -m json.tool

# Структурированный (JSON с pricing, bundle, references)
curl -s -X POST http://localhost:8000/query_structured \
  -H "Content-Type: application/json" \
  -d '{"query": "световая вывеска под ключ аптека"}' | python -m json.tool
```

---

## 5. Eval

```bash
# Все кейсы (46 штук, нужен запущенный сервер)
python scripts/eval.py --verbose

# Только конкретные категории
python scripts/eval.py --categories auto_priced guided_priced manual_priced

# Bundle и ambiguous
python scripts/eval.py --categories bundle_query direction_ambiguous edge_case
```

Результаты:
- `eval/results.json` — детальные результаты
- `reports/model_readiness_report.md` — отчёт
- `reports/model_readiness_report.json` — машиночитаемый отчёт

---

## 6. Пересборка индекса

```bash
# Через API (сервер не останавливается, индекс меняется в фоне)
curl -X POST http://localhost:8000/rebuild_index

# Или через скрипт напрямую
python scripts/ingest.py
python scripts/build_index.py --recreate
```

---

## 7. Диагностика

### Сервер не стартует

```
# Проверить Qdrant
curl http://localhost:6333/

# Проверить модели
ls models/embeddings/BAAI_bge-m3/
ls models/reranker/BAAI_bge-reranker-v2-m3/

# Проверить .env
cat configs/.env

# Проверить data
ls data/
ls data/processed/
```

### CUDA / GPU проблемы

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Если CUDA недоступна — принудительный CPU режим
# В configs/.env добавить:
# DEVICE=cpu
```

### Кодировка кракозябры в Windows

```bash
# В PowerShell / CMD:
chcp 65001

# Проверить фактическую кодировку файлов:
python -c "
import chardet
with open('data/processed/product_docs.jsonl', 'rb') as f:
    raw = f.read(1000)
print(chardet.detect(raw))
"
```

### Qdrant коллекция повреждена

```bash
# Полное пересоздание
python scripts/build_index.py --recreate --verbose
```

### Deepseek API недоступен

```
# Система продолжает работать — retrieval и pricing работают без LLM
# Ответ /query вернёт structured data без summary и reasoning
```

### Высокая задержка

```
# Типичные причины:
# 1. Qdrant на HDD — переместить qdrant_data/ на SSD
# 2. Embedding model на CPU — убедиться в CUDA
# 3. Слишком большой batch — уменьшить RETRIEVAL_TOP_K в .env
```

---

## 8. Мониторинг

```bash
# Логи структурированные (JSON)
tail -f logs/app.log | python -m json.tool

# Health endpoint
watch -n 5 'curl -s http://localhost:8000/health | python -m json.tool'

# Последний eval
curl http://localhost:8000/eval_summary
```

---

## 9. Перенос на новую машину

```bash
# Что копировать:
rsync -av --progress \
  models/ \
  data/processed/ \
  indexes/ \
  configs/.env \
  user@new-machine:~/SALES_RAG/RAG_RUNTIME/

# На новой машине:
cd RAG_RUNTIME
pip install -r requirements.txt
docker compose up qdrant -d
python scripts/build_index.py --recreate   # переиндексация в Qdrant из data/
uvicorn app.main:app --port 8000 --workers 1
```

---

## 10. Обновление данных

Если обновились источники в `RAG_ANALYTICS/output/`:

```bash
# Полный цикл пересборки
python scripts/data_report.py           # проверка новых данных
python scripts/ingest.py                # пересоздать knowledge docs
python scripts/build_index.py --recreate # пересобрать индекс
python scripts/eval.py --verbose        # проверить качество
```
