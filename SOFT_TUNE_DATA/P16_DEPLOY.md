# P16 Phase A Deploy Manifest

Эта инструкция — для развёртывания **P16 Phase A** на прод (62.217.178.117).
Все code-патчи готовы локально и smoke-протестированы. Этот документ нужен
чтобы прокатить их на live API.

## Что меняется

| Слой | Файл | Что |
|---|---|---|
| Config | `RAG_RUNTIME/app/config.py` | + `reranker_finetuned_path` field |
| Reranker | `RAG_RUNTIME/app/core/reranker.py` | Fallback chain finetuned→base→HF, weight validation |
| Retriever | `RAG_RUNTIME/app/core/retriever.py` | `_lexical_match_boost()` post-RRF |
| Query | `RAG_RUNTIME/app/routers/query.py` | `_ensure_deal_items_filled`, `_ensure_lead_time_filled`, HyDE default-ON, few-shot block |
| Intent | `RAG_RUNTIME/app/core/intent_classifier.py` | Fallback path resolution |
| Configs | `configs/intent_prototypes.yaml` | objection_arguments 2→40, category_clarify 0→31 |
| Configs | `RAG_RUNTIME/configs/prompts.yaml` | + `few_shot_examples` block |
| Model | `SOFT_TUNE_DATA/reranker_finetuned_v2/` | 2.2GB fine-tuned weights (binary, не в git) |

## Pre-flight: что должно быть готово

- [x] Все code-патчи локально smoke-протестированы (`PYTHONIOENCODING=utf-8 RAG_RUNTIME/.venv/Scripts/python.exe -c "from app.routers import query; ..."`)
- [x] `SOFT_TUNE_DATA/reranker_finetuned_v2/` весит ~2.2GB
- [x] `git status` clean кроме untracked artefacts (training scripts v3, q-gen v13)
- [ ] git commit + push (по запросу пользователя)

## Deploy последовательность (~30 min)

### Шаг 0: SSH на prod

```bash
ssh root@62.217.178.117
cd /opt/rag
```

### Шаг 1: git pull (если коммит сделан)

```bash
git pull origin main
```

### Шаг 2: docker cp кода

```bash
# App folder — все изменения в core, routers, config
docker cp RAG_RUNTIME/app/. labus_api:/app/app/

# Configs — intent_prototypes (root) + prompts (RAG_RUNTIME)
docker cp configs/intent_prototypes.yaml labus_api:/app/configs/intent_prototypes.yaml
docker cp RAG_RUNTIME/configs/prompts.yaml labus_api:/app/configs/prompts.yaml
```

### Шаг 3: docker cp fine-tuned reranker v3 (CRITICAL)

⚠️ Используем **v3** (P16.A.2 trained 2026-05-16, collision-aware) — НЕ v2.
v3 binary eval 0.935, collision smoke 4/5 (vs v2 baseline 1/5).
Config по умолчанию указывает на `models/reranker_finetuned_v3`.

```bash
# 2.2GB — займёт ~3-5 минут
docker cp SOFT_TUNE_DATA/reranker_finetuned_v3/. labus_api:/app/models/reranker_finetuned_v3/

# Verify
docker exec labus_api ls -lh /app/models/reranker_finetuned_v3/model.safetensors
# Should show ~2.2GB

# Optional: if you have older _v2 model on disk and want to free space:
# docker exec labus_api rm -rf /app/models/reranker_finetuned_v2
```

### Шаг 4: restart API

```bash
docker restart labus_api
sleep 60   # wait for embedding + reranker load
```

### Шаг 5: smoke test

```bash
# Health
curl -s -k https://62.217.178.117/health
# {"status":"ok"}

# Verify reranker_finetuned_v2 loaded (look in logs)
docker logs labus_api 2>&1 | grep -i "reranker" | tail -5
# Expect: "Loading reranker path=/app/models/reranker_finetuned_v2"

# Verify HyDE auto-ON
docker logs labus_api 2>&1 | grep -i "hyde" | tail -3
# Expect cache hits or "HyDE query enriched" lines after a few queries

# Five-test smoke
for Q in \
  "Сколько стоят 1000 визиток" \
  "Хочу заказать буклеты для стоматологии" \
  "Дайте статистику возражение дорого" \
  "У вас есть фото проектов для ресторанов" \
  "Как обосновать цену через ROI"; do
  echo "=== $Q ==="
  curl -s -k -X POST https://62.217.178.117/query_structured \
    -H "Host: ai.labus.pro" \
    -d "{\"query\":\"$Q\",\"top_k\":5}" | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(f'  intent ok | price={r.get(\"estimated_price\",{}).get(\"value\") if r.get(\"estimated_price\") else None}')
print(f'  lead_time={r.get(\"estimated_lead_time\")}')
print(f'  deal_items={len(r.get(\"deal_items\",[]))}')
print(f'  summary={r.get(\"summary\",\"\")[:100]}')
"
done
```

**Critical observation:** `lead_time` and `deal_items` should be filled
for pricing queries (P16.A.5 + A.6 guarantees).

### Шаг 6 (опционально, после A.2 train): replace reranker_v2 → v3

Когда `reranker_finetuned_v3/` готов:

```bash
docker cp SOFT_TUNE_DATA/reranker_finetuned_v3/. labus_api:/app/models/reranker_finetuned_v3/
# Update env to force v3 path
echo "RERANKER_FINETUNED_PATH=/app/models/reranker_finetuned_v3" >> .env
docker restart labus_api
```

## Откат

Если smoke падает или 5 минут логов показывают регрессию:

```bash
# Roll back code (если git commit был)
git revert HEAD
docker cp RAG_RUNTIME/app/. labus_api:/app/app/
docker cp configs/intent_prototypes.yaml labus_api:/app/configs/intent_prototypes.yaml
docker cp RAG_RUNTIME/configs/prompts.yaml labus_api:/app/configs/prompts.yaml
docker restart labus_api

# Reranker не откатывается — fallback chain автоматически берёт base
# если папка finetuned пустая или удалена:
docker exec labus_api rm -rf /app/models/reranker_finetuned_v2
docker restart labus_api
# В логах: "Skipping reranker path (no weights)" → "Loading reranker /app/models/reranker/..."
```

## После успешного deploy → запуск v13

1. **Generate v13 questions (~5 min):**
   ```powershell
   $env:PYTHONIOENCODING="utf-8"
   RAG_RUNTIME\.venv\Scripts\python.exe SOFT_TUNE_DATA\scripts\generate_questions_v13.py
   ```

2. **Collect 3200 answers (~3-4h):**
   ```powershell
   $env:RUN_SUFFIX="_v13"
   RAG_RUNTIME\.venv\Scripts\python.exe SOFT_TUNE_DATA\scripts\collect_answers.py
   ```

3. **Judge 1600 pairs with recalibrated prompt (~3-4h):**
   ```powershell
   $env:RUN_SUFFIX="_v13"
   $env:JUDGE_PROMPT_VERSION="v2"
   RAG_RUNTIME\.venv\Scripts\python.exe SOFT_TUNE_DATA\scripts\judge_and_recommend.py
   ```

4. **Final report:** автоматически — `SOFT_TUNE_DATA/recommendations_v13.md`.

## Параллельный план: B.2 baseline (БЕЗ deploy)

В background сейчас (PID `btir0tvqt`):
```
RUN_SUFFIX=_v12 OUT_SUFFIX=_v12_recalibrated JUDGE_PROMPT_VERSION=v2 \
  python SOFT_TUNE_DATA/scripts/judge_and_recommend.py
```

Output: `SOFT_TUNE_DATA/scores_v12_recalibrated.jsonl` + `recommendations_v12_recalibrated.md`.

Этот прогон НЕ требует прод-деплоя — использует existing v12 answers и
только перепроверяет judge. Покажет how much из 27% no_rag wins были
artificial penalties за pricing_grounded calibration.

**Ожидание:** v12 win rate 73% → 88-92% после recalibration (по гипотезе agent 2).
Если так — Phase B математически работает, остаётся только Phase A (через
deploy + v13 collect) для выхода на 95%+.
