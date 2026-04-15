# AGENTS.md

## Project: Labus Sales RAG

Production RAG-система для labus.pro: подбор товаров/услуг, детерминированная оценка смет, КП-генерация на базе истории Bitrix24.

**Stack:** Python 3.11 · FastAPI · Qdrant · BGE-M3 · BGE-Reranker-v2-m3 · Deepseek API · Node.js (analytics) · Docker · SQLite (чаты+feedback) · Gemini/Ollama Vision

**Key entry points:**
- API: [RAG_RUNTIME/app/main.py](RAG_RUNTIME/app/main.py) — FastAPI factory, lifespan загружает retriever/reranker/generator/pricing/vision/feedback/deal_lookup/photo_index/smeta_engine
- Routers: [app/routers/](RAG_RUNTIME/app/routers/) — `health`, `auth_router`, `chats`, `query`, `admin`, `eval`
- Core: [app/core/](RAG_RUNTIME/app/core/) — `retriever`, `reranker`, `generator`, `pricing_resolver`, `smeta_engine`, `query_parser`, `query_decomposer`, `deal_lookup`, `parametric_calculator`, `photo_index`, `feedback_store`, `vision`
- Analytics (Node, build-host only): [RAG_ANALYTICS/](RAG_ANALYTICS/) — `buildSmetaTemplates.mjs`, `buildFacts.mjs`, `buildPricing.mjs`, `runAll.mjs`
- Ingest/index: [RAG_RUNTIME/scripts/](RAG_RUNTIME/scripts/) — `ingest.py`, `ingest_knowledge.py`, `ingest_roadmaps.py`, `build_index.py`, `embed_smeta_categories.py`, `vision_analysis_local.py`

**Data layout:**
- `RAG_DATA/` — сырые CSV из Bitrix24 (`goods.csv`, `offers.csv`, `orders.csv`) — НЕ в git
- `RAG_ANALYTICS/output/` — fact tables, pricing recs, smeta_templates.json, category embeddings
- `RAG_RUNTIME/data/processed/` — JSONL knowledge docs для индекса
- `RAG_RUNTIME/models/` — BGE-M3 + Reranker (артефакты, НЕ в git)

**Prod:** `root@62.217.178.117` → Docker `labus_api` (image `rag-api`) → `https://ai.labus.pro`. Deploy через git pull + docker cp. Детали: [reference_prod_deploy_paths.md](../../.claude/projects/d--SALES-RAG/memory/reference_prod_deploy_paths.md) в memory, процедуры в [README.md](README.md) §3.

**Docs:**
- [README.md](README.md) — структура репо, процедуры деплоя
- [ARCHITECTURE.md](ARCHITECTURE.md) — компоненты, data flow, деградация
- [PLAN.MD](PLAN.MD) — roadmap и статус этапов
- [RUNBOOK.md](RUNBOOK.md) — операционные процедуры локальной разработки

**Code conventions:**
- `workers=1` для uvicorn (BGE-M3 ~2.2GB в процессе)
- `PYTHONIOENCODING=utf-8` для Windows скриптов
- Прод-артефакты smeta живут по абсолютному пути `/RAG_ANALYTICS/output/` в контейнере (НЕ под `/app/`)
- Тяжёлые артефакты идут через scp в `/opt/rag/artifacts/`, компактные (smeta_templates.json ~1.5MB) — коммитятся в git
- SEED_DEALS + keyword override в SmetaEngine — пре-семантический safety net для коротких запросов (см. [smeta_engine.py](RAG_RUNTIME/app/core/smeta_engine.py))

---

## Codex MCP Policy

This machine has a preinstalled unified Codex MCP stack under `%USERPROFILE%\.codex\mcp-memory-stack`.

Use the full available MCP surface by default when it improves accuracy, recall, or speed:

1. `memory_chroma_l1`
2. `memory_milvus_l2`
3. `memory_milvus_l2_b`
4. `memory_qdrant_l3`
5. `memory_qdrant_l3_b`
6. `memory_sql_l4_a`
7. `memory_sql_l4_b`
8. `memory_lexical_l5`
9. `memory_reranker_l5`
10. `playwright`
11. `ui_ux_pro`

## Default Behavior For Every New Project

1. Assume the unified MCP stack should be used unless the user explicitly disables it.
2. Before substantial work, check stack status with:

```powershell
powershell -ExecutionPolicy Bypass -File "%USERPROFILE%\.codex\mcp-memory-stack\scripts\status-stack.ps1" -Hardware auto
```

3. If the stack looks unhealthy or a tool fails unexpectedly, run:

```powershell
powershell -ExecutionPolicy Bypass -File "%USERPROFILE%\.codex\mcp-memory-stack\scripts\validate-stack.ps1" -Client codex -Hardware auto
```

4. For a newly opened project, strongly prefer indexing the workspace into project memory before deep work:

```powershell
powershell -ExecutionPolicy Bypass -File "%USERPROFILE%\.codex\mcp-memory-stack\scripts\ingest-workspace-to-chroma.ps1" -Root "<PROJECT_ROOT>"
powershell -ExecutionPolicy Bypass -File "%USERPROFILE%\.codex\mcp-memory-stack\scripts\ingest-workspace-to-qdrant.ps1" -Root "<PROJECT_ROOT>"
```

5. Treat `memory_chroma_l1` as persistent local workspace memory.
6. Treat `memory_qdrant_l3/_b` as durable semantic project memory.
7. Treat `memory_milvus_l2/_b` as dense vector retrieval windows for large or evolving context.
8. Use `memory_lexical_l5` and `memory_reranker_l5` when retrieval quality matters more than raw speed.
9. Use `memory_sql_l4_a/_b` for structured memory and SQLite-backed inspection.
10. Use `playwright` for real browser tasks.
11. Use `ui_ux_pro` for UI, UX, design-system, and component work.

## Operational Guardrails

1. Do not switch the stack back to global `uv tool`, `.local\bin`, or unpinned `@latest` runtimes.
2. Do not disable Qdrant post-index warmup after reindexing.
3. Do not change launchers, manifest, or generated Codex config unless the task actually requires stack maintenance.
4. If launchers or Codex MCP config change, restart the VS Code window before trusting new MCP behavior.
5. Do not treat one slow first call as a functional failure; distinguish cold-start from a broken server.

## When To Run Deeper Checks

Run deeper checks only when the task or symptoms justify it:

```powershell
powershell -ExecutionPolicy Bypass -File "%USERPROFILE%\.codex\mcp-memory-stack\scripts\run-mcp-latency-regression.ps1" -Iterations 2
powershell -ExecutionPolicy Bypass -File "%USERPROFILE%\.codex\mcp-memory-stack\scripts\run-stress-audit-60s.ps1" -CleanupAfterRun
```

## Reference Docs

Use these docs when stack setup, recovery, or operational details are needed:

1. `QUICKSTART.md`
2. `OPERATOR_CHECKLIST.md`
3. `NEW_MACHINE_BOOTSTRAP.md`
4. `CODEX_MCP_STACK_ROADMAP.md`
