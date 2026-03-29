# AGENTS.md

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
