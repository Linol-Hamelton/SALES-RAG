"""P22.C.2: RAGAS-style multi-metric judge using DeepSeek-reasoner as evaluator.

Computes 4 metrics per (query, answer, refs) tuple following RAGAS methodology:
  - Faithfulness: % of claims in answer that are supported by refs
  - Answer Relevancy: how well answer addresses the query (0-1)
  - Context Precision: % of refs that are relevant to query (0-1)
  - Context Recall: % of needed info that is present in refs (0-1)
  - Composite: harmonic mean of the four

Adapted from RAGAS (Es et al. 2023) — since we don't have ground truth,
we ask the judge LLM to score each metric directly using rubrics.

Reads:
  SOFT_TUNE_DATA/answers_rag{SUFFIX}.jsonl

Writes:
  SOFT_TUNE_DATA/scores_ragas{SUFFIX}.jsonl

Concurrency capped to avoid rate limit. ~$1-2 for 1600 questions on deepseek-reasoner.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "SOFT_TUNE_DATA"
SUFFIX = os.environ.get("RUN_SUFFIX", "")
IN_FILE = DATA_DIR / f"answers_rag{SUFFIX}.jsonl"
OUT_FILE = DATA_DIR / f"scores_ragas{SUFFIX}.jsonl"

BASE = "https://62.217.178.117"
HOST = "ai.labus.pro"
URL = f"{BASE}/query_no_rag"
CONCURRENCY = 4
TIMEOUT = 300.0

JUDGE_SYSTEM = """Ты — эксперт-аналитик B2B-агентства полиграфии/рекламы (Лабус, Махачкала).
Оцениваешь качество RAG-ответа по 4 метрикам RAGAS:

1. FAITHFULNESS (0-1): доля утверждений в ответе, которые ПОДТВЕРЖДАЮТСЯ refs.
   - 1.0: каждое утверждение (цена, deal_id, факт) есть в refs
   - 0.5: половина утверждений подтверждена
   - 0.0: ответ содержит галлюцинации (выдуманные цены/deal_id/факты)

2. ANSWER RELEVANCY (0-1): насколько ответ отвечает НА ВОПРОС (не размывает тему).
   - 1.0: ответ прямо отвечает на конкретный вопрос
   - 0.5: частично отвечает / есть лишние темы
   - 0.0: ответ не по теме / общая вода

3. CONTEXT PRECISION (0-1): доля refs которые РЕАЛЬНО релевантны вопросу.
   - 1.0: все refs полезны для ответа
   - 0.5: половина refs полезны
   - 0.0: refs нерелевантны вопросу

4. CONTEXT RECALL (0-1): доля КЛЮЧЕВЫХ ФАКТОВ для ответа, которые есть в refs.
   - 1.0: refs содержат все нужные факты (цена, deal_id, состав, сроки)
   - 0.5: половина ключевых фактов есть в refs
   - 0.0: refs не содержат нужных для ответа фактов

Отвечай ТОЛЬКО валидным JSON. Без markdown."""

JUDGE_PROMPT_TEMPLATE = """ВОПРОС ({persona}, intent={intent}):
{query}

=== ОТВЕТ RAG-СИСТЕМЫ ===
{summary}

=== ПОИСКОВАЯ ВЫДАЧА (refs, top-{n_refs}) ===
{refs_block}

=== ЗАДАЧА ===
Оцени каждую из 4 метрик по шкале 0-1 (десятичная дробь):
- faithfulness: подтверждаются ли утверждения ответа в refs?
- answer_relevancy: отвечает ли ответ ТОЧНО на вопрос?
- context_precision: насколько релевантны refs к вопросу?
- context_recall: содержат ли refs все нужные факты для качественного ответа?

Также определи 2-3 missing_facts: какие конкретные факты должны быть в идеальном
ответе но отсутствуют в refs или ответе.

Верни ТОЛЬКО JSON:
{{
  "faithfulness": <0.0-1.0>,
  "answer_relevancy": <0.0-1.0>,
  "context_precision": <0.0-1.0>,
  "context_recall": <0.0-1.0>,
  "missing_facts": ["<факт 1>", "<факт 2>", ...],
  "hallucinations": ["<выдуманное утверждение>", ...],
  "reasoning": "<2-3 предложения почему такие оценки>"
}}"""


def harmonic_mean(values: list[float]) -> float:
    """Harmonic mean of 4 metrics — penalizes low scores more than arithmetic."""
    values = [v for v in values if v is not None]
    if not values:
        return 0.0
    if any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def format_refs_block(answer_row: dict, max_refs: int = 10, max_chars_per: int = 600) -> str:
    """Build refs text block for judge prompt."""
    full = answer_row.get("full_response") or {}
    refs = full.get("references") or []
    if not refs:
        return "(нет refs)"
    out = []
    for i, r in enumerate(refs[:max_refs], 1):
        doc_type = r.get("doc_type") or "?"
        text = (r.get("text") or r.get("searchable_text") or "")[:max_chars_per]
        out.append(f"[{i}] doc_type={doc_type}\n{text}")
    return "\n\n".join(out)


async def judge_one(client: httpx.AsyncClient, row: dict, sem: asyncio.Semaphore) -> dict:
    """Run 4-metric RAGAS judge on one Q/A/refs tuple."""
    query = row.get("query") or ""
    summary = row.get("summary") or ""
    full = row.get("full_response") or {}
    refs = full.get("references") or []
    n_refs = len(refs)

    user_prompt = JUDGE_PROMPT_TEMPLATE.format(
        persona=row.get("persona", "?"),
        intent=row.get("intent") or "?",
        query=query,
        summary=summary,
        n_refs=n_refs,
        refs_block=format_refs_block(row),
    )

    payload = {
        "query": user_prompt,
        "mode": "structured",
        "system_prompt_mode": "custom",
        "custom_system_prompt": JUDGE_SYSTEM,
        "model_override": "deepseek-reasoner",
        "response_format_override": "json",
        "temperature": 0.1,
        "max_tokens_override": 2000,
    }

    async with sem:
        try:
            resp = await client.post(
                URL, json=payload, headers={"Host": HOST}, timeout=TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("raw_response") or "{}"
            scores = json.loads(raw)

            metrics = [
                float(scores.get("faithfulness") or 0),
                float(scores.get("answer_relevancy") or 0),
                float(scores.get("context_precision") or 0),
                float(scores.get("context_recall") or 0),
            ]
            composite = round(harmonic_mean(metrics), 3)

            return {
                "id": row.get("id"),
                "persona": row.get("persona"),
                "topic": row.get("topic"),
                "intent": row.get("intent"),
                "query": query[:200],
                "faithfulness": metrics[0],
                "answer_relevancy": metrics[1],
                "context_precision": metrics[2],
                "context_recall": metrics[3],
                "composite": composite,
                "missing_facts": scores.get("missing_facts") or [],
                "hallucinations": scores.get("hallucinations") or [],
                "reasoning": scores.get("reasoning") or "",
                "n_refs": n_refs,
                "error": None,
            }
        except Exception as e:
            return {
                "id": row.get("id"),
                "error": f"{type(e).__name__}: {e}",
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
                "composite": None,
            }


async def main():
    sys.stdout.reconfigure(encoding="utf-8")

    if not IN_FILE.exists():
        print(f"!! {IN_FILE} missing — run collect_answers.py first", flush=True)
        sys.exit(1)

    rows: list[dict] = []
    for line in IN_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(rows)} answer rows from {IN_FILE.name}", flush=True)

    # Resume support: skip already-judged ids
    judged_ids = set()
    if OUT_FILE.exists():
        for line in OUT_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    judged_ids.add(json.loads(line).get("id"))
                except json.JSONDecodeError:
                    continue
        print(f"Resume: skipping {len(judged_ids)} already-judged", flush=True)

    todo = [r for r in rows if r.get("id") not in judged_ids]
    print(f"TODO: {len(todo)} rows to judge", flush=True)
    if not todo:
        print("All done.", flush=True)
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    mode = "a" if judged_ids else "w"
    t0 = time.monotonic()
    n_done = 0

    async with httpx.AsyncClient(verify=False) as client:
        with OUT_FILE.open(mode, encoding="utf-8") as fout:
            BATCH = 20
            for batch_start in range(0, len(todo), BATCH):
                batch = todo[batch_start:batch_start + BATCH]
                results = await asyncio.gather(*[judge_one(client, r, sem) for r in batch])
                for r in results:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += len(batch)
                rate = n_done / max(0.1, time.monotonic() - t0)
                eta_min = (len(todo) - n_done) / max(0.01, rate) / 60
                print(f"  [{n_done}/{len(todo)}] rate={rate:.2f}/s eta={eta_min:.1f}min",
                      flush=True)

    print(f"\nWrote {OUT_FILE} ({OUT_FILE.stat().st_size:,} bytes)", flush=True)
    elapsed_min = (time.monotonic() - t0) / 60
    print(f"Total time: {elapsed_min:.1f} min", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
