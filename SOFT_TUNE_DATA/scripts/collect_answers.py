"""P14 Phase D: collect 200×2 answers from prod (with RAG + without RAG).

Reads:
  SOFT_TUNE_DATA/questions_client.jsonl
  SOFT_TUNE_DATA/questions_manager.jsonl

Writes:
  SOFT_TUNE_DATA/answers_rag.jsonl    — POST /query_structured (full RAG)
  SOFT_TUNE_DATA/answers_no_rag.jsonl — POST /query_no_rag (bypass)

Concurrency capped at 6 to avoid DeepSeek rate limits.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "SOFT_TUNE_DATA"
OUT_RAG = DATA_DIR / "answers_rag.jsonl"
OUT_NO_RAG = DATA_DIR / "answers_no_rag.jsonl"

BASE = "https://62.217.178.117"
HOST = "ai.labus.pro"
RAG_URL = f"{BASE}/query_structured"
NO_RAG_URL = f"{BASE}/query_no_rag"

CONCURRENCY = 6
TIMEOUT = 180.0


def load_questions() -> list[dict]:
    rows: list[dict] = []
    for fname in ("questions_client.jsonl", "questions_manager.jsonl"):
        p = DATA_DIR / fname
        if not p.exists():
            print(f"!! {p} missing — run generate_questions.py first")
            sys.exit(1)
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def call_rag(client: httpx.AsyncClient, q: dict) -> dict:
    t0 = time.monotonic()
    try:
        resp = await client.post(
            RAG_URL,
            json={"query": q["question"], "top_k": 10},
            headers={"Host": HOST, "Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "id": q["id"],
            "persona": q["persona"],
            "topic": q["topic"],
            "query": q["question"],
            "summary": data.get("summary") or "",
            "deal_items_count": len(data.get("deal_items") or []),
            "references_count": len(data.get("references") or []),
            "references_doc_types": [r.get("doc_type") for r in (data.get("references") or [])],
            "historical_deals_count": len(data.get("historical_deals") or []),
            "total_value": (data.get("total_value_suggestion") or {}).get("value"),
            "intent": data.get("intent"),
            "intent_confidence": data.get("intent_confidence"),
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "error": None,
            "full_response": data,
        }
    except Exception as e:
        return {
            "id": q["id"], "persona": q["persona"], "topic": q["topic"],
            "query": q["question"], "summary": "", "error": f"{type(e).__name__}: {e}",
            "latency_ms": int((time.monotonic() - t0) * 1000),
        }


async def call_no_rag(client: httpx.AsyncClient, q: dict) -> dict:
    t0 = time.monotonic()
    try:
        resp = await client.post(
            NO_RAG_URL,
            json={
                "query": q["question"],
                "mode": "structured",
                "system_prompt_mode": "full",
            },
            headers={"Host": HOST, "Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "id": q["id"],
            "persona": q["persona"],
            "topic": q["topic"],
            "query": q["question"],
            "summary": data.get("summary") or "",
            "raw_response": data.get("raw_response") or "",
            "model": data.get("model"),
            "prompt_tokens": data.get("prompt_tokens", 0),
            "completion_tokens": data.get("completion_tokens", 0),
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "error": None,
        }
    except Exception as e:
        return {
            "id": q["id"], "persona": q["persona"], "topic": q["topic"],
            "query": q["question"], "summary": "", "error": f"{type(e).__name__}: {e}",
            "latency_ms": int((time.monotonic() - t0) * 1000),
        }


async def run_pool(questions: list[dict], call_fn, label: str) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[dict] = []
    counter = {"done": 0, "ok": 0, "fail": 0}
    total = len(questions)

    async with httpx.AsyncClient(verify=False) as client:
        async def worker(q):
            async with sem:
                r = await call_fn(client, q)
                counter["done"] += 1
                if r.get("error"):
                    counter["fail"] += 1
                else:
                    counter["ok"] += 1
                if counter["done"] % 10 == 0 or counter["done"] == total:
                    print(f"  [{label}] {counter['done']}/{total} ok={counter['ok']} fail={counter['fail']}", flush=True)
                return r

        results = await asyncio.gather(*[worker(q) for q in questions])
    return results


async def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    print("\n=== Phase D-1: collecting RAG answers ===")
    rag = await run_pool(questions, call_rag, "RAG")
    with OUT_RAG.open("w", encoding="utf-8") as f:
        for r in rag:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ {len(rag)} → {OUT_RAG}")

    print("\n=== Phase D-2: collecting no-RAG answers ===")
    no_rag = await run_pool(questions, call_no_rag, "NO-RAG")
    with OUT_NO_RAG.open("w", encoding="utf-8") as f:
        for r in no_rag:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ {len(no_rag)} → {OUT_NO_RAG}")


if __name__ == "__main__":
    import warnings, urllib3
    warnings.filterwarnings("ignore")
    urllib3.disable_warnings()
    asyncio.run(main())
