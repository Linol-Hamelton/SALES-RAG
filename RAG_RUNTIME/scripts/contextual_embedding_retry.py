"""P22.A.1.retry: find docs with empty context_prefix in data_contextual/ and retry.

Why: during initial run some calls hit 502 Bad Gateway and were saved with
context_prefix="" (no context). This script re-processes only those failed
docs using the same DeepSeek pipeline but with retry-on-5xx logic.

Reads:  RAG_RUNTIME/data_contextual/*.jsonl  (existing partial output)
Writes: in-place updated jsonl files (atomic: write to .tmp then rename)

Usage: python RAG_RUNTIME/scripts/contextual_embedding_retry.py [--only FILE.jsonl]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_contextual"

PROD_URL = "https://62.217.178.117/query_no_rag"
HOST_HEADER = "ai.labus.pro"

CONCURRENCY = 8
TIMEOUT = 60.0
MAX_CONTEXT_TOKENS = 100
MAX_RETRIES = 5

# Doc types that needed LLM context (template was sufficient for others).
# These are the ones where empty context_prefix means LLM failure.
NARRATIVE_DOC_TYPES = {
    "product", "bundle", "historical_deal", "deal_profile", "offer_profile",
    "offer_composition", "knowledge", "faq", "photo_analysis", "service_page",
}

CONTEXT_PROMPT_TEMPLATE = """Дай короткое описание (50-100 токенов на русском) контекста этого
документа для улучшения поиска. Включи: направление (Цех/Печать/Дизайн/РИК/Мерч/Сольвент),
тип изделия (буклет/листовка/визитка/объёмные буквы/баннер/логотип/etc.),
ключевой ценовой/сделочный контекст. Только сам текст контекста, без преамбулы.

Документ:
{doc_text}

Контекст для поиска:"""


async def llm_context(client: httpx.AsyncClient, doc_text: str,
                     sem: asyncio.Semaphore) -> str | None:
    doc_text = doc_text[:2000]
    payload = {
        "query": CONTEXT_PROMPT_TEMPLATE.format(doc_text=doc_text),
        "mode": "human",
        "system_prompt_mode": "minimal",
        "model_override": "deepseek-chat",
        "temperature": 0.2,
        "max_tokens_override": MAX_CONTEXT_TOKENS + 20,
    }
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    PROD_URL, json=payload, headers={"Host": HOST_HEADER},
                    timeout=TIMEOUT,
                )
                if resp.status_code in (500, 502, 503, 504):
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                resp.raise_for_status()
                data = resp.json()
                ctx = (data.get("summary") or "").strip()
                for prefix in ("Контекст для поиска:", "Контекст:", "Контекст -"):
                    if ctx.lower().startswith(prefix.lower()):
                        ctx = ctx[len(prefix):].strip()
                return ctx[:500] if ctx else None
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None
            except Exception as e:
                print(f"  retry err: {type(e).__name__}: {e}", flush=True)
                return None
        return None


async def process_file(path: Path):
    """Re-process docs with empty context_prefix in-place."""
    print(f"\n=== {path.name} ===", flush=True)
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"  Loaded {len(docs)} docs", flush=True)

    # Find docs with empty context_prefix AND narrative doc_type
    failed = [
        (i, d) for i, d in enumerate(docs)
        if not (d.get("context_prefix") or "").strip()
        and d.get("doc_type") in NARRATIVE_DOC_TYPES
    ]
    if not failed:
        print(f"  No failed docs to retry. Skipping.", flush=True)
        return

    print(f"  Found {len(failed)} failed docs to retry", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.monotonic()
    success = 0
    still_failed = 0

    async with httpx.AsyncClient(verify=False) as client:
        for batch_start in range(0, len(failed), 50):
            batch = failed[batch_start:batch_start + 50]
            tasks = [
                llm_context(client,
                           d.get("searchable_text_orig") or d.get("searchable_text") or "",
                           sem)
                for _, d in batch
            ]
            results = await asyncio.gather(*tasks)
            for (idx, doc), ctx in zip(batch, results):
                if ctx:
                    orig = doc.get("searchable_text_orig") or doc.get("searchable_text") or ""
                    doc["context_prefix"] = ctx
                    doc["searchable_text"] = f"{ctx}\n\n{orig}"
                    success += 1
                else:
                    still_failed += 1
            elapsed = time.monotonic() - t0
            print(f"  [{batch_start + len(batch)}/{len(failed)}] "
                  f"ok={success} stuck={still_failed} elapsed={elapsed:.0f}s",
                  flush=True)

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fout:
        for d in docs:
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"  ✓ {path.name}: recovered {success}, still {still_failed} stuck", flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="Process only this file (basename)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    import warnings
    warnings.filterwarnings("ignore")
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    files = sorted(DATA_DIR.glob("*.jsonl"))
    for f in files:
        if args.only and f.name != args.only:
            continue
        await process_file(f)

    print("\n=== Retry pass done ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
