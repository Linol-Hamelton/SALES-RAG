"""P22.A.1: Contextual Retrieval prefix generation (Anthropic methodology).

Source: https://www.anthropic.com/news/contextual-retrieval (Sept 2024)

For each doc/chunk, generate 50-100 token Russian context prefix via DeepSeek
(cheap, our existing integration). Prepend to chunk text BEFORE embedding.
Anthropic's published numbers: +35% retrieval improvement with this technique.

LLM: deepseek-chat (~$11-22 total for 62k docs vs $62 для Claude Haiku).

Input:  RAG_RUNTIME/data/*.jsonl (original 62k docs)
Output: RAG_RUNTIME/data_contextual/*.jsonl
        Same structure, but searchable_text = context_prefix + "\\n\\n" + original

Optimization paths:
1. Template-based context for metadata-rich doc types (free, deterministic)
2. LLM-generated context for narrative docs (knowledge, faq, deal descriptions)
3. Skip generation if doc already has rich context (service_pricing_bridge, roadmap)

Run: python scripts/contextual_embedding_prep.py [--limit N] [--resume]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "data_contextual"

# Prod DeepSeek endpoint (no local DEEPSEEK_API_KEY needed — server key reused)
PROD_URL = "https://62.217.178.117/query_no_rag"
HOST_HEADER = "ai.labus.pro"

CONCURRENCY = 10  # parallel calls
TIMEOUT = 60.0    # per-call timeout
MAX_CONTEXT_TOKENS = 100  # output length per prefix
SKIP_FILES = {"photo_analysis_raw.jsonl", "bridge_unresolved.jsonl"}


CONTEXT_PROMPT_TEMPLATE = """Дай короткое описание (50-100 токенов на русском) контекста этого
документа для улучшения поиска. Включи: направление (Цех/Печать/Дизайн/РИК/Мерч/Сольвент),
тип изделия (буклет/листовка/визитка/объёмные буквы/баннер/логотип/etc.),
ключевой ценовой/сделочный контекст. Только сам текст контекста, без преамбулы.

Документ:
{doc_text}

Контекст для поиска:"""


# ---------------------------------------------------------------------------
# Template-based context (free, deterministic) для metadata-rich doc types.
# Используется для документов где payload уже содержит достаточно структурной
# информации — LLM call избыточен.
# ---------------------------------------------------------------------------
def template_context(payload: dict, doc_type: str) -> str | None:
    """Generate context from payload metadata. Returns None если требуется LLM."""
    direction = payload.get("direction", "") or ""

    if doc_type == "pricing_policy":
        return (f"Политика ценообразования направления {direction}. "
                f"Тип: {payload.get('price_mode', 'manual')}. "
                f"{(payload.get('rule', '') or '')[:120]}")

    elif doc_type == "service_pricing_bridge":
        service = payload.get("service", "") or ""
        return (f"Сводный прайс пакетов услуги «{service}» направления {direction}. "
                f"Содержит цены пакетов от базового до премиум, "
                f"состав, ROI-аргументы и связанные товары.")

    elif doc_type == "roadmap":
        title = payload.get("roadmap_title", "") or ""
        section = payload.get("section", "") or ""
        return (f"Регламент производства: {title}{(' — '+section) if section else ''}. "
                f"Направление {direction}. Этапы, сроки, ценовые ориентиры.")

    elif doc_type == "service_composition":
        return (f"Состав услуг направления {direction}. "
                f"Основные и опциональные компоненты, материалы, "
                f"кросс-категорийные комбинации.")

    elif doc_type == "timeline_fact":
        return (f"Срок производства {payload.get('group_type', '')} "
                f"{payload.get('group_key', '')} направления {direction}.")

    elif doc_type == "retrieval_support":
        return (f"Вспомогательный документ для retrieval. "
                f"Направление {direction}.")

    elif doc_type == "roi_anchor":
        return (f"ROI-якорь направления {direction}: пример обоснования "
                f"возврата инвестиций через результат для клиента.")

    # NULL = use LLM-generated context для следующих doc_types:
    # product, bundle, historical_deal, deal_profile, offer_profile,
    # offer_composition, knowledge, faq, photo_analysis, service_page
    return None


async def llm_context(
    client: httpx.AsyncClient,
    doc_text: str,
    sem: asyncio.Semaphore,
) -> str | None:
    """Generate LLM-based context via DeepSeek-chat."""
    # Truncate doc_text to avoid token blowup
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
        try:
            resp = await client.post(
                PROD_URL,
                json=payload,
                headers={"Host": HOST_HEADER},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            ctx = (data.get("summary") or "").strip()
            # Strip common boilerplate
            for prefix in ("Контекст для поиска:", "Контекст:", "Контекст -"):
                if ctx.lower().startswith(prefix.lower()):
                    ctx = ctx[len(prefix):].strip()
            return ctx[:500] if ctx else None
        except Exception as e:
            print(f"  LLM context error: {type(e).__name__}: {e}", flush=True)
            return None


async def process_doc(
    client: httpx.AsyncClient,
    doc: dict,
    sem: asyncio.Semaphore,
) -> dict:
    """Add contextual_text field to doc."""
    doc_type = doc.get("doc_type", "")
    payload = dict(doc.get("metadata", {}))
    payload.update(doc.get("payload", {}))
    payload["doc_type"] = doc_type

    orig_text = doc.get("searchable_text", "") or ""

    # Try template first (free)
    ctx = template_context(payload, doc_type)

    # Otherwise LLM (paid)
    if ctx is None:
        ctx = await llm_context(client, orig_text, sem)

    if ctx:
        doc["context_prefix"] = ctx
        doc["searchable_text_orig"] = orig_text
        doc["searchable_text"] = f"{ctx}\n\n{orig_text}"
    else:
        # LLM failed — keep original
        doc["context_prefix"] = ""
        doc["searchable_text_orig"] = orig_text
        # searchable_text remains orig
    return doc


async def process_file(
    in_path: Path,
    out_path: Path,
    resume: bool = False,
    limit: int | None = None,
):
    """Process one jsonl file: add context_prefix to each doc."""
    print(f"\n=== {in_path.name} ===", flush=True)

    # Load source docs
    src_docs = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                src_docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"  Loaded {len(src_docs)} docs", flush=True)

    # Resume support: skip docs already done
    processed_ids = set()
    if resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    processed_ids.add(json.loads(line).get("doc_id"))
                except json.JSONDecodeError:
                    continue
        print(f"  Resume: skipping {len(processed_ids)} already-processed", flush=True)

    todo = [d for d in src_docs if d.get("doc_id") not in processed_ids]
    if limit:
        todo = todo[:limit]
    print(f"  TODO: {len(todo)} docs", flush=True)

    if not todo:
        print(f"  All done.", flush=True)
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    template_count = 0
    llm_count = 0
    fail_count = 0
    start_ts = time.monotonic()

    out_mode = "a" if (resume and out_path.exists()) else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(verify=False) as client:
        # Process in batches to write progressively (resume-safe)
        BATCH_SIZE = 50
        with out_path.open(out_mode, encoding="utf-8") as fout:
            for batch_start in range(0, len(todo), BATCH_SIZE):
                batch = todo[batch_start:batch_start + BATCH_SIZE]
                results = await asyncio.gather(*[
                    process_doc(client, d, sem) for d in batch
                ])
                for r in results:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    if r.get("context_prefix"):
                        if any(t in (r.get("context_prefix") or "") for t in
                               ("Политика ценообразования направления",
                                "Сводный прайс пакетов услуги",
                                "Регламент производства",
                                "Состав услуг направления",
                                "Срок производства",
                                "Вспомогательный документ",
                                "ROI-якорь")):
                            template_count += 1
                        else:
                            llm_count += 1
                    else:
                        fail_count += 1
                fout.flush()

                done = batch_start + len(batch)
                rate = done / max(0.1, (time.monotonic() - start_ts))
                eta = (len(todo) - done) / max(0.01, rate)
                print(f"  [{done}/{len(todo)}] tmpl={template_count} llm={llm_count} fail={fail_count} "
                      f"rate={rate:.1f}/s eta={eta/60:.1f}min",
                      flush=True)

    elapsed = time.monotonic() - start_ts
    print(f"\n  ✓ {in_path.name}: {template_count} template + {llm_count} LLM + {fail_count} fail "
          f"in {elapsed/60:.1f} min", flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit docs per file (test)")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed doc_ids in output")
    parser.add_argument("--only", default=None, help="Process только этот jsonl file")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"=== P22.A.1 Contextual Retrieval prefix generator ===", flush=True)
    print(f"  Input:  {DATA_DIR}", flush=True)
    print(f"  Output: {OUT_DIR}", flush=True)
    print(f"  Concurrency: {CONCURRENCY}, timeout: {TIMEOUT}s", flush=True)

    # Disable SSL warnings (-k equivalent)
    import warnings
    warnings.filterwarnings("ignore")
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(DATA_DIR.glob("*.jsonl"))
    for f in jsonl_files:
        if f.name in SKIP_FILES:
            print(f"\n  Skipping {f.name} (raw source)", flush=True)
            continue
        if args.only and f.name != args.only:
            continue
        out_path = OUT_DIR / f.name
        await process_file(f, out_path, resume=args.resume, limit=args.limit)

    print(f"\n=== DONE. Output in {OUT_DIR} ===", flush=True)
    print(f"Next step: build_index.py --data-dir {OUT_DIR} --collection labus_docs_v10", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
