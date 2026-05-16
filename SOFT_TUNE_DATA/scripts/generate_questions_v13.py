"""P16.C.1+C.2: data-informed v13 question generation (target 1600 questions).

Differences from generate_questions.py:
  1. Expanded targets: 800 client + 800 manager = 1600 (vs SCALE=4 → 400+400).
  2. Three NEW topics:
     - CLIENT: portfolio_examples (backing on photo_analysis_docs)
     - MANAGER: project_timeline (backing on roadmap_docs + timeline_docs)
     - MANAGER: service_bundling (backing on bundle_docs + service_composition)
  3. For new topics: each question carries `backing_doc_ids` derived from
     a randomly-sampled real doc_id pulled from RAG_RUNTIME/data/*.jsonl.
     This enables a retrieval-recall smoke test post-generation.

Output:
  SOFT_TUNE_DATA/questions_client_v13.jsonl  (~800 rows)
  SOFT_TUNE_DATA/questions_manager_v13.jsonl (~800 rows)

Usage:
  PYTHONIOENCODING=utf-8 python SOFT_TUNE_DATA/scripts/generate_questions_v13.py
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import warnings
from pathlib import Path

import httpx

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "SOFT_TUNE_DATA"
DATA_DIR = ROOT / "RAG_RUNTIME" / "data"

OUT_CLIENT = OUT_DIR / "questions_client_v13.jsonl"
OUT_MANAGER = OUT_DIR / "questions_manager_v13.jsonl"

PROD_URL = "https://62.217.178.117/query_no_rag"
HOST_HEADER = "ai.labus.pro"

SEED = 42

# Per-topic target counts (sum should be ~800 each)
CLIENT_TARGETS = {
    "design_logo": 65,
    "design_brand": 65,
    "ad_signboard": 65,
    "ad_banner": 65,
    "ad_misc": 60,
    "print_visitka": 65,
    "print_flyer": 65,
    "print_misc": 60,
    "merch": 60,
    "stickers": 60,
    "consultation": 65,
    "objections": 65,
    "portfolio_examples": 40,  # NEW data-informed
}
# total: 65*7 + 60*4 + 65*1 + 40 = 455 + 240 + 65 + 40 = 800

MANAGER_TARGETS = {
    "objection_handling": 75,
    "smeta_assist": 75,
    "historical_links": 75,
    "upsell": 70,
    "discovery": 70,
    "signage_specs": 70,
    "vocabulary": 65,
    "directions": 70,
    "pricing_justify": 70,
    "logistics": 60,
    "project_timeline": 50,    # NEW data-informed
    "service_bundling": 50,    # NEW data-informed
}
# total: 75*3 + 70*5 + 65 + 60 + 50*2 = 225 + 350 + 65 + 60 + 100 = 800


CLIENT_TOPICS_LABELS = {
    "design_logo": "дизайн / логотип",
    "design_brand": "дизайн / брендбук + фирменный стиль",
    "ad_signboard": "реклама / вывеска (объёмные буквы / световой короб)",
    "ad_banner": "реклама / баннер + брендмауэр",
    "ad_misc": "реклама / штендер + табличка",
    "print_visitka": "полиграфия / визитки",
    "print_flyer": "полиграфия / листовки + буклеты",
    "print_misc": "полиграфия / меню + стенды",
    "merch": "сувенирка / футболки + кружки + бейсболки",
    "stickers": "наклейки + плёнка",
    "consultation": "консультация / общие вопросы по компании",
    "objections": "возражения / цена / сроки / гарантия",
    "portfolio_examples": "примеры работ / портфолио",
}

MANAGER_TOPICS_LABELS = {
    "objection_handling": "как ответить клиенту с возражением 'у конкурентов дешевле'",
    "smeta_assist": "осмечивание под параметры клиента (high / low spec)",
    "historical_links": "ссылки на похожие закрытые сделки из истории",
    "upsell": "upsell-опции к текущему запросу клиента",
    "discovery": "когда не давать цену сразу — discovery-вопросы",
    "signage_specs": "вывески с подсветкой — производственные параметры",
    "vocabulary": "брендмауэр vs объёмные буквы — где какая технология",
    "directions": "сводка по бизнес-направлениям Лабус (Цех/Печать/Дизайн/РИК/Мерч)",
    "pricing_justify": "как обосновать цену через ROI / результат для клиента",
    "logistics": "логистика / монтаж / выезд / автовышка",
    "project_timeline": "графики, вехи и сроки проектов",
    "service_bundling": "комплектация: какие услуги идут вместе",
}

# Topics that should be data-backed (samples real doc_ids as context for generation)
DATA_BACKED_TOPICS = {
    "portfolio_examples": ("photo_analysis_docs.jsonl", "photo_analysis"),
    "project_timeline": ("roadmap_docs.jsonl", "roadmap"),
    "service_bundling": ("bundle_docs.jsonl", "bundle"),
}


CLIENT_SYSTEM = """Ты симулируешь поток входящих запросов от клиентов B2B-агентства полиграфии,
рекламы, дизайна и сувенирной продукции в Махачкале (Россия). Клиенты — это
владельцы малого/среднего бизнеса (рестораны, школы, клиники, магазины,
строительные компании, частные лица). Стилистика — живая, разговорная,
русскоязычная, с типичными ошибками и сокращениями. Запрос может быть как
коротким («сколько стоят 1000 визиток?»), так и развёрнутым.

ТРЕБОВАНИЯ:
- Каждый вопрос реалистичен и отражает реальный путь клиента к покупке.
- Не повторяй формулировки — diversity критична.
- Включай разные стадии воронки: первое касание, уточнения, возражения, согласование.
- Используй разные доли спецификации: от полного underspec до пере-спецификации.
- НЕ задавай вопросы про API/код/техническую интеграцию — только бизнес-сценарии.

Отвечай ТОЛЬКО валидным JSON по запрошенной структуре."""

MANAGER_SYSTEM = """Ты симулируешь поток вопросов от менеджеров по продажам внутри
B2B-агентства полиграфии, рекламы, дизайна и сувенирной продукции в Махачкале.
Менеджеры пользуются RAG-системой для подготовки ответов клиенту, осмечивания
и поиска похожих сделок. Стилистика — деловая, конкретная, профессиональная.

ТРЕБОВАНИЯ:
- Каждый вопрос отражает реальную задачу менеджера: ответить на возражение,
  осметить под параметры, найти аналог, прикинуть upsell.
- Менеджеры часто пишут от первого лица: «как мне ответить если клиент...»,
  «дай ссылки на сделки...», «осметь под клиента следующее: ...».
- Не повторяй формулировки.
- НЕ задавай вопросы про API/код — только sales workflow.

Отвечай ТОЛЬКО валидным JSON по запрошенной структуре."""


def sample_backing_docs(topic_key: str, n_samples: int = 5) -> list[dict]:
    """For data-backed topics, sample real docs from RAG_RUNTIME/data to use as context."""
    if topic_key not in DATA_BACKED_TOPICS:
        return []
    file_name, _ = DATA_BACKED_TOPICS[topic_key]
    path = DATA_DIR / file_name
    if not path.exists():
        return []
    rng = random.Random(SEED + hash(topic_key) % 1000)
    samples = []
    with path.open(encoding="utf-8") as f:
        # Reservoir sampling — read whole file once
        all_lines = [l for l in f if l.strip()]
    if not all_lines:
        return []
    chosen = rng.sample(all_lines, min(n_samples, len(all_lines)))
    for line in chosen:
        try:
            d = json.loads(line)
            text = (d.get("searchable_text") or "")[:300]
            samples.append({
                "doc_id": d.get("doc_id", ""),
                "snippet": text,
            })
        except Exception:
            continue
    return samples


async def generate_for_topic(
    client: httpx.AsyncClient,
    persona: str,
    topic_label: str,
    topic_key: str,
    n: int,
    sys_prompt: str,
) -> list[dict]:
    # Build user prompt with optional data backing
    extra_context = ""
    backing_doc_ids: list[str] = []
    if topic_key in DATA_BACKED_TOPICS:
        backing = sample_backing_docs(topic_key, n_samples=5)
        if backing:
            backing_doc_ids = [b["doc_id"] for b in backing]
            context_lines = [f"- {b['doc_id']}: {b['snippet'][:200]}" for b in backing]
            extra_context = (
                "\n\nКОНТЕКСТ — реальные документы из базы (используй как inspiration "
                "для генерации правдоподобных вопросов):\n" + "\n".join(context_lines)
            )

    user_prompt = f"""Сгенерируй РОВНО {n} разнообразных вопросов по теме:
**«{topic_label}»** (persona: {persona}).{extra_context}

Формат ответа — JSON-объект:
{{
  "questions": [
    {{"question": "...", "expected_intent": "product_query|consultation|historical_request|persuasion|underspec|general"}},
    ...
  ]
}}

ВАЖНО:
- РОВНО {n} элементов в массиве.
- Каждый question — отдельная строка, без markdown.
- expected_intent выбери одно значение для каждого вопроса.
- Diversity: разные форматы (короткий/длинный), разные стадии воронки."""

    payload = {
        "query": user_prompt,
        "mode": "structured",
        "system_prompt_mode": "custom",
        "custom_system_prompt": sys_prompt,
        "model_override": "deepseek-reasoner",
        "response_format_override": "json",
        "temperature": 0.9,
        "max_tokens_override": 8000,
    }
    try:
        resp = await client.post(PROD_URL, json=payload, headers={"Host": HOST_HEADER}, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("raw_response") or "{}"
        parsed = json.loads(raw)
        items = parsed.get("questions") or []
        rows: list[dict] = []
        for i, item in enumerate(items[:n], 1):
            q = (item.get("question") or "").strip()
            if not q:
                continue
            row = {
                "id": f"{persona[0]}_{topic_key}_{i:03d}",
                "persona": persona,
                "topic": topic_key,
                "topic_label": topic_label,
                "question": q,
                "expected_intent": item.get("expected_intent") or "general",
            }
            if backing_doc_ids:
                row["backing_doc_ids"] = backing_doc_ids
            rows.append(row)
        print(f"  [{persona}/{topic_key}] generated {len(rows)} (asked {n})", flush=True)
        return rows
    except Exception as e:
        print(f"  [{persona}/{topic_key}] FAILED: {type(e).__name__}: {e}", flush=True)
        return []


async def main():
    target_client = sum(CLIENT_TARGETS.values())
    target_manager = sum(MANAGER_TARGETS.values())
    print(f"=== TARGETS: client={target_client}, manager={target_manager}, total={target_client+target_manager} ===", flush=True)

    async with httpx.AsyncClient(verify=False) as client:
        client_tasks = []
        for key, n in CLIENT_TARGETS.items():
            label = CLIENT_TOPICS_LABELS[key]
            client_tasks.append(generate_for_topic(client, "client", label, key, n, CLIENT_SYSTEM))

        manager_tasks = []
        for key, n in MANAGER_TARGETS.items():
            label = MANAGER_TOPICS_LABELS[key]
            manager_tasks.append(generate_for_topic(client, "manager", label, key, n, MANAGER_SYSTEM))

        print(f"\n=== Generating CLIENT ({len(client_tasks)} topics) ===", flush=True)
        client_results = await asyncio.gather(*client_tasks)
        client_rows = [row for batch in client_results for row in batch]

        print(f"\n=== Generating MANAGER ({len(manager_tasks)} topics) ===", flush=True)
        manager_results = await asyncio.gather(*manager_tasks)
        manager_rows = [row for batch in manager_results for row in batch]

    with OUT_CLIENT.open("w", encoding="utf-8") as f:
        for row in client_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with OUT_MANAGER.open("w", encoding="utf-8") as f:
        for row in manager_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    backed_client = sum(1 for r in client_rows if r.get("backing_doc_ids"))
    backed_manager = sum(1 for r in manager_rows if r.get("backing_doc_ids"))
    print(f"\n✓ wrote {len(client_rows)} client questions → {OUT_CLIENT}")
    print(f"  data-backed: {backed_client}")
    print(f"✓ wrote {len(manager_rows)} manager questions → {OUT_MANAGER}")
    print(f"  data-backed: {backed_manager}")
    print(f"\nTotal: {len(client_rows) + len(manager_rows)} (target 1600)")


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    asyncio.run(main())
