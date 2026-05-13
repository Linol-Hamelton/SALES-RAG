"""P14 Phase C: generate 100 client + 100 manager questions via prod /query_no_rag.

Uses prod endpoint (https://62.217.178.117/query_no_rag, Host: ai.labus.pro) so
that no local DEEPSEEK_API_KEY is required — server's key is reused.
Model is overridden to `deepseek-reasoner` per request for higher diversity/quality.

Output:
  SOFT_TUNE_DATA/questions_client.jsonl
  SOFT_TUNE_DATA/questions_manager.jsonl

Each line: {"id", "persona", "question", "topic", "expected_intent"}

Usage:
  python SOFT_TUNE_DATA/scripts/generate_questions.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "SOFT_TUNE_DATA"
OUT_DIR.mkdir(parents=True, exist_ok=True)
import os
SCALE = int(os.environ.get("QGEN_SCALE", "1"))  # 1=100/100, 2=200/200, 4=400/400, etc.
# RUN_SUFFIX overrides SCALE-based naming when explicitly set (e.g. "_v4")
SUFFIX = os.environ.get("RUN_SUFFIX") or ("_v2" if SCALE >= 2 else "")

OUT_CLIENT = OUT_DIR / f"questions_client{SUFFIX}.jsonl"
OUT_MANAGER = OUT_DIR / f"questions_manager{SUFFIX}.jsonl"

PROD_URL = "https://62.217.178.117/query_no_rag"
HOST_HEADER = "ai.labus.pro"

CLIENT_TOPICS = [
    ("дизайн / логотип", "design_logo"),
    ("дизайн / брендбук + фирменный стиль", "design_brand"),
    ("реклама / вывеска (объёмные буквы / световой короб)", "ad_signboard"),
    ("реклама / баннер + брендмауэр", "ad_banner"),
    ("реклама / штендер + табличка", "ad_misc"),
    ("полиграфия / визитки", "print_visitka"),
    ("полиграфия / листовки + буклеты", "print_flyer"),
    ("полиграфия / меню + стенды", "print_misc"),
    ("сувенирка / футболки + кружки + бейсболки", "merch"),
    ("наклейки + плёнка", "stickers"),
    ("консультация / общие вопросы по компании", "consultation"),
    ("возражения / цена / сроки / гарантия", "objections"),
]
MANAGER_TOPICS = [
    ("как ответить клиенту с возражением 'у конкурентов дешевле'", "objection_handling"),
    ("осмечивание под параметры клиента (high / low spec)", "smeta_assist"),
    ("ссылки на похожие закрытые сделки из истории", "historical_links"),
    ("upsell-опции к текущему запросу клиента", "upsell"),
    ("когда не давать цену сразу — discovery-вопросы", "discovery"),
    ("вывески с подсветкой — производственные параметры", "signage_specs"),
    ("брендмауэр vs объёмные буквы — где какая технология", "vocabulary"),
    ("сводка по бизнес-направлениям Лабус (Цех/Печать/Дизайн/РИК/Мерч)", "directions"),
    ("как обосновать цену через ROI / результат для клиента", "pricing_justify"),
    ("логистика / монтаж / выезд / автовышка", "logistics"),
]


CLIENT_SYSTEM = """Ты симулируешь поток входящих запросов от клиентов B2B-агентства полиграфии,
рекламы, дизайна и сувенирной продукции в Махачкале (Россия). Клиенты — это
владельцы малого/среднего бизнеса (рестораны, школы, клиники, магазины,
строительные компании, частные лица). Стилистика — живая, разговорная,
русскоязычная, с типичными ошибками и сокращениями. Запрос может быть как
коротким («сколько стоят 1000 визиток?»), так и развёрнутым («нужна вывеска
для нового ресторана ГОГОЛЬ-МОГОЛЬ, буквы 50 см, подсветка...»).

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


async def generate_for_topic(
    client: httpx.AsyncClient,
    persona: str,
    topic_label: str,
    topic_key: str,
    n: int,
    sys_prompt: str,
) -> list[dict]:
    user_prompt = f"""Сгенерируй РОВНО {n} разнообразных вопросов по теме:
**«{topic_label}»** (persona: {persona}).

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
            rows.append({
                "id": f"{persona[0]}_{topic_key}_{i:03d}",
                "persona": persona,
                "topic": topic_key,
                "topic_label": topic_label,
                "question": q,
                "expected_intent": item.get("expected_intent") or "general",
            })
        print(f"  [{persona}/{topic_key}] generated {len(rows)} (asked {n}) — model={data.get('model')}", flush=True)
        return rows
    except Exception as e:
        print(f"  [{persona}/{topic_key}] FAILED: {type(e).__name__}: {e}", flush=True)
        return []


async def main():
    target_client = 100 * SCALE
    target_manager = 100 * SCALE
    async with httpx.AsyncClient(verify=False) as client:
        per_client = target_client // len(CLIENT_TOPICS)
        extra_client = target_client - per_client * len(CLIENT_TOPICS)
        print(f"=== CLIENT: {per_client}×{len(CLIENT_TOPICS)} + {extra_client} = {target_client} ===", flush=True)
        client_tasks = []
        for i, (label, key) in enumerate(CLIENT_TOPICS):
            n = per_client + (1 if i < extra_client else 0)
            client_tasks.append(generate_for_topic(client, "client", label, key, n, CLIENT_SYSTEM))
        client_results = await asyncio.gather(*client_tasks)
        client_rows = [row for batch in client_results for row in batch][:target_client]

        per_manager = target_manager // len(MANAGER_TOPICS)
        extra_manager = target_manager - per_manager * len(MANAGER_TOPICS)
        print(f"\n=== MANAGER: {per_manager}×{len(MANAGER_TOPICS)} + {extra_manager} = {target_manager} ===", flush=True)
        manager_tasks = []
        for i, (label, key) in enumerate(MANAGER_TOPICS):
            n = per_manager + (1 if i < extra_manager else 0)
            manager_tasks.append(generate_for_topic(client, "manager", label, key, n, MANAGER_SYSTEM))
        manager_results = await asyncio.gather(*manager_tasks)
        manager_rows = [row for batch in manager_results for row in batch][:target_manager]

    with OUT_CLIENT.open("w", encoding="utf-8") as f:
        for row in client_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with OUT_MANAGER.open("w", encoding="utf-8") as f:
        for row in manager_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✓ wrote {len(client_rows)} client questions → {OUT_CLIENT}")
    print(f"✓ wrote {len(manager_rows)} manager questions → {OUT_MANAGER}")


if __name__ == "__main__":
    # Silence urllib3 SSL warnings (we use --no-verify-ssl analog)
    import warnings, urllib3
    warnings.filterwarnings("ignore")
    urllib3.disable_warnings()
    asyncio.run(main())
