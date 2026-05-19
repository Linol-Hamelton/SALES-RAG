"""P22.B.4: 200 hard test cases targeting v18 failure modes.

Failure mode targets (from v18 analysis of 380 no_rag wins):
  - 57% (~50 cases): Missing/vague pricing extraction → "Точная цена X для Y"
  - 23% (~50 cases): Vague generic output → "Покажи сделку #XXXXX"
  - 10% (~50 cases): Task-specific (письмо, аргументы, чек-лист)
  - 5% (~50 cases): Multi-faceted (multiple criteria in one query)

Output:
  SOFT_TUNE_DATA/questions_v22_hard.jsonl (~200 rows)

Usage:
  PYTHONIOENCODING=utf-8 python SOFT_TUNE_DATA/scripts/generate_questions_v22_hard.py
"""
from __future__ import annotations

import asyncio
import json
import warnings
from pathlib import Path

import httpx

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "SOFT_TUNE_DATA"
OUT_FILE = OUT_DIR / "questions_v22_hard.jsonl"

PROD_URL = "https://62.217.178.117/query_no_rag"
HOST_HEADER = "ai.labus.pro"

# === Hard-case targets (50 each → 200 total) ===

HARD_TARGETS = {
    "explicit_pricing": {
        "n": 50,
        "label": "Точные цены под параметры — НЕ диапазон, а конкретное число",
        "patterns": [
            "Сколько ТОЧНО стоит 1000 визиток 350 г/м² с двусторонней печатью + ламинацией?",
            "Точная цена логотипа пакет «Стандарт» — без 'от X до Y', нужна конкретная сумма",
            "Цена объёмных букв высотой 60 см, 12 букв, с подсветкой — итого под ключ",
            "Сколько стоит 500 буклетов А5 2-fold на мелованной 130 г/м² с УФ-лаком?",
            "Точная цена монтажа вывески 5 м² на фасад с автовышкой",
        ],
        "must_contain_in_answer": ["конкретная цена", "источник (deal_id / пакет)", "lead_time"],
    },
    "deal_id_refs": {
        "n": 50,
        "label": "Запросы на конкретные сделки из истории — нужны deal_id с ссылками",
        "patterns": [
            "Покажи 3 сделки по брендированию авто — нужны ID и итоговые суммы",
            "Дай ссылки на сделки с вывесками для ресторанов 2024-2025 гг.",
            "Какая самая крупная сделка по брендбукам за последний год?",
            "Найди сделку где клиент торговался и в итоге купил",
            "Похожие закрытые сделки по объёмным буквам высотой 50-80 см",
        ],
        "must_contain_in_answer": ["≥2 deal_id", "Bitrix URL", "цена сделки", "год"],
    },
    "task_specific": {
        "n": 50,
        "label": "Менеджерские задачи: письма, аргументы, чек-листы — НЕ цены",
        "patterns": [
            "Составь письмо клиенту с КП на 100 000 ₽, мягко закрывая возражение «дорого»",
            "Дай 5 аргументов почему наш брендбук стоит 120к, а не 40к как у фрилансера",
            "Чек-лист уточняющих вопросов перед осметом вывески под ключ",
            "Скрипт ответа на возражение «у конкурентов в 2 раза дешевле»",
            "Шаблон отчёта по проекту для согласования с клиентом",
        ],
        "must_contain_in_answer": ["структурированный output (письмо/чек-лист/скрипт)",
                                   "конкретный текст, не цены"],
    },
    "multi_faceted": {
        "n": 50,
        "label": "Сложные запросы с многими критериями одновременно",
        "patterns": [
            "Логотип для кафе быстро (за 3 дня) + брендбук + визитки на 1000 шт — итого под ключ",
            "Вывеска на 3 фасада здания: объёмные буквы 50 см + лайтбокс 2 м² + дизайн, срочно",
            "Полное брендирование автопарка 8 машин (дизайн + резка плёнки + монтаж) — за 2 недели",
            "Мерч под мероприятие: 200 футболок + 100 кружек + 500 ручек с одним логотипом",
            "Каталог продукции 32 полосы А4 тираж 500 + 1000 буклетов-вкладышей А5",
        ],
        "must_contain_in_answer": ["разбивка по компонентам с ценами", "общий итог", "lead_time"],
    },
}

SYSTEM_PROMPT = """Ты симулируешь поток ОЧЕНЬ КОНКРЕТНЫХ запросов к RAG-системе
B2B-агентства полиграфии/рекламы/дизайна Лабус (Махачкала). ТВОЯ ЦЕЛЬ — создать
HARD CASES которые ломают наивные generic ответы. Каждый вопрос должен ТРЕБОВАТЬ
из системы конкретики: либо точную цену с источником (deal_id), либо ссылки на
реальные сделки, либо структурированный output (письмо/чек-лист/скрипт), либо
комплексный multi-component расчёт.

ТРЕБОВАНИЯ:
- НЕ генерируй простые вопросы типа «сколько стоит логотип?» — это softball.
- Каждый вопрос требует конкретики (точная цена, конкретный deal_id, структура output).
- Реалистично — как реально пишут клиенты/менеджеры в проде.
- Используй паттерны-примеры из контекста, но ВАРЬИРУЙ детали (тиражи, размеры, бизнес).

Отвечай ТОЛЬКО валидным JSON по запрошенной структуре."""


async def generate_for_category(
    client: httpx.AsyncClient,
    cat_key: str,
    cat_spec: dict,
) -> list[dict]:
    n = cat_spec["n"]
    label = cat_spec["label"]
    patterns = cat_spec["patterns"]
    must_contain = cat_spec["must_contain_in_answer"]

    pattern_block = "\n".join(f"  - {p}" for p in patterns)
    must_block = "\n".join(f"  - {m}" for m in must_contain)

    user_prompt = f"""Сгенерируй РОВНО {n} разнообразных HARD-CASE вопросов по категории:

**«{label}»**

ПРИМЕРЫ ПАТТЕРНОВ (варьируй детали — тираж, размер, бизнес, материалы):
{pattern_block}

ОТВЕТ СИСТЕМЫ на эти вопросы ДОЛЖЕН содержать:
{must_block}

Формат ответа — JSON:
{{
  "questions": [
    {{"question": "...", "expected_intent": "product_query|smeta_request|historical_request|objection_arguments|describe|consultation"}},
    ...
  ]
}}

ВАЖНО:
- РОВНО {n} элементов в массиве.
- Каждый question — конкретный, требует структурированного output (не вода).
- Diversity критична: варьируй тиражи (100/500/1000/5000), размеры, бизнес-сферы.
- Реалистичный язык, можно сокращения и опечатки."""

    payload = {
        "query": user_prompt,
        "mode": "structured",
        "system_prompt_mode": "custom",
        "custom_system_prompt": SYSTEM_PROMPT,
        "model_override": "deepseek-reasoner",
        "response_format_override": "json",
        "temperature": 0.95,
        "max_tokens_override": 8000,
    }
    try:
        resp = await client.post(
            PROD_URL, json=payload, headers={"Host": HOST_HEADER}, timeout=300.0
        )
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
                "id": f"hard_{cat_key}_{i:03d}",
                "persona": "client" if cat_key in ("explicit_pricing", "multi_faceted") else "manager",
                "topic": cat_key,
                "topic_label": label,
                "question": q,
                "expected_intent": item.get("expected_intent") or "general",
                "hard_category": cat_key,
            }
            rows.append(row)
        print(f"  [{cat_key}] generated {len(rows)} (asked {n})", flush=True)
        return rows
    except Exception as e:
        print(f"  [{cat_key}] FAILED: {type(e).__name__}: {e}", flush=True)
        return []


async def main():
    target = sum(cat["n"] for cat in HARD_TARGETS.values())
    print(f"=== P22.B.4 HARD CASES: target={target} ===", flush=True)

    OUT_DIR.mkdir(exist_ok=True)

    async with httpx.AsyncClient(verify=False) as client:
        tasks = [
            generate_for_category(client, cat_key, cat_spec)
            for cat_key, cat_spec in HARD_TARGETS.items()
        ]
        results = await asyncio.gather(*tasks)

    rows = [r for batch in results for r in batch]
    print(f"\nTotal generated: {len(rows)} / target {target}")

    OUT_FILE.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_FILE} ({OUT_FILE.stat().st_size:,} bytes)")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
