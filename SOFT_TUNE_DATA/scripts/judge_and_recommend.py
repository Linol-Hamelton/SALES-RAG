"""P14 Phase E: LLM-judge сравнивает каждую пару (answer_rag, answer_no_rag),
выставляет оценки и формирует recommendations.md.

Reads:
  SOFT_TUNE_DATA/answers_rag.jsonl
  SOFT_TUNE_DATA/answers_no_rag.jsonl

Writes:
  SOFT_TUNE_DATA/scores.jsonl
  SOFT_TUNE_DATA/recommendations.md

Judge model: deepseek-reasoner (via prod /query_no_rag, model_override).
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

import os
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "SOFT_TUNE_DATA"
SUFFIX = os.environ.get("RUN_SUFFIX", "")
# P16.B.2: output suffix can differ from input (e.g. rejudge v12 answers under new prompt).
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", SUFFIX)
IN_RAG = DATA_DIR / f"answers_rag{SUFFIX}.jsonl"
IN_NO_RAG = DATA_DIR / f"answers_no_rag{SUFFIX}.jsonl"
OUT_SCORES = DATA_DIR / f"scores{OUT_SUFFIX}.jsonl"
OUT_RECOMMENDATIONS = DATA_DIR / f"recommendations{OUT_SUFFIX}.md"

BASE = "https://62.217.178.117"
HOST = "ai.labus.pro"
URL = f"{BASE}/query_no_rag"
CONCURRENCY = 4
TIMEOUT = 300.0

JUDGE_SYSTEM = """Ты — эксперт-аналитик B2B-агентства полиграфии/рекламы (Лабус, Махачкала).
Твоя задача — оценить две версии ответа (RAG и no-RAG) на один и тот же вопрос
от клиента или менеджера. Цель — определить, насколько RAG-система даёт реальный
прирост качества по сравнению с чистым LLM-ответом.

Ответ ВСЕГДА — валидный JSON по запрошенной схеме. Без markdown."""


# P16.B.1: recalibrated judge prompt. Activated when env JUDGE_PROMPT_VERSION=v2.
# Main change: pricing_grounded criterion expanded — TRUE if price tied to ANY
# verifiable source (deal, catalog, bridge, knowledge price range), FALSE only
# for genuine hallucinations (no source / contradicts catalog / made-up amounts).
# Rationale: in scores_v12, ~60-65% of pricing_grounded=FALSE were false positives
# where price came from real pricelist without deal_id link.
JUDGE_SYSTEM_V2 = """Ты — эксперт-аналитик B2B-агентства полиграфии/рекламы (Лабус, Махачкала).
Оцениваешь две версии ответа (RAG и no-RAG) на один и тот же вопрос.
Цель — измерить реальный прирост качества от RAG-поиска.

КРИТИЧНО про pricing_grounded:
- TRUE если цена имеет ЛЮБОЙ верифицируемый источник:
  * упоминается конкретная сделка (#XXXX, "Сделка #62122 на 30 500 ₽")
  * цена из прайса/каталога ("кружка от 250 ₽/шт", "логотип 25-44 тыс по прайсу")
  * диапазон с указанием базиса ("медиана сделок", "по 47 заказам")
  * бридж/набор/комплект из контекста ("под ключ из набора #YYY")
  * knowledge-источник с ценовым диапазоном ("по нашему опыту 50-100 тыс")
- FALSE ТОЛЬКО если:
  * цена выдумана из воздуха, без любого источника
  * цена противоречит каталогу/прайсу
  * цена с "примерно", "ориентировочно" БЕЗ обоснования
  * нет ни конкретного числа, ни диапазона, ни ссылки на источник

ВАЖНО: цена из прайса без deal_id — это TRUE (источник есть, это каталог).
ВАЖНО: цена с диапазоном "10-25 тыс ₽" если контекст показывает медиану — TRUE.
ВАЖНО: цена в no-RAG ответе почти всегда без источника — обычно FALSE.

Ответ ВСЕГДА — валидный JSON по запрошенной схеме. Без markdown."""

JUDGE_PROMPT_TEMPLATE = """ВОПРОС ({persona}):
{query}

=== ОТВЕТ A (RAG, с поиском по корпоративным данным) ===
{rag_summary}

(Метаданные RAG: deal_items={deal_items}, refs={refs}, hist={hist}, total={total}, intent={intent})

=== ОТВЕТ B (no-RAG, чистый LLM) ===
{no_rag_summary}

=== ЗАДАЧА ===
Оцени обе версии по 4 критериям (шкала 0-10), укажи победителя и сформулируй точки роста.
ВЕРНИ ТОЛЬКО JSON:

{{
  "rag_completeness": <0-10>,
  "rag_accuracy": <0-10>,
  "rag_format_quality": <0-10>,
  "rag_pricing_grounded": <true|false>,
  "no_rag_completeness": <0-10>,
  "no_rag_accuracy": <0-10>,
  "no_rag_format_quality": <0-10>,
  "no_rag_pricing_grounded": <true|false>,
  "winner": "rag" | "no_rag" | "tie",
  "winner_reason": "<1-2 предложения>",
  "gap_dimensions": ["<точка роста 1>", "<точка роста 2>", ...]
}}

КРИТЕРИИ:
- completeness: насколько ответ покрывает все аспекты вопроса (цена, состав, сроки, уточнения)
- accuracy: корректность цифр, ссылок на сделки, правильность технологии (буквы vs баннер)
- format_quality: структура, читаемость, наличие конкретики (а не общих фраз)
- pricing_grounded: TRUE если цена имеет ЛЮБОЙ верифицируемый источник —
                    конкретная сделка #XXXX, ИЛИ позиция из прайса/каталога,
                    ИЛИ диапазон с указанием базиса ("медиана 47 сделок"),
                    ИЛИ бридж/набор из контекста, ИЛИ knowledge price range.
                    FALSE ТОЛЬКО если цена выдумана БЕЗ источника, противоречит
                    каталогу, либо "примерно X" без обоснования.
                    Цена из прайса без deal_id — это TRUE.
- gap_dimensions: что конкретно можно улучшить в RAG-ответе (если RAG проиграл или почти выиграл)
  Примеры: "не хватает ссылок на похожие сделки", "цена без диапазона", "нет упоминания технологии",
           "слишком общий — без cifr", "галлюцинация про несуществующий пакет"
"""


# P16.B.1: switch system+template based on JUDGE_PROMPT_VERSION env.
# Default (v1) keeps backward compatibility with v3-v12 score files.
# v2 enables recalibrated pricing_grounded for v13.
def _select_judge_prompts() -> tuple[str, str]:
    """Return (system, template) based on JUDGE_PROMPT_VERSION env."""
    ver = os.environ.get("JUDGE_PROMPT_VERSION", "v1")
    if ver == "v2":
        return JUDGE_SYSTEM_V2, JUDGE_PROMPT_TEMPLATE
    return JUDGE_SYSTEM, JUDGE_PROMPT_TEMPLATE


def load_pairs() -> list[tuple[dict, dict]]:
    rag_idx = {}
    for line in IN_RAG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rag_idx[r["id"]] = r
    pairs = []
    for line in IN_NO_RAG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            n = json.loads(line)
            r = rag_idx.get(n["id"])
            if r:
                pairs.append((r, n))
    return pairs


async def judge_pair(client: httpx.AsyncClient, rag: dict, no_rag: dict) -> dict:
    judge_system, judge_template = _select_judge_prompts()
    prompt = judge_template.format(
        persona=rag["persona"],
        query=rag["query"],
        rag_summary=(rag.get("summary") or "")[:2000],
        deal_items=rag.get("deal_items_count", 0),
        refs=rag.get("references_count", 0),
        hist=rag.get("historical_deals_count", 0),
        total=rag.get("total_value"),
        intent=rag.get("intent"),
        no_rag_summary=(no_rag.get("summary") or "")[:2000],
    )
    payload = {
        "query": prompt,
        "mode": "structured",
        "system_prompt_mode": "custom",
        "custom_system_prompt": judge_system,
        "model_override": "deepseek-reasoner",
        "response_format_override": "json",
        "temperature": 0.1,
        "max_tokens_override": 4096,
    }
    t0 = time.monotonic()
    try:
        resp = await client.post(URL, json=payload, headers={"Host": HOST}, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("raw_response") or "{}"
        parsed = json.loads(raw)
        out = {
            "id": rag["id"],
            "persona": rag["persona"],
            "topic": rag["topic"],
            "query": rag["query"],
            **parsed,
            "judge_latency_ms": int((time.monotonic() - t0) * 1000),
            "human_score": None,
            "human_notes": None,
        }
        return out
    except Exception as e:
        return {
            "id": rag["id"], "persona": rag["persona"], "topic": rag["topic"],
            "query": rag["query"], "error": f"{type(e).__name__}: {e}",
            "judge_latency_ms": int((time.monotonic() - t0) * 1000),
        }


async def main():
    pairs = load_pairs()
    print(f"Loaded {len(pairs)} pairs")

    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[dict] = []
    done = 0

    async with httpx.AsyncClient(verify=False) as client:
        async def worker(rag, no_rag):
            nonlocal done
            async with sem:
                r = await judge_pair(client, rag, no_rag)
                done += 1
                if done % 10 == 0 or done == len(pairs):
                    print(f"  judged {done}/{len(pairs)}", flush=True)
                return r
        results = await asyncio.gather(*[worker(r, n) for r, n in pairs])

    with OUT_SCORES.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ scores → {OUT_SCORES}")

    # Aggregation
    valid = [r for r in results if "error" not in r and r.get("winner")]
    if not valid:
        print("!! no valid scores, aborting recommendations")
        return

    win_counter = Counter(r["winner"] for r in valid)
    persona_win = defaultdict(Counter)
    topic_win = defaultdict(Counter)
    for r in valid:
        persona_win[r["persona"]][r["winner"]] += 1
        topic_win[r["topic"]][r["winner"]] += 1

    rag_avg = lambda field: sum(r.get(f"rag_{field}", 0) for r in valid) / len(valid)
    nr_avg = lambda field: sum(r.get(f"no_rag_{field}", 0) for r in valid) / len(valid)

    grounded_rag = sum(1 for r in valid if r.get("rag_pricing_grounded")) / max(1, len(valid))
    grounded_nr = sum(1 for r in valid if r.get("no_rag_pricing_grounded")) / max(1, len(valid))

    # Gap dimensions
    all_gaps = []
    for r in valid:
        for g in (r.get("gap_dimensions") or [])[:5]:
            all_gaps.append(g)
    gap_counter = Counter(all_gaps)

    # Top fails (RAG lost)
    rag_fails = [r for r in valid if r.get("winner") == "no_rag"]
    rag_fails_sorted = sorted(rag_fails, key=lambda r: (
        (r.get("no_rag_completeness", 0) - r.get("rag_completeness", 0))
        + (r.get("no_rag_accuracy", 0) - r.get("rag_accuracy", 0))
    ), reverse=True)[:10]

    # Top wins (RAG won decisively)
    rag_wins = [r for r in valid if r.get("winner") == "rag"]
    rag_wins_sorted = sorted(rag_wins, key=lambda r: (
        (r.get("rag_completeness", 0) - r.get("no_rag_completeness", 0))
        + (r.get("rag_accuracy", 0) - r.get("no_rag_accuracy", 0))
    ), reverse=True)[:10]

    # Hallucination cases (rag NOT grounded)
    hallu = [r for r in valid if r.get("rag_pricing_grounded") is False]

    md = []
    md.append("# P14 Phase E — Soft-tune Recommendations Report\n")
    md.append(f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"Всего пар: **{len(valid)}** валидных (из {len(results)} попыток)\n")

    md.append("## Aggregate Win-rate\n")
    md.append(f"- **RAG wins**: {win_counter['rag']} ({100*win_counter['rag']/len(valid):.1f}%)")
    md.append(f"- **no-RAG wins**: {win_counter['no_rag']} ({100*win_counter['no_rag']/len(valid):.1f}%)")
    md.append(f"- **Tie**: {win_counter['tie']} ({100*win_counter['tie']/len(valid):.1f}%)\n")

    md.append("## Average Scores (0-10)\n")
    md.append("| Critère | RAG | no-RAG | Δ |")
    md.append("|---|---|---|---|")
    for field in ("completeness", "accuracy", "format_quality"):
        r_v = rag_avg(field)
        n_v = nr_avg(field)
        md.append(f"| {field} | {r_v:.2f} | {n_v:.2f} | **{r_v - n_v:+.2f}** |")
    md.append(f"| pricing_grounded | {100*grounded_rag:.1f}% | {100*grounded_nr:.1f}% | **{100*(grounded_rag-grounded_nr):+.1f}pp** |\n")

    md.append("## Win-rate by Persona\n")
    md.append("| Persona | RAG | no-RAG | Tie |")
    md.append("|---|---|---|---|")
    for p, c in persona_win.items():
        md.append(f"| {p} | {c['rag']} | {c['no_rag']} | {c['tie']} |")
    md.append("")

    md.append("## Win-rate by Topic (top 10 worst для RAG)\n")
    topic_summary = []
    for t, c in topic_win.items():
        total = c['rag'] + c['no_rag'] + c['tie']
        rag_pct = 100 * c['rag'] / total if total else 0
        topic_summary.append((t, c, rag_pct, total))
    topic_summary.sort(key=lambda x: x[2])  # worst RAG % first
    md.append("| Topic | RAG | no-RAG | Tie | RAG win% |")
    md.append("|---|---|---|---|---|")
    for t, c, rag_pct, total in topic_summary[:10]:
        md.append(f"| {t} | {c['rag']} | {c['no_rag']} | {c['tie']} | {rag_pct:.0f}% |")
    md.append("")

    md.append("## Top-10 Gap Dimensions (часто упоминаемые точки роста)\n")
    for gap, n in gap_counter.most_common(15):
        md.append(f"- ({n}×) {gap}")
    md.append("")

    md.append("## Top-10 RAG-Fails (где no-RAG явно лучше)\n")
    for i, r in enumerate(rag_fails_sorted, 1):
        md.append(f"### Fail #{i} — `{r['id']}` ({r['topic']})")
        md.append(f"**Q:** {r['query'][:200]}")
        md.append(f"**Reason:** {r.get('winner_reason', '')}")
        md.append(f"**Gaps:** {', '.join(r.get('gap_dimensions') or [])[:300]}")
        md.append("")

    md.append("## Top-10 RAG-Wins (где RAG бьёт baseline)\n")
    for i, r in enumerate(rag_wins_sorted, 1):
        md.append(f"### Win #{i} — `{r['id']}` ({r['topic']})")
        md.append(f"**Q:** {r['query'][:200]}")
        md.append(f"**Reason:** {r.get('winner_reason', '')}")
        md.append("")

    if hallu:
        md.append(f"## Pricing Hallucinations (RAG выдал необоснованную цену) — {len(hallu)} случаев\n")
        for r in hallu[:10]:
            md.append(f"- `{r['id']}` ({r['topic']}): {r['query'][:120]}")
        md.append("")

    md.append("## Actionable Soft-tune Recommendations\n")
    md.append("На основе агрегатов выше — конкретные правки (заполняется вручную после ревью):\n")
    md.append("1. **prompts.yaml**: ...")
    md.append("2. **intent_classifier**: ...")
    md.append("3. **retrieval strategies (retriever.py)**: ...")
    md.append("4. **dialog_state gates (dialog_state.py)**: ...")
    md.append("5. **few-shot examples (Phase F-alt-1)**: топ-20 RAG-wins из таблицы выше → инжектить в `human_query` / `structured_query` промпт.")

    OUT_RECOMMENDATIONS.write_text("\n".join(md), encoding="utf-8")
    print(f"✓ recommendations → {OUT_RECOMMENDATIONS}")


if __name__ == "__main__":
    import warnings, urllib3
    warnings.filterwarnings("ignore")
    urllib3.disable_warnings()
    asyncio.run(main())
