"""Analyze feedback replay results.

Input:  replays.json (from replay.py)
Output: analysis.json, report.md

Scoring strategy:
  1. Per-pair diff: price, confidence, flags, summary length
  2. Manual topic taxonomy (keyword-based)
  3. Critique taxonomy (8 types) — heuristic check if new response addresses it
  4. Per-topic cross-comparison: aggregate how the model behaves on same topic
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent

TOPICS = {
    "outdoor_install":   [r"монтаж\s+наружн", r"монтаж\s+вывеск", r"монтаж\s+баннер"],
    "printing_leaflets": [r"листовк", r"тираж", r"брошюр", r"флаер", r"буклет"],
    "lit_sign_named":    [r"вывеск.*шаурм", r"вывеск.*кофейн", r"вывеск.*куруш",
                          r"световая\s+вывеск", r"объ[её]мн\w+\s+букв"],
    "brendbook_logo":    [r"брендбук", r"фирменн\w+\s+стил", r"логотип", r"брендинг"],
    "manager_script":    [r"первого\s+ответа\s+клиент", r"что\s+ему\s+ответить",
                          r"скрипт", r"что\s+мне\s+ответить", r"ответить\s+клиент",
                          r"клиент\s+спрашивает", r"клиент\s+ответил"],
    "smeta_general":     [r"дай\s+смету", r"напиши\s+смету", r"осмечивание",
                          r"смет[аы]\s+для", r"составь\s+смету"],
    "banner_pricing":    [r"\bбаннер\b(?!\s*на)", r"баннер\s+\d+\s*[xх]\s*\d+"],
    "full_package":      [r"полн\w+\s+упаковк", r"под\s+ключ", r"пакет\s+услуг",
                          r"комплексн\w+\s+брендинг"],
    "photo_followup":    [r"вместо\s+этой", r"вместо\s+этого", r"хочу\s+повесить"],
    "panel_bracket":     [r"панел.*кронштейн", r"кронштейн.*панел"],
}

# Critique taxonomy — each entry: name, trigger regex (in expert comment), check fn
CRITIQUE_TYPES = [
    ("needs_clarification",    r"уточни|уточнить|необходимо\s+уточн"),
    ("wrong_catalog",          r"goods\.csv|offers\.csv|orders\.csv|артикул\s*\d+|labus\.pro"),
    ("wrong_math",             r"сумма\s+(невероятно\s+)?завышен|реальная\s+стоимост|медиана\s+(услуг|сделок)|ориентир"),
    ("missing_refs",           r"примеров\s+из\s+реальных\s+сделок|ссылк\w+\s+на\s+страниц|3[-–]5\s+примеров"),
    ("forbidden_phrase",       r"перестань\s+использовать|не\s+использовать|форбидден"),
    ("workflow_guidance",      r"брифы\s+и\s+скрипты|дорожн\w+\s+карт|менеджер\w+\s+скрипт"),
    ("brand_voice",            r"рыночн\w+\s+данн\w+\s+по\s+дагестан|по\s+рынку\s+дагестан"),
    ("regression",             r"деградаци|было\s+лучше|предыдущие\s+ответы\s+были\s+точн"),
    ("cites_roadmap",          r"регламент|этап\w*\s+работ|процесс\s+создани|как\s+(вы\s+)?делает|порядок\s+работ"),
    # P12.2.1 — new critique types (from prior cluster-analysis)
    ("parametric_mismatch",    r"тираж|экземпляр|\d+\s*(м|см)\s*[xх×]\s*\d+|высот[ае]\s+\d|формат\s*a\d|размер\s+\d"),
    ("manager_script_intent",  r"как\s+(мне\s+)?ответить|что\s+(мне|ему)\s+ответить|скрипт\s+(для|переговор)|ответ(ить)?\s+клиент|клиент\s+спрашивает|клиент\s+говорит|возражен"),
    ("brand_tier_miscalc",     r"логотип\s+(за|стоит|должен)\s+\d|брендбук\s+(должен|стоит|за)\s+\d|тир\s+(логотип|бренд)|калибров\w+\s+(цен|тир)"),
    ("missing_article_link",   r"goods\.csv.*id|артикул\s+\d|labus\.pro/product|ссылк\w+\s+на\s+товар|url\s+товара"),
    ("missing_deal_example",   r"пример\w*\s+сделок|показать\s+заказ|реальн\w+\s+сделок\w*|id\s+сделк|примеры\s+\d"),
]


def detect_topic(query: str) -> str:
    q = query.lower()
    for topic, patterns in TOPICS.items():
        if any(re.search(p, q, re.IGNORECASE) for p in patterns):
            return topic
    return "other"


def detect_critique_types(comment: str) -> list[str]:
    c = comment.lower()
    hits = []
    for name, pattern in CRITIQUE_TYPES:
        if re.search(pattern, c, re.IGNORECASE):
            hits.append(name)
    return hits


def addresses(critique: str, new: dict, comment: str) -> str:
    """Return 'yes' | 'partial' | 'no' | 'n/a' — does the new response address this critique."""
    summary = (new.get("key_fields", {}).get("summary") or "").lower()
    flags = [f.lower() for f in new.get("key_fields", {}).get("flags", [])]
    reasoning = (new.get("key_fields", {}).get("reasoning") or "").lower()
    full_text = summary + " " + " ".join(flags) + " " + reasoning

    if critique == "needs_clarification":
        has_q = "?" in summary
        has_flag = any("уточн" in f for f in flags)
        has_clarify_text = bool(re.search(
            r"уточните|какую\s+(именно|конкретно)\s+услугу|не\s+указан\s+товар"
            r"|прикрепите\s+фото|укажите\s+параметр|пожалуйста,?\s+уточни",
            summary,
        ))
        if (has_q and has_flag) or (has_clarify_text and has_flag):
            return "yes"
        if has_q or has_flag or has_clarify_text:
            return "partial"
        return "no"

    if critique == "wrong_catalog":
        has_catalog_ref = bool(re.search(
            r"артикул\s*\d+|арт\.\s*\d+|labus\.pro|goods\.csv|offers\.csv|orders\.csv"
            r"|id\s*\d{4,}|сделки?:\s*\d+|#\s*\d{4,}|набор\s+сделки",
            full_text,
        ))
        return "yes" if has_catalog_ref else "no"

    if critique == "wrong_math":
        # Hard to check without expected numbers — return n/a
        return "n/a"

    if critique == "missing_refs":
        bundle_size = new.get("key_fields", {}).get("suggested_bundle_size", 0)
        deal_citations = len(re.findall(r"id\s*\d{4,}|#\s*\d{4,}|сделки?:\s*\d+", full_text))
        if deal_citations >= 2 or bundle_size >= 3:
            return "yes"
        if deal_citations >= 1 or bundle_size >= 1:
            return "partial"
        return "no"

    if critique == "forbidden_phrase":
        return "n/a"  # needs context-specific phrase

    if critique == "workflow_guidance":
        has_script = bool(re.search(
            r"скрипт|дорожн|этап\s+\d|шаг\s+\d|бриф|переговорн\w+\s+сценарий"
            r"|ручной\s+расчёт|уточн\w+\s+у\s+клиент|менеджер\s+соберёт",
            full_text,
        ))
        return "yes" if has_script else "no"

    if critique == "brand_voice":
        # fb#45 rule: no "Дагестан" in output
        has_dagestan = "дагестан" in full_text
        return "yes" if not has_dagestan else "no"

    if critique == "regression":
        return "n/a"

    if critique == "cites_roadmap":
        # P11-R4: check if new response mentions roadmap keywords (регламент/этап/сроки/процесс)
        cited = bool(re.search(
            r"регламент|этапы?|сроки?|занимает\s+\d|в\s+течение|порядок\s+работ|процесс\s+(создани|разработ|производ)",
            full_text,
        ))
        return "yes" if cited else "no"

    if critique == "parametric_mismatch":
        # P12.2: tiraж/формат/площадь были в запросе — ответ должен их учесть (упомянуть в reasoning,
        # разбивке либо запросить параметр явно).
        has_param = bool(re.search(
            r"тираж\s*\d|\d+\s*шт|формат\s*a\d|\d+\s*(м|см)\s*[xх×]\s*\d|высот[аые]\s+\d|площад\w+\s+\d",
            full_text,
        ))
        asks_param = bool(re.search(
            r"уточните\s+тираж|на\s+какой\s+тираж|какой\s+формат|какой\s+размер|какая\s+высот",
            full_text,
        ))
        if has_param:
            return "yes"
        if asks_param:
            return "partial"
        return "no"

    if critique == "manager_script_intent":
        # P12.2: intent = скрипт/макро, ответ должен быть не ценой а репликой/сценарием.
        has_script = bool(re.search(
            r"скрипт|макро|ответ(ьте|ить)?\s+так|реплик\w+|сценари\w+\s+переговор|возражен\w+\s+клиент"
            r"|предложите\s+клиенту|скажите\s+клиенту|используйте\s+формулировк",
            full_text,
        ))
        return "yes" if has_script else "no"

    if critique == "brand_tier_miscalc":
        # P12.2: tier-aware ответ — цена должна быть в диапазоне, упомянут уровень (эконом/стандарт/премиум).
        has_tier = bool(re.search(
            r"эконом|стандарт|премиум|тир\s+\d|уровень\s+(разработ|оформ)|пакет\s+(базов|расширен)",
            full_text,
        ))
        return "yes" if has_tier else "no"

    if critique == "missing_article_link":
        # P12.2: ответ должен содержать артикул и URL labus.pro.
        has_article = bool(re.search(r"артикул\s*\d+|арт\.\s*\d+|id\s*\d{4,}", full_text))
        has_url = bool(re.search(r"labus\.pro/product|labus\.pro/\w", full_text))
        if has_article and has_url:
            return "yes"
        if has_article or has_url:
            return "partial"
        return "no"

    if critique == "missing_deal_example":
        # P12.2: ≥2 конкретных deal_id цитируется в reasoning/summary.
        deal_refs = len(re.findall(r"сделк\w*\s*(?:#|№|id)?\s*\d{4,}|#\s*\d{4,}\s*[:\-]", full_text))
        if deal_refs >= 2:
            return "yes"
        if deal_refs == 1:
            return "partial"
        return "no"

    return "n/a"


def verdict_for(pair: dict, critique_status: dict) -> str:
    """Overall verdict for this pair: improved | same | regressed | error."""
    if pair["new"].get("error"):
        return "error"
    critiques = pair.get("critique_types", [])
    if not critiques:
        return "n/a"
    addressed = sum(1 for c in critiques if critique_status.get(c) == "yes")
    partial = sum(1 for c in critiques if critique_status.get(c) == "partial")
    na = sum(1 for c in critiques if critique_status.get(c) == "n/a")
    unaddressed = len(critiques) - addressed - partial - na
    if addressed + partial * 0.5 > (len(critiques) - na) * 0.6:
        return "improved"
    if unaddressed > 0 and addressed == 0:
        return "regressed" if pair["rating"] == -1 else "same"
    return "same"


def format_price(val) -> str:
    if val is None:
        return "—"
    return f"{val:,.0f} ₽".replace(",", " ")


def build_report(replays: list[dict]) -> tuple[str, dict]:
    # Phase 1: per-pair enrichment
    for r in replays:
        r["topic"] = detect_topic(r["user_query"])
        r["critique_types"] = detect_critique_types(r["expert_comment"])
        status = {c: addresses(c, r["new"], r["expert_comment"]) for c in r["critique_types"]}
        r["critique_status"] = status
        r["verdict"] = verdict_for(r, status)

    # Phase 2: aggregation
    by_topic = defaultdict(list)
    by_verdict = defaultdict(int)
    by_critique = defaultdict(lambda: {"count": 0, "yes": 0, "partial": 0, "no": 0, "n/a": 0})

    for r in replays:
        by_topic[r["topic"]].append(r)
        by_verdict[r["verdict"]] += 1
        for c, s in r["critique_status"].items():
            by_critique[c]["count"] += 1
            by_critique[c][s] = by_critique[c].get(s, 0) + 1

    # Phase 3: report markdown
    lines = []
    ap = lines.append
    ap("# Feedback Replay — Findings Report")
    ap("")
    ap(f"**Source DB:** `_dbdump/2026-04-09/labus_rag.db`")
    ap(f"**Pairs:** {len(replays)} (expert-commented)")
    ap(f"**Target:** current prod (post P8.7-C, 75.5% eval pass rate)")
    ap("")
    ap("## Verdict Summary")
    ap("")
    ap("| Verdict | Count |")
    ap("|---|---|")
    for v in ("improved", "same", "regressed", "error", "n/a"):
        ap(f"| {v} | {by_verdict.get(v, 0)} |")
    ap("")
    ap("## Critique Coverage")
    ap("")
    ap("| Critique Type | Hits | Addressed | Partial | Not addressed | N/A |")
    ap("|---|---|---|---|---|---|")
    for c, _ in CRITIQUE_TYPES:
        row = by_critique.get(c, {})
        ap(f"| {c} | {row.get('count', 0)} | {row.get('yes', 0)} | "
           f"{row.get('partial', 0)} | {row.get('no', 0)} | {row.get('n/a', 0)} |")
    ap("")
    ap("## Topic Distribution")
    ap("")
    ap("| Topic | Count | Pairs |")
    ap("|---|---|---|")
    for topic, pairs in sorted(by_topic.items(), key=lambda x: -len(x[1])):
        ids = ", ".join(f"fb#{p['feedback_id']:02d}" for p in pairs)
        ap(f"| {topic} | {len(pairs)} | {ids} |")
    ap("")
    ap("## Per-Pair Details")
    ap("")
    for r in replays:
        ap(f"### fb#{r['feedback_id']:02d} — {r['topic']} — rating={r['rating']:+d} — verdict={r['verdict']}")
        ap("")
        ap(f"**Query:** {r['user_query']}")
        ap("")
        ap(f"**Expert critique:** {r['expert_comment'][:400]}{'…' if len(r['expert_comment'])>400 else ''}")
        ap("")
        old_kf = r["old"]["key_fields"]
        new_kf = r["new"]["key_fields"]
        ap(f"| Field | Old (from DB) | New (current prod) |")
        ap(f"|---|---|---|")
        ap(f"| confidence | {old_kf.get('confidence')} | {new_kf.get('confidence')} |")
        ap(f"| estimated_price | {format_price(old_kf.get('estimated_price'))} | {format_price(new_kf.get('estimated_price'))} |")
        ap(f"| price_band | {format_price(old_kf.get('price_band_min'))} – {format_price(old_kf.get('price_band_max'))} | {format_price(new_kf.get('price_band_min'))} – {format_price(new_kf.get('price_band_max'))} |")
        ap(f"| bundle_size | {old_kf.get('suggested_bundle_size', 0)} | {new_kf.get('suggested_bundle_size', 0)} |")
        ap(f"| flags_count | {len(old_kf.get('flags', []))} | {len(new_kf.get('flags', []))} |")
        ap("")
        if r["new"].get("error"):
            ap(f"**ERROR:** `{r['new']['error']}`")
            ap("")
            continue
        ap(f"**New summary:**")
        ap(f"> {new_kf.get('summary', '')[:600]}")
        ap("")
        if r["new"]["key_fields"].get("flags"):
            ap("**New flags:**")
            for f in r["new"]["key_fields"]["flags"]:
                ap(f"- {f}")
            ap("")
        if r["critique_types"]:
            ap("**Critique analysis:**")
            for c in r["critique_types"]:
                ap(f"- `{c}` → **{r['critique_status'].get(c, '—')}**")
            ap("")
        ap("---")
        ap("")

    return "\n".join(lines), {
        "replays": replays,
        "by_verdict": dict(by_verdict),
        "by_topic": {k: [r["feedback_id"] for r in v] for k, v in by_topic.items()},
        "by_critique": dict(by_critique),
    }


def build_cluster_report(replays: list[dict]) -> str:
    """P12.2.2: group pairs by (topic, critique), rank by count × severity.

    Severity: unaddressed critique = 2, partial = 1, addressed = 0. A cluster's
    score is sum of severities across its pairs. Top-10 clusters become the
    prioritized backlog for the current cycle.
    """
    clusters: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in replays:
        topic = r.get("topic", "other")
        for c, status in r.get("critique_status", {}).items():
            clusters[(topic, c)].append({
                "fb_id": r["feedback_id"],
                "rating": r["rating"],
                "verdict": r["verdict"],
                "status": status,
                "query": r["user_query"][:80],
                "comment": r["expert_comment"][:200],
            })

    severity_map = {"no": 2, "partial": 1, "yes": 0, "n/a": 0}
    ranked = []
    for (topic, critique), pairs in clusters.items():
        score = sum(severity_map.get(p["status"], 0) for p in pairs)
        ranked.append((score, topic, critique, pairs))
    ranked.sort(key=lambda x: (-x[0], -len(x[3])))

    lines = ["# Cluster Report — (topic × critique) prioritization", ""]
    lines.append(f"**Total pairs:** {len(replays)}")
    lines.append(f"**Distinct clusters:** {len(ranked)}")
    lines.append("")
    lines.append("## Top-10 by severity × count")
    lines.append("")
    lines.append("| Rank | Topic | Critique | Pairs | Severity | Unaddressed |")
    lines.append("|---|---|---|---|---|---|")
    for i, (score, topic, critique, pairs) in enumerate(ranked[:10], 1):
        unaddressed = sum(1 for p in pairs if p["status"] == "no")
        lines.append(f"| {i} | {topic} | {critique} | {len(pairs)} | {score} | {unaddressed} |")
    lines.append("")

    lines.append("## Detail")
    lines.append("")
    for i, (score, topic, critique, pairs) in enumerate(ranked[:15], 1):
        lines.append(f"### {i}. {topic} × {critique}  (severity={score}, pairs={len(pairs)})")
        lines.append("")
        for p in pairs:
            lines.append(f"- **fb#{p['fb_id']:02d}** rating={p['rating']:+d} verdict={p['verdict']} status={p['status']}")
            lines.append(f"  - Q: {p['query']}")
            if p["comment"].strip():
                lines.append(f"  - Expert: {p['comment']}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "replays.json"))
    ap.add_argument("--report", default=str(ROOT / "report.md"))
    ap.add_argument("--analysis", default=str(ROOT / "analysis.json"))
    ap.add_argument("--cluster-report", default=str(ROOT / "cluster_report.md"))
    args = ap.parse_args()

    replays = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report, analysis = build_report(replays)
    cluster_report = build_cluster_report(analysis["replays"])

    Path(args.report).write_text(report, encoding="utf-8")
    Path(args.analysis).write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.cluster_report).write_text(cluster_report, encoding="utf-8")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.analysis}")
    print(f"Wrote {args.cluster_report}")


if __name__ == "__main__":
    main()
