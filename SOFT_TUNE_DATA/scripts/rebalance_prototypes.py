"""P14.6: rebalance intent prototypes — move queries from describe to
more specific intents (objection_arguments, bundle_query) based on content.

After v4 measured -10pp regressions on objection_handling/upsell/logistics:
hypothesis is that describe prototype set (60 examples) is too broad and
pulls manager-workflow queries away from their correct intents.

Strategy: read configs/intent_prototypes.yaml, walk describe set,
redistribute by content rules:
  - «почему [...] дороже/дешевле/выше», «у конкурентов» → objection_arguments
  - «как осметить», «прикинь», «рассчитайте» → smeta_request
  - «как предложить», «увеличить чек», «upsell», «дополнительно» → bundle_query
  - keep rest in describe

Cap describe at 30 (smaller, more focused on script-drafting).

Usage:
  PYTHONIOENCODING=utf-8 python SOFT_TUNE_DATA/scripts/rebalance_prototypes.py
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
YAML_PATH = ROOT / "configs" / "intent_prototypes.yaml"

OBJECTION_PATTERNS = [
    re.compile(r"\bу\s+конкурент", re.IGNORECASE),
    re.compile(r"в\s+(соседн\w+|другой)\s+(типограф|магазин|конторе)", re.IGNORECASE),
    re.compile(r"\bдешевле\b|\bдороже\b|\bдешев\w+\s+на\b", re.IGNORECASE),
    re.compile(r"почему\s+(у\s+нас\s+|наш\w*\s+|вы\w*\s+)?(дороже|выше|больше|дорог)", re.IGNORECASE),
    re.compile(r"скидк\w*\s+(дадите|сделайте|давайте)", re.IGNORECASE),
    re.compile(r"(аргументир\w+|обоснов\w+)\s+(цен|стоимост)", re.IGNORECASE),
    re.compile(r"(убедить?|объяснить?)\s+клиент\w*", re.IGNORECASE),
    re.compile(r"\bКП\b.+конкурент|конкурент\w*.+\bКП\b", re.IGNORECASE),
    re.compile(r"клиент\s+(говорит|пишет|сомневается|просит\s+скидк)", re.IGNORECASE),
    re.compile(r"(\d+\s*%\s+ниже|на\s+\d+\s*%)", re.IGNORECASE),
]

BUNDLE_UPSELL_PATTERNS = [
    re.compile(r"(увеличить?\s+чек|повысить?\s+стоимост|upsell)", re.IGNORECASE),
    re.compile(r"премиум(\w+)?(\s+картон|\s+бумаг|\s+материал)", re.IGNORECASE),
    re.compile(r"(дополнительн\w+|апсейл|кросс-?продаж)", re.IGNORECASE),
    re.compile(r"как\s+предложить\s+\w+\s+премиум", re.IGNORECASE),
    re.compile(r"тиснени\w+\s+и\s+(фольг|ламинац)", re.IGNORECASE),
    re.compile(r"закаж\w+\s+(оптом|объем|партии)", re.IGNORECASE),
]

SMETA_ASSIST_PATTERNS = [
    re.compile(r"(осмет\w+|рассчита\w+)\s+(low|high|стандарт)", re.IGNORECASE),
    re.compile(r"как\s+мне\s+осмет\w+", re.IGNORECASE),
    re.compile(r"прикин\w+\s+(цен|стоимост|смет)", re.IGNORECASE),
    re.compile(r"low-?spec\s+альтернатив", re.IGNORECASE),
]


def classify(query: str) -> str:
    """Return target intent for a query currently in describe."""
    if any(rx.search(query) for rx in OBJECTION_PATTERNS):
        return "objection_arguments"
    if any(rx.search(query) for rx in BUNDLE_UPSELL_PATTERNS):
        return "bundle_query"
    if any(rx.search(query) for rx in SMETA_ASSIST_PATTERNS):
        return "smeta_request"
    return "describe"


def main():
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8")) or {}
    describe = data.get("describe", [])
    print(f"Loaded describe: {len(describe)} prototypes")

    redistributed = {
        "describe": [],
        "objection_arguments": [],
        "bundle_query": [],
        "smeta_request": [],
    }

    for q in describe:
        target = classify(q)
        redistributed[target].append(q)

    print("\nRedistribution:")
    for intent, queries in redistributed.items():
        print(f"  {intent}: {len(queries)}")

    # Merge into existing intents
    for intent in ("objection_arguments", "bundle_query", "smeta_request"):
        existing = data.get(intent, [])
        merged = list(existing)
        seen = set(existing)
        for q in redistributed[intent]:
            if q not in seen:
                merged.append(q)
                seen.add(q)
        data[intent] = merged

    # Cap describe at 30 (most generic script-drafting examples first)
    data["describe"] = redistributed["describe"][:30]

    print("\nFinal counts:")
    for intent in sorted(data.keys()):
        print(f"  {intent}: {len(data[intent])}")

    # Write back
    lines = [
        "# Intent prototypes for BGE-M3 embedding-based classification (Tier 2).",
        "# Threshold: best match > 0.75 → classify as that intent.",
        "#",
        "# P14.5: expanded from 58 → 345 via labelled v1/v2/v3 dataset.",
        "# P14.6: rebalanced — moved 25+ queries from describe → objection_arguments /",
        "# bundle_query / smeta_request to fix v4 manager-topic regressions.",
        "",
    ]
    for intent in sorted(data.keys()):
        queries = data[intent]
        if not queries:
            continue
        lines.append(f"{intent}:")
        for q in queries:
            escaped = q.replace("'", "''")
            lines.append(f"  - '{escaped}'")
        lines.append("")
    YAML_PATH.write_text("\n".join(lines), encoding="utf-8")
    total = sum(len(v) for v in data.values())
    print(f"\n✓ Wrote {len(data)} intents, {total} total prototypes → {YAML_PATH}")


if __name__ == "__main__":
    main()
