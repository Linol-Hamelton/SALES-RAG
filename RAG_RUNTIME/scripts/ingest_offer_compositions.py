#!/usr/bin/env python3
"""
Generate offer_composition documents from offers.normalized.csv.

Each offer_composition doc represents one commercial offer (КП) with its full
product breakdown — so the LLM can reference exact catalog product names
and prices when building estimates.

По умолчанию эмитит все offers с ≥2 line-items (порог, чтобы отсечь одиночные
добавки/правки). `--only-key` — исторический whitelist KEY_OFFER_IDS
(для debug и быстрой отладки link-following).

Usage:
    python scripts/ingest_offer_compositions.py [--verbose] [--only-key]

Output:
    data/offer_composition_docs.jsonl
"""
import argparse
import csv
import json
import re
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).parent.parent
OFFERS_CSV = PROJECT_ROOT.parent / "RAG_ANALYTICS" / "output" / "normalized" / "offers.normalized.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "offer_composition_docs.jsonl"
GENERATED_AT = datetime.now(timezone.utc).isoformat()

# Key offer IDs to always include (from expert feedback + bridge analysis)
KEY_OFFER_IDS = {
    # Logo packages
    21208, 21210, 21214,
    # Brand identity packages
    21216, 21218, 21220,
    # Brandbook packages
    37272, 48444, 48452,
    # Industry brandbooks
    49276, 49278, 49280, 49282, 49284,
    # Standalone logo components
    30912, 30914, 31792, 31794, 31796, 31798, 31800, 31802,
}

# Package tier mapping by offer ID
PACKAGE_TIER = {
    21208: "Стандарт", 21210: "Базовый", 21214: "Креативная группа",
    21216: "Стандарт", 21218: "Базовый", 21220: "Креативная группа",
    37272: "Стандарт", 48444: "Базовый", 48452: "Креативная группа",
    49276: "Для ивента", 49278: "Ресторанный бизнес",
    49280: "Медицинский центр", 49282: "Строительная компания",
    49284: "Салон красоты",
}


def slugify(text: str) -> str:
    tr = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
        "ж": "zh", "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
    result = []
    for ch in text.lower():
        if ch in tr:
            result.append(tr[ch])
        elif ch.isascii() and ch.isalnum():
            result.append(ch)
        elif ch in (" ", "-", "_"):
            result.append("_")
    slug = re.sub(r"_+", "_", "".join(result)).strip("_")
    return slug[:60]


def load_offers(csv_path: Path) -> dict:
    """Load offers grouped by offer ID."""
    offers = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            oid = row.get("ID", "")
            if not oid:
                continue
            oid_int = int(oid)
            if oid_int not in offers:
                offers[oid_int] = {
                    "id": oid_int,
                    "title": row.get("TITLE", ""),
                    "opportunity": row.get("OPPORTUNITY", ""),
                    "description": row.get("DESCRIPTION", ""),
                    "direction": row.get("NORMALIZED_DIRECTION", "") or row.get("Направление", ""),
                    "begin_date": row.get("BEGINDATE", ""),
                    "close_date": row.get("CLOSEDATE", ""),
                    "items": [],
                }
            price = float(row.get("PRICE", "0") or "0")
            qty = float(row.get("QUANTITY", "0") or "0")
            offers[oid_int]["items"].append({
                "product_name": row.get("PRODUCT_NAME", ""),
                "product_id": row.get("PRODUCT_ID", ""),
                "good_id": row.get("GOOD_ID", ""),
                "price": price,
                "quantity": qty,
                "line_total": price * qty,
                "base_price": float(row.get("BASE_PRICE", "0") or "0"),
            })
    return offers


def build_searchable_text(offer: dict, tier: str) -> str:
    """Build text for embedding."""
    parts = []
    title = offer["title"]
    direction = offer["direction"]

    parts.append(f"[Коммерческое предложение: {title}]")
    if direction:
        parts.append(f"Направление: {direction}")
    if tier:
        parts.append(f"Пакет: {tier}")

    total = sum(item["line_total"] for item in offer["items"])
    parts.append(f"Итого: {total:,.0f} ₽")

    parts.append("Состав:")
    for item in offer["items"]:
        name = item["product_name"]
        qty = item["quantity"]
        price = item["price"]
        line = f"  • {name}"
        if qty != 1:
            line += f" × {qty:g}"
        line += f" — {price:,.0f} ₽"
        parts.append(line)

    if offer.get("description"):
        desc = offer["description"][:500]
        parts.append(f"\nОписание: {desc}")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate offer_composition JSONL")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--only-key", action="store_true",
                        help="Debug: эмитить только KEY_OFFER_IDS whitelist (22 docs)")
    parser.add_argument("--min-items", type=int, default=2,
                        help="Минимум line-items на offer (default: 2 — отсекает мусор)")
    args = parser.parse_args()

    if not OFFERS_CSV.exists():
        print(f"ERROR: offers CSV not found: {OFFERS_CSV}")
        return

    all_offers = load_offers(OFFERS_CSV)
    print(f"Loaded {len(all_offers)} unique offers from {OFFERS_CSV.name}")

    if args.only_key:
        target_ids = KEY_OFFER_IDS & set(all_offers.keys())
        print(f"  --only-key: filtered to {len(target_ids)} whitelisted offers")
    else:
        target_ids = {
            oid for oid, offer in all_offers.items()
            if len(offer["items"]) >= args.min_items
        }
        print(f"  default: {len(target_ids)} offers with ≥{args.min_items} line-items")

    docs = []
    for oid in sorted(target_ids):
        offer = all_offers[oid]
        tier = PACKAGE_TIER.get(oid, "")
        total = sum(item["line_total"] for item in offer["items"])

        searchable = build_searchable_text(offer, tier)

        metadata = {
            "source": "offer_composition",
            "offer_id": oid,
            "title": offer["title"],
            "direction": offer["direction"],
            "package_tier": tier,
            "total_price": total,
            "item_count": len(offer["items"]),
            "products": [
                {
                    "name": item["product_name"],
                    "qty": item["quantity"],
                    "price": item["price"],
                    "line_total": item["line_total"],
                }
                for item in offer["items"]
            ],
        }

        doc = {
            "doc_id": f"offer_{oid}",
            "doc_type": "offer_composition",
            "searchable_text": searchable,
            "metadata": metadata,
            "provenance": {
                "source": "ingest_offer_compositions.py",
                "generated_at": GENERATED_AT,
            },
        }
        docs.append(doc)

        if args.verbose:
            print(f"  offer_{oid}: {offer['title'][:50]} | {total:,.0f}₽ | {len(offer['items'])} items | tier={tier}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nWritten: {OUTPUT_PATH} ({len(docs)} offer compositions)")


if __name__ == "__main__":
    main()
