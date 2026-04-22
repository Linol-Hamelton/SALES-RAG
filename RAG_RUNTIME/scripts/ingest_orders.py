#!/usr/bin/env python3
"""
P13.3 / T7: Ingest historical deals from orders.csv as semantic references.

Source: RAG_DATA/orders.csv (~50K rows, ~12.6K unique deals 2021-2026).

Each unique deal -> one doc with doc_type="historical_deal". Used by
retriever for pricing/sizing/scope_estimation intents to surface
"похожие закрытые сделки" — concrete past sales the manager can cite.

PII guard (mandatory per plan R5):
  - COMPANY_ID, CONTACT_ID, EXECUTOR are dropped entirely from payload.
  - TITLE is kept (deal name like "0101", "1325" — internal Bitrix labels).
  - searchable_text contains ONLY product names + categories + direction,
    never names of company/contact.

Filters applied:
  - drop deals with OPPORTUNITY <= 0
  - drop deals with no items having PARENT_SECTION
  - drop deals older than 2022-01-01 (low retrieval value)

Output: data/historical_deal_docs.jsonl
"""
import csv
import json
import sys
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).parent.parent
RAG_DATA = PROJECT_ROOT.parent / "RAG_DATA"
OUTPUT_DIR = PROJECT_ROOT / "data"

CSV_PATH = RAG_DATA / "orders.csv"
OUTPUT_PATH = OUTPUT_DIR / "historical_deal_docs.jsonl"

GENERATED_AT = datetime.now(timezone.utc).isoformat()

# Drop deals closed before this date — older quotes drift too much from
# current price reality and waste index space.
MIN_CLOSE_YEAR = 2022


def _norm(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _bucket_price(total: float) -> str:
    if total < 5_000:
        return "до 5К"
    if total < 25_000:
        return "5–25К"
    if total < 100_000:
        return "25–100К"
    if total < 500_000:
        return "100–500К"
    return "от 500К"


def _build_searchable(title: str, year: int, parent_sections: list[str],
                      directions: list[str], product_names: list[str],
                      total_price: float) -> str:
    """Create dense-retrieval text. Order matters — high-signal up top."""
    parts = []
    if parent_sections:
        # Top categories first (most distinct PARENT_SECTION values)
        parts.append("Категории: " + ", ".join(parent_sections[:6]))
    if directions:
        parts.append("Направление: " + ", ".join(directions[:4]))
    if product_names:
        parts.append("Состав сделки: " + "; ".join(product_names[:12]))
    parts.append(f"Сделка №{title} ({year}). Сумма: {int(total_price):,} ₽".replace(",", " "))
    parts.append(f"Ценовой сегмент: {_bucket_price(total_price)}")
    return "\n".join(parts)


@click.command()
@click.option("--verbose", is_flag=True, default=False)
@click.option("--limit", type=int, default=0, help="Max deals to process (0=all)")
def main(verbose: bool, limit: int):
    if not CSV_PATH.exists():
        print(f"[ERR] csv not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    deals: dict[str, list[dict]] = defaultdict(list)
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            did = _norm(row.get("ID"))
            if did:
                deals[did].append(row)

    print(f"Loaded {sum(len(v) for v in deals.values())} rows / {len(deals)} unique deals")

    docs = []
    skipped_no_money = 0
    skipped_no_items = 0
    skipped_old = 0
    skipped_no_section = 0

    for deal_id, items in deals.items():
        first = items[0]
        opportunity = _parse_float(first.get("OPPORTUNITY"))
        if opportunity <= 0:
            skipped_no_money += 1
            continue

        close_date = _norm(first.get("CLOSEDATE"))
        if not close_date:
            skipped_old += 1
            continue
        try:
            year = int(close_date[:4])
        except ValueError:
            skipped_old += 1
            continue
        if year < MIN_CLOSE_YEAR:
            skipped_old += 1
            continue

        # Aggregate item-level data
        product_names: list[str] = []
        parent_sections: list[str] = []
        directions: list[str] = []
        item_records: list[dict] = []
        item_total = 0.0

        for r in items:
            pname = _norm(r.get("PRODUCT_NAME"))
            psection = _norm(r.get("PARENT_SECTION"))
            direction_blk = _norm(r.get("Направление"))
            price = _parse_float(r.get("PRICE"))
            qty = _parse_int(r.get("QUANTITY"))
            line_total = price * qty
            item_total += line_total

            if pname:
                product_names.append(pname)
            if psection:
                parent_sections.append(psection)
            if direction_blk:
                directions.append(direction_blk)

            item_records.append({
                "product_id": _norm(r.get("PRODUCT_ID")),
                "product_name": pname,
                "parent_section": psection,
                "section_name": _norm(r.get("SECTION_NAME")),
                "direction": direction_blk,
                "price": price,
                "quantity": qty,
                "line_total": round(line_total, 2),
            })

        if not item_records:
            skipped_no_items += 1
            continue

        # Distinct ordering preserves first-occurrence rank for searchable_text
        seen = set()
        unique_sections = [s for s in parent_sections if not (s in seen or seen.add(s))]
        seen = set()
        unique_directions = [d for d in directions if not (d in seen or seen.add(d))]
        if not unique_sections:
            skipped_no_section += 1
            continue

        # Use OPPORTUNITY as canonical total (Bitrix authoritative); fall back to sum
        total_price = opportunity if opportunity > 0 else round(item_total, 2)

        title = _norm(first.get("TITLE")) or deal_id

        searchable = _build_searchable(
            title=title,
            year=year,
            parent_sections=unique_sections,
            directions=unique_directions,
            product_names=product_names,
            total_price=total_price,
        )

        doc_id = f"historical_deal_{deal_id}"

        # Top-3 PARENT_SECTION as the deal's signature categories
        sig_sections = [s for s, _ in Counter(parent_sections).most_common(3)]

        doc = {
            "doc_id": doc_id,
            "doc_type": "historical_deal",
            "searchable_text": searchable,
            "payload": {
                "doc_id": doc_id,
                "doc_type": "historical_deal",
                "deal_id": deal_id,
                "deal_title": title,
                "year": year,
                "close_date": close_date[:10],
                "total_price": round(total_price, 2),
                "price_bucket": _bucket_price(total_price),
                "is_return_customer": _norm(first.get("IS_RETURN_CUSTOMER")) == "Y",
                "signature_sections": sig_sections,
                "all_sections": unique_sections,
                "directions": unique_directions,
                "items": item_records[:30],  # cap to keep payload bounded
                "items_count": len(item_records),
                "searchable_text": searchable,
                "display_label": f"Сделка №{title} ({year}) — {_bucket_price(total_price)}",
            },
            "provenance": {
                "source": "orders.csv",
                "deal_id": deal_id,
                "generated_at": GENERATED_AT,
            },
        }
        docs.append(doc)
        if verbose and len(docs) <= 5:
            msg = f"  [OK] {doc_id}: {len(item_records)} items, {int(total_price):,} RUB, sections={sig_sections}"
            sys.stdout.buffer.write((msg + "\n").encode("utf-8"))

        if limit and len(docs) >= limit:
            break

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(docs)} historical_deal docs to {OUTPUT_PATH}")
    print(f"Skipped: no_money={skipped_no_money}, no_items={skipped_no_items}, "
          f"old(<{MIN_CLOSE_YEAR})={skipped_old}, no_section={skipped_no_section}")


if __name__ == "__main__":
    main()
