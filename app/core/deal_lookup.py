"""
DealLookup: loads real deal line items from normalized offers/orders CSV
and serves them as DealItem lists for the estimate feature.

Architecture:
- At startup, loads offers.normalized.csv + orders.normalized.csv into memory
- Groups rows by deal ID
- On estimate: finds deal IDs from retrieved deal_profile docs, returns their real items
- Falls back to LLM-generated items if no matching deal found
"""
import csv
from pathlib import Path
from app.schemas.pricing import DealItem
from app.utils.logging import get_logger

logger = get_logger("deal_lookup")

# Columns we need from normalized CSV
_COL_ID           = "ID"
_COL_PRODUCT_NAME = "PRODUCT_NAME"
_COL_QUANTITY     = "QUANTITY"
_COL_PRICE        = "PRICE"
_COL_LINE_TOTAL   = "LINE_TOTAL"
_COL_NAME         = "NAME"           # full spec: "Монтаж буквы простой в цеху 11-50 мп Цех"
_COL_SECTION      = "SECTION_NAME" if "SECTION_NAME" else "SECTION"
_COL_PARENT       = "PARENT_SECTION"
_COL_DIRECTION    = "Направление"
_COL_UNIT         = None             # unit is embedded in NAME, will be parsed


# Parent section → Bitrix24 catalog section mapping
_PARENT_TO_B24 = {
    "Буквы":            "Цех",
    "Короба":           "Цех",
    "Листовые":         "Цех",
    "Электрика":        "Цех",
    "металл":           "Цех",
    "Пленки":           "Цех",
    "Самоклейка":       "Печатная",
    "Визитки":          "Печатная",
    "Баннер":           "Сольвент",
    "Дизайн":           "Дизайн",
    "Монтаж":           "РИК",
    "Мерч":             "Мерч",
}


def _parent_to_b24_section(parent: str, direction: str) -> str:
    """Map PARENT_SECTION + Направление to Bitrix24 section label."""
    if direction in ("Печатная", "Сольвент", "Дизайн", "РИК", "Мерч"):
        return direction
    return _PARENT_TO_B24.get(parent, direction or "Цех")


def _parse_unit_from_name(name: str) -> str:
    """Guess unit from product name string."""
    n = (name or "").lower()
    if any(x in n for x in ["кв.м", "кв. м", "m²"]):
        return "кв.м"
    if "мп" in n:
        return "мп"
    if "пог" in n or "погон" in n:
        return "мп"
    if any(x in n for x in ["шт", "штук"]):
        return "шт"
    if "компл" in n:
        return "компл"
    return "шт"


def _safe_float(val, default=0.0) -> float:
    try:
        return float(str(val).replace(",", ".").strip()) if val else default
    except (ValueError, TypeError):
        return default


class DealLookup:
    """In-memory index of deal line items from normalized CSV files."""

    def __init__(self, analytics_root: Path):
        self._analytics_root = analytics_root
        # deal_id (str) -> list of raw row dicts
        self._deals: dict[str, list[dict]] = {}
        self._loaded = False

    def load(self):
        """Load offers.normalized.csv and orders.normalized.csv into memory."""
        files = [
            self._analytics_root / "normalized" / "offers.normalized.csv",
            self._analytics_root / "normalized" / "orders.normalized.csv",
        ]
        total_rows = 0
        for fpath in files:
            if not fpath.exists():
                logger.warning("DealLookup: file not found", path=str(fpath))
                continue
            try:
                with open(fpath, encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f, delimiter=";")
                    for row in reader:
                        deal_id = str(row.get(_COL_ID, "")).strip()
                        if not deal_id:
                            continue
                        self._deals.setdefault(deal_id, []).append(dict(row))
                        total_rows += 1
            except Exception as e:
                logger.error("DealLookup: failed to load file", path=str(fpath), error=str(e))

        self._loaded = True
        logger.info("DealLookup loaded",
                    deals=len(self._deals),
                    rows=total_rows)

    def get_deal_items(self, deal_id: str) -> list[DealItem]:
        """Return DealItem list for a specific deal ID."""
        rows = self._deals.get(str(deal_id), [])
        return self._rows_to_items(rows)

    def find_best_deal_id(self, retrieved_docs: list[dict]) -> str | None:
        """
        Extract the most relevant deal ID from retrieved deal_profile docs.
        Returns the deal ID with the highest relevance score.
        """
        for doc in retrieved_docs:
            payload = doc.get("payload", {})
            if payload.get("doc_type") != "deal_profile":
                continue
            deal_id = payload.get("deal_id") or ""
            # Strip prefix "deal_profile_" if present
            deal_id = deal_id.replace("deal_profile_", "")
            if deal_id and str(deal_id) in self._deals:
                return str(deal_id)
        return None

    def find_best_deal_items(self, retrieved_docs: list[dict]) -> tuple[list[DealItem], str]:
        """
        Find the best matching deal and return its items + deal title.
        Returns (items, deal_title). Items is empty list if no match found.
        """
        deal_id = self.find_best_deal_id(retrieved_docs)
        if not deal_id:
            return [], ""
        rows = self._deals.get(deal_id, [])
        title = rows[0].get("TITLE", "") if rows else ""
        items = self._rows_to_items(rows)
        logger.info("DealLookup matched deal", deal_id=deal_id,
                    title=title[:60], items=len(items))
        return items, title

    def _rows_to_items(self, rows: list[dict]) -> list[DealItem]:
        items = []
        for row in rows:
            # Prefer NAME (full spec) over PRODUCT_NAME for display
            name = (row.get(_COL_NAME) or row.get(_COL_PRODUCT_NAME) or "").strip()
            if not name:
                continue

            qty = _safe_float(row.get("QUANTITY_NUM") or row.get(_COL_QUANTITY))
            unit_price = _safe_float(row.get("PRICE_NUM") or row.get(_COL_PRICE))
            line_total = _safe_float(row.get(_COL_LINE_TOTAL))
            if line_total == 0 and unit_price > 0 and qty > 0:
                line_total = round(qty * unit_price, 2)

            parent = str(row.get(_COL_PARENT, "") or "").strip()
            direction = str(row.get(_COL_DIRECTION, "") or "").strip()
            b24_section = _parent_to_b24_section(parent, direction)
            unit = _parse_unit_from_name(name)

            items.append(DealItem(
                product_name=name,
                quantity=qty,
                unit=unit,
                unit_price=unit_price,
                total=line_total,
                b24_section=b24_section,
                notes="",
            ))
        return items
