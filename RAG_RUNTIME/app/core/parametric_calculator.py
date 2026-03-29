"""
Parametric Calculator: given component specs and retrieved product docs,
compute a line-item price estimate.

Extracts per-unit prices from Qdrant payload metadata and multiplies
by estimated quantities (linear meters, hours, etc.).
"""
import re
from dataclasses import dataclass, field

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LineItem:
    component: str          # component type key
    label: str              # human label
    product_name: str       # matched product name
    product_id: str         # doc_id
    unit_price: float       # price per unit
    quantity: float
    unit: str
    total: float
    confidence_tier: str    # auto/guided/manual/low/medium/high
    price_source: str       # "order_p50", "recommended", "base_catalog"
    quantity_basis: str     # how quantity was derived


@dataclass
class ParametricEstimate:
    line_items: list[LineItem] = field(default_factory=list)
    total_estimate: float = 0.0
    total_min: float = 0.0
    total_max: float = 0.0
    is_parametric: bool = False
    missing_components: list[str] = field(default_factory=list)  # components with no match
    confidence: str = "guided"   # overall confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_price_from_metadata(meta: dict) -> tuple[float, str]:
    """Return (price, source_label) from product metadata, best available."""
    # Priority: order p50 > recommended > base catalog
    p50 = meta.get("order_price_p50")
    if p50 and p50 > 0:
        return float(p50), "order_p50"

    rec = meta.get("recommended_price")
    if rec and rec > 0:
        return float(rec), "recommended"

    base = meta.get("current_base_price")
    if base and base > 0:
        return float(base), "base_catalog"

    return 0.0, "unknown"


def _parse_quantity_range(product_name: str) -> tuple[float, float]:
    """Parse quantity range like '3-10 мп', '11-50 мп', 'до 3 мп' from name.
    Returns (min, max) in мп, or (-inf, +inf) if no range found.
    """
    # Pattern: "3-10 мп" or "11–50 мп"
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*мп", product_name, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Pattern: "до 3 мп"
    m = re.search(r"до\s*(\d+)\s*мп", product_name, re.IGNORECASE)
    if m:
        return 0.0, float(m.group(1))

    # Pattern: "более 100 мп" or ">100 мп"
    m = re.search(r"(?:более|свыше|>)\s*(\d+)\s*мп", product_name, re.IGNORECASE)
    if m:
        return float(m.group(1)), float("inf")

    return float("-inf"), float("inf")  # no range constraint


def _range_score(product_name: str, quantity: float) -> float:
    """Score how well a product's range fits the target quantity.
    Lower is better (0 = perfect).
    """
    lo, hi = _parse_quantity_range(product_name)
    if lo == float("-inf"):
        return 50.0  # generic, deprioritize

    if lo <= quantity <= hi:
        # Perfect fit — prefer narrower ranges
        range_width = hi - lo if hi != float("inf") else 1000
        return range_width / 1000.0  # tiebreaker: narrower is better

    # Undershooting (quantity < range)
    if quantity < lo:
        return lo - quantity

    # Overshooting (quantity > range)
    return quantity - hi


def _find_best_doc(docs: list[dict], quantity: float, unit: str) -> dict | None:
    """Find the product doc whose quantity range best fits the target quantity."""
    if not docs:
        return None

    scored = []
    for doc in docs:
        payload = doc.get("payload", {})
        # Prices are stored directly in payload (not nested under "metadata")
        name = payload.get("product_name", "") or payload.get("searchable_text", "")[:60]

        # Only consider product docs for parametric pricing
        if payload.get("doc_type") != "product":
            continue

        price, source = _extract_price_from_metadata(payload)
        if price <= 0:
            continue

        # Score by quantity range fit
        if unit == "мп" and quantity > 0:
            score = _range_score(name, quantity)
        else:
            score = 0.0  # no range constraint — any will do

        scored.append((score, doc, price, source, name))

    if not scored:
        return None

    # Sort: lower score = better
    scored.sort(key=lambda x: x[0])
    best_score, best_doc, price, source, name = scored[0]

    # Attach resolved price and source to doc for caller
    best_doc["_resolved_price"] = price
    best_doc["_price_source"] = source
    best_doc["_product_name"] = name
    return best_doc


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------

def calculate(
    components: list,  # list[ComponentSpec] from query_decomposer
    doc_pools: dict[str, list[dict]],  # component_type → retrieved docs
) -> ParametricEstimate:
    """
    Calculate parametric price from component specs and retrieved docs.

    Args:
        components: list of ComponentSpec objects
        doc_pools: dict mapping component type → list of retrieved doc results

    Returns:
        ParametricEstimate with line_items and total
    """
    line_items: list[LineItem] = []
    missing: list[str] = []

    for comp in components:
        docs = doc_pools.get(comp.type, [])
        best = _find_best_doc(docs, comp.quantity, comp.unit)

        if best is None:
            logger.warning("No matching doc for component",
                           component=comp.type, qty=comp.quantity)
            missing.append(comp.label)
            continue

        unit_price = best["_resolved_price"]
        source = best["_price_source"]
        product_name = best["_product_name"]
        payload = best.get("payload", {})

        quantity = comp.quantity if comp.quantity > 0 else 1.0
        total = unit_price * quantity

        confidence_tier = payload.get("confidence_tier", "low")

        line_items.append(LineItem(
            component=comp.type,
            label=comp.label,
            product_name=product_name,
            product_id=payload.get("doc_id", ""),
            unit_price=unit_price,
            quantity=quantity,
            unit=comp.unit,
            total=total,
            confidence_tier=confidence_tier,
            price_source=source,
            quantity_basis=comp.quantity_basis,
        ))

    if not line_items:
        return ParametricEstimate(is_parametric=False, missing_components=missing)

    # Aggregate
    total_estimate = sum(item.total for item in line_items)

    # Confidence: weighted by component total (heavier items drive confidence more)
    tier_weights = {"auto": 3, "high": 3, "guided": 2, "medium": 2, "low": 1, "manual": 1}
    weighted_sum = sum(tier_weights.get(item.confidence_tier, 1) * item.total for item in line_items)
    weighted_total = total_estimate if total_estimate > 0 else 1.0
    avg_weight = weighted_sum / weighted_total

    if avg_weight >= 2.8:
        overall_confidence = "auto"
        band_factor = 0.10
    elif avg_weight >= 1.8:
        overall_confidence = "guided"
        band_factor = 0.25
    else:
        overall_confidence = "manual"
        band_factor = 0.35

    total_min = round(total_estimate * (1 - band_factor))
    total_max = round(total_estimate * (1 + band_factor))

    logger.info("Parametric estimate",
                total=total_estimate,
                items=len(line_items),
                missing=len(missing),
                confidence=overall_confidence)

    return ParametricEstimate(
        line_items=line_items,
        total_estimate=round(total_estimate, 2),
        total_min=float(total_min),
        total_max=float(total_max),
        is_parametric=True,
        missing_components=missing,
        confidence=overall_confidence,
    )


def format_breakdown(estimate: ParametricEstimate) -> str:
    """Format parametric breakdown as human-readable text for LLM context."""
    if not estimate.is_parametric or not estimate.line_items:
        return ""

    lines = [
        "ВНУТРЕННИЕ НОРМЫ ПРОИЗВОДСТВА (себестоимость операций цеха — НЕ розничная цена клиенту):",
        "Розничная цена ≈ себестоимость × 10–15. Используй данные [Набор] для реальной цены продажи.",
    ]
    for item in estimate.line_items:
        lines.append(
            f"  • {item.label}: {item.quantity:.1f} {item.unit} × "
            f"{item.unit_price:,.0f} ₽/ед (себест.) = {item.total:,.0f} ₽"
            f" [{item.confidence_tier}] ({item.price_source})"
        )

    lines.append(f"  Себестоимость итого: {estimate.total_estimate:,.0f} ₽ (НЕ цена клиенту)")
    lines.append(f"  Уверенность данных: {estimate.confidence}")

    if estimate.missing_components:
        lines.append(f"  Не найдено в базе: {', '.join(estimate.missing_components)}")

    return "\n".join(lines)
