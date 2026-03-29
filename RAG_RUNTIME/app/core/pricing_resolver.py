"""
Pricing resolution logic.
Mirrors classifyProductFact() from RAG_ANALYTICS/lib/pipeline.mjs.
"""
from dataclasses import dataclass, field
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Russian flag messages per manual review reason
MANUAL_REVIEW_FLAGS = {
    "insufficient_history": "Мало данных о заказах. Требуется ручная оценка менеджером.",
    "high_price_variance": "Высокая вариативность цены. Запросите индивидуальный расчёт.",
    "financial_modifier": "Финансовый модификатор (безнал/скидка/надбавка). Не является самостоятельным продуктом.",
    "manual_or_missing_catalog": "Нестандартный продукт или позиция не из каталога. Требуется расчёт по ТЗ.",
    "no_price_anchor": "Нет ценового ориентира. Уточните требования и сформируйте ТЗ.",
}


@dataclass
class PricingResolution:
    confidence: str = "manual"   # "auto" | "guided" | "manual"
    estimated_value: float | None = None
    estimated_basis: str = ""
    price_band_min: float | None = None
    price_band_max: float | None = None
    flags: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    is_financial_modifier: bool = False
    source_doc_id: str = ""
    source_doc_type: str = ""
    # Pre-calculated under-key totals (so LLM doesn't do arithmetic)
    total_under_key_min: float | None = None
    total_under_key_max: float | None = None
    price_breakdown: dict = field(default_factory=dict)  # {"буквы": (min, max), ...}


# Section-level markup ranges derived from Bitrix24 deal data (P50 order price / cost_price)
# Calibrated 2026-03-24 from 2726 products with known cost + order history
_SECTION_MARKUP_RANGES: dict[str, tuple[float, float]] = {
    # Цех — объёмные буквы, вывески
    "Объем": (1.5, 2.5),          # median 2.04, range 1.41-11.63
    "Аппликация": (1.3, 2.0),     # median 1.60
    "Монтаж": (1.2, 1.8),         # median 1.52
    "Щит": (1.1, 1.6),            # median 1.30, range 0.36-3.52
    # Печатная продукция
    "Печать": (2.0, 3.5),         # median 2.73
    "Ламинация": (1.5, 2.5),      # median 2.16
    "Полиграфия": (1.5, 2.5),     # median 2.00
    "Бумага": (1.2, 1.8),         # median 1.49
    "Брошюры": (1.1, 1.5),        # median 1.26
    "Офсет": (1.1, 1.5),          # median 1.25
}
_DEFAULT_MARKUP = (1.3, 2.0)

# Height-dependent service cost ranges (RUB)
_SERVICE_COSTS = {
    # height_threshold: (mounting_range, frame_range)
    50:  ((20_000, 30_000), (10_000, 15_000)),
    79:  ((30_000, 45_000), (15_000, 25_000)),   # 51-79 cm
    999: ((45_000, 60_000), (25_000, 40_000)),    # 80+ cm
}
_DESIGN_COST = (5_000, 15_000)

# Height-dependent per-letter market rates (RUB/letter)
# Thresholds are exclusive upper bounds so exact boundary goes to higher tier
_LETTER_RATES = [
    (30,  2_500,  4_000),   # up to 30 cm
    (50,  4_500,  8_000),   # 31-50 cm
    (79,  8_000, 16_000),   # 51-79 cm
    (120, 18_000, 35_000),  # 80-120 cm
]


class PricingResolver:
    """
    Resolves pricing information from retrieved documents.
    Implements the same logic as the Node.js analytics pipeline.
    """

    def __init__(self, settings):
        self.settings = settings
        self.bundle_band_factor = 0.2   # ±20% for bundle price bands
        self.analog_discount = 0.8       # confidence discount for analog-based prices

    @staticmethod
    def _get_service_costs(height_cm: float) -> tuple:
        """Return ((mount_min, mount_max), (frame_min, frame_max), (design_min, design_max)) for height."""
        for threshold in sorted(_SERVICE_COSTS.keys()):
            if height_cm <= threshold:
                mounting, frame = _SERVICE_COSTS[threshold]
                return mounting, frame, _DESIGN_COST
        # Fallback to largest
        mounting, frame = _SERVICE_COSTS[999]
        return mounting, frame, _DESIGN_COST

    @staticmethod
    def _get_letter_rate(height_cm: float, technology: str = "") -> tuple[float, float]:
        """Return (min_per_letter, max_per_letter) market rate for given height and technology."""
        from app.core.query_decomposer import TECH_MULTIPLIERS
        low, high = _LETTER_RATES[-1][1], _LETTER_RATES[-1][2]
        for max_h, lo, hi in _LETTER_RATES:
            if height_cm <= max_h:
                low, high = lo, hi
                break
        # Apply technology multiplier
        mult = TECH_MULTIPLIERS.get(technology, 1.0)
        return round(low * mult), round(high * mult)

    def _compute_under_key(self, res: PricingResolution, decomp) -> PricingResolution:
        """Compute total under-key price from market rates + service costs.

        Always uses market rates for letter component to avoid double-counting
        (bundle medians may already include some services).
        """
        if decomp is None or decomp.letter_count <= 0:
            return res

        height = getattr(decomp, 'height_cm', 0) or 0
        if height <= 0:
            return res

        letter_count = decomp.letter_count
        tech = getattr(decomp, 'technology', '') or ''
        mounting, frame, design = self._get_service_costs(height)

        # Always use market rates for consistent under-key calculation
        rate_min, rate_max = self._get_letter_rate(height, tech)
        letters_min = rate_min * letter_count
        letters_max = rate_max * letter_count

        total_min = letters_min + mounting[0] + frame[0] + design[0]
        total_max = letters_max + mounting[1] + frame[1] + design[1]

        res.total_under_key_min = round(total_min)
        res.total_under_key_max = round(total_max)
        res.price_breakdown = {
            "буквы": (round(letters_min), round(letters_max)),
            "монтаж": mounting,
            "каркас": frame,
            "дизайн": design,
        }
        return res

    def resolve(self, docs: list[dict], decomp=None) -> PricingResolution:
        """
        Resolve pricing from the top retrieved documents.

        Strategy: scan top-5 docs for the best pricing anchor (not just top-1).
        Priority: auto product > guided product > bundle > manual product with P50.
        """
        if not docs:
            res = PricingResolution(
                confidence="manual",
                flags=["Нет подходящих товаров/услуг в базе данных."],
            )
            if decomp and decomp.letter_count > 0 and getattr(decomp, 'height_cm', 0) > 0:
                tech = getattr(decomp, 'technology', '') or ''
                rate_min, rate_max = self._get_letter_rate(decomp.height_cm, tech)
                res.estimated_value = round((rate_min + rate_max) / 2 * decomp.letter_count)
                res.estimated_basis = "Рыночный ориентир по высоте букв"
                res.price_band_min = round(rate_min * decomp.letter_count)
                res.price_band_max = round(rate_max * decomp.letter_count)
                res.confidence = "guided"
                res = self._compute_under_key(res, decomp)
            return res

        # Scan top-10 docs for the best pricing anchor (product or bundle only)
        best_doc, best_priority = None, -1
        for doc in docs[:10]:
            payload = doc.get("payload", {})
            doc_type = payload.get("doc_type", "")

            # Skip non-pricing doc types entirely
            if doc_type in ("deal_profile", "service_composition", "timeline_fact",
                            "knowledge", "faq", "retrieval_support"):
                continue

            price_mode = payload.get("price_mode", "manual")
            confidence_tier = payload.get("confidence_tier", "low")
            has_p50 = payload.get("order_price_p50") is not None

            if doc_type == "product":
                if price_mode == "auto" and confidence_tier == "high":
                    priority = 4
                elif price_mode == "guided":
                    priority = 3
                elif has_p50:
                    priority = 1
                else:
                    priority = 0
            elif doc_type == "bundle":
                median = payload.get("median_deal_value") or payload.get("matched_value_median")
                priority = 2 if median else 0
            else:
                priority = -1

            if priority > best_priority:
                best_priority = priority
                best_doc = doc

        # No product/bundle found — try market-rate calculation if we have decomp
        if best_doc is None:
            res = PricingResolution(
                confidence="manual",
                flags=["Прямых ценовых данных (product/bundle) не найдено в контексте."],
            )
            if decomp and decomp.letter_count > 0 and getattr(decomp, 'height_cm', 0) > 0:
                tech = getattr(decomp, 'technology', '') or ''
                rate_min, rate_max = self._get_letter_rate(decomp.height_cm, tech)
                res.estimated_value = round((rate_min + rate_max) / 2 * decomp.letter_count)
                res.estimated_basis = "Рыночный ориентир по высоте букв"
                res.price_band_min = round(rate_min * decomp.letter_count)
                res.price_band_max = round(rate_max * decomp.letter_count)
                res.confidence = "guided"
                res = self._compute_under_key(res, decomp)
            return res

        payload = best_doc.get("payload", {})
        doc_type = payload.get("doc_type", "")

        if doc_type == "bundle":
            res = self._resolve_bundle(payload, best_doc, decomp=decomp)
        elif doc_type == "product":
            res = self._resolve_product(payload, best_doc, docs)
        else:
            return PricingResolution(
                confidence="manual",
                flags=["Прямых данных о цене не найдено. Обратитесь к менеджеру."],
                source_doc_id=payload.get("doc_id", ""),
                source_doc_type=doc_type,
            )

        # Compute total under-key price if we have letter/height info
        return self._compute_under_key(res, decomp)

    def _resolve_bundle(self, payload: dict, doc: dict, decomp=None) -> PricingResolution:
        """Resolve pricing for a bundle document with completeness and size adjustments."""
        import re
        median_value = payload.get("median_deal_value") or payload.get("matched_value_median")
        band_factor = self.bundle_band_factor

        # --- Completeness analysis ---
        sample_products = (payload.get("sample_products", "") or "").lower()
        bundle_direction = (payload.get("direction", "") or "").lower()
        missing_services = []
        completeness_surcharge = 1.0

        # Physical services surcharge only for physical products (not Дизайн, Полиграфия, etc.)
        _DIGITAL_DIRECTIONS = {"дизайн", "полиграфия", "брендбук", "фирменный стиль"}
        is_digital = any(d in bundle_direction for d in _DIGITAL_DIRECTIONS)

        if not is_digital:
            if not any(kw in sample_products for kw in ["монтаж вывески", "монтаж буквы", "монтаж конструкции"]):
                missing_services.append("монтаж")
                completeness_surcharge += 0.25
            if not any(kw in sample_products for kw in ["сварка каркаса", "каркас", "профильная труба"]):
                missing_services.append("каркас")
                completeness_surcharge += 0.15
        if not any(kw in sample_products for kw in ["дизайн", "макет"]):
            missing_services.append("дизайн")
            completeness_surcharge += 0.10

        # --- Size adjustment ---
        size_ratio = 1.0
        if decomp and decomp.letter_count > 0 and median_value and median_value > 0:
            bundle_title = payload.get("sample_title", "") or ""
            caps = re.findall(r"[А-ЯЁ]{3,}", bundle_title)
            bundle_letter_count = len(max(caps, key=len)) if caps else 0
            if bundle_letter_count > 0 and decomp.letter_count != bundle_letter_count:
                size_ratio = decomp.letter_count / bundle_letter_count
                size_ratio = max(0.5, min(size_ratio, 3.0))

        res = PricingResolution(
            confidence="guided",
            source_doc_id=payload.get("doc_id", ""),
            source_doc_type="bundle",
        )

        if median_value is not None and median_value > 0:
            adjusted = median_value * completeness_surcharge * size_ratio
            res.estimated_value = round(adjusted)
            basis_parts = [f"Медиана сделок ({payload.get('deal_count', '?')} сделок)"]
            if completeness_surcharge > 1.0:
                basis_parts.append(f"+{int((completeness_surcharge - 1) * 100)}% за {', '.join(missing_services)}")
            if size_ratio != 1.0:
                basis_parts.append(f"масштаб x{size_ratio:.1f} по кол-ву букв")
            res.estimated_basis = "; ".join(basis_parts)
            res.price_band_min = round(adjusted * (1 - band_factor))
            res.price_band_max = round(adjusted * (1 + band_factor))
            if missing_services:
                res.flags.append(f"Набор не включает: {', '.join(missing_services)}. Цена скорректирована вверх.")
            else:
                res.flags.append("Ориентировочная цена по аналогичным комплектам.")
        else:
            res.confidence = "manual"
            res.flags.append("Цена набора не определена. Требуется расчёт менеджером.")

        return res

    def _resolve_product(self, payload: dict, doc: dict, all_docs: list[dict]) -> PricingResolution:
        """Resolve pricing for a product document."""
        price_mode = payload.get("price_mode", "manual")
        confidence_tier = payload.get("confidence_tier", "low")
        manual_reason = payload.get("manual_review_reason", "")
        base_price = payload.get("current_base_price")
        recommended = payload.get("recommended_price")
        suggested_min = payload.get("suggested_min")
        suggested_max = payload.get("suggested_max")
        p25 = payload.get("order_price_p25")
        p50 = payload.get("order_price_p50")
        p75 = payload.get("order_price_p75")
        nearest_analogs = payload.get("nearest_analogs", []) or []

        res = PricingResolution(
            source_doc_id=payload.get("doc_id", ""),
            source_doc_type="product",
        )

        # Financial modifier — special case
        if manual_reason == "financial_modifier" or \
           any(kw in (payload.get("product_name", "") or "").lower()
               for kw in ["безнал", "скидка", "надбавка", "ндс"]):
            res.confidence = "manual"
            res.is_financial_modifier = True
            res.estimated_value = None
            res.flags.append(MANUAL_REVIEW_FLAGS["financial_modifier"])
            return res

        if price_mode == "auto" and confidence_tier == "high":
            # Auto pricing — highest confidence
            res.confidence = "auto"
            res.estimated_value = recommended or p50 or base_price
            res.estimated_basis = "Медиана цен заказов P50 (стабильная цена)"
            if p25 and p75:
                res.price_band_min = p25
                res.price_band_max = p75
            elif recommended and base_price:
                res.price_band_min = round(recommended * 0.9)
                res.price_band_max = round(recommended * 1.1)

        elif price_mode == "guided":
            # Guided pricing — sufficient data, needs confirmation
            res.confidence = "guided"
            res.estimated_value = recommended or p50 or base_price
            res.estimated_basis = "Ориентировочная цена по истории заказов"
            res.price_band_min = suggested_min or (p25 if p25 else None)
            res.price_band_max = suggested_max or (p75 if p75 else None)
            res.flags.append("Ориентировочная цена. Рекомендуется согласование с менеджером.")

        else:
            # Manual pricing — try to provide something useful
            res.confidence = "manual"

            # Add appropriate flag based on reason
            if manual_reason and manual_reason in MANUAL_REVIEW_FLAGS:
                res.flags.append(MANUAL_REVIEW_FLAGS[manual_reason])
            else:
                res.flags.append("Цена требует ручной оценки менеджером.")

            # Try cost_price + section markup if available (better than raw base_price)
            cost_price = payload.get("cost_price")
            markup_ratio = payload.get("markup_ratio")
            section_name = payload.get("section_name", "")
            if cost_price and cost_price > 0 and not res.estimated_value:
                if markup_ratio and markup_ratio > 1.0:
                    # Use actual markup ratio from deal history
                    res.estimated_value = round(cost_price * markup_ratio)
                    res.estimated_basis = f"Себестоимость × {markup_ratio:.1f} (по истории сделок)"
                    res.price_band_min = round(cost_price * max(markup_ratio * 0.85, 1.1))
                    res.price_band_max = round(cost_price * markup_ratio * 1.15)
                    res.confidence = "guided"
                elif section_name:
                    # Use section-level markup range
                    lo, hi = _DEFAULT_MARKUP
                    for sect_key, (s_lo, s_hi) in _SECTION_MARKUP_RANGES.items():
                        if sect_key.lower() in section_name.lower():
                            lo, hi = s_lo, s_hi
                            break
                    mid = (lo + hi) / 2
                    res.estimated_value = round(cost_price * mid)
                    res.estimated_basis = f"Себестоимость × {mid:.1f} (наценка по секции «{section_name[:30]}»)"
                    res.price_band_min = round(cost_price * lo)
                    res.price_band_max = round(cost_price * hi)
                    res.confidence = "guided"

            # Use base price as rough reference if available
            if base_price and base_price > 0 and not res.estimated_value:
                res.estimated_value = base_price
                res.estimated_basis = "Базовая цена каталога (требует уточнения)"
                if manual_reason == "high_price_variance":
                    # Wider band for high variance
                    res.price_band_min = round(base_price * 0.5)
                    res.price_band_max = round(base_price * 2.0)
                    res.risks.append("Фактические цены могут существенно отличаться от базовой.")
                else:
                    res.price_band_min = round(base_price * 0.8)
                    res.price_band_max = round(base_price * 1.3)

            # Analog fallback
            if nearest_analogs and (base_price is None or base_price <= 0):
                res = self._try_analog_pricing(res, nearest_analogs, all_docs)

        return res

    def _try_analog_pricing(self, res: PricingResolution, analogs: list[str], all_docs: list[dict]) -> PricingResolution:
        """Try to get price from nearest analog product."""
        import re
        for analog_raw in analogs[:3]:
            if not analog_raw:
                continue
            # Strip ID prefix like "13612:" from analog name
            analog_name = re.sub(r"^\d+:", "", analog_raw).strip()
            if not analog_name:
                continue
            for doc in all_docs[1:]:
                payload = doc.get("payload", {})
                product_name = payload.get("product_name") or ""
                # Skip self-references
                if payload.get("doc_id") == res.source_doc_id:
                    continue
                if product_name == analog_name or analog_name in product_name:
                    analog_mode = payload.get("price_mode", "manual")
                    analog_price = payload.get("recommended_price") or payload.get("order_price_p50")
                    if analog_price:
                        res.estimated_value = analog_price
                        res.estimated_basis = f"Цена аналога: {analog_name[:50]}"
                        if analog_mode not in ("auto", "guided"):
                            res.risks.append("Цена аналога ориентировочная (ручной режим).")
                        else:
                            res.risks.append(
                                f"Цена основана на аналоге ({analog_name[:40]}), не на прямых данных."
                            )
                        return res
        return res
