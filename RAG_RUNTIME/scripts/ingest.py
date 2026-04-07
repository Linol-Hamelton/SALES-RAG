#!/usr/bin/env python3
"""
Ingestion pipeline: reads analytics artifacts → builds canonical docs → writes JSONL.

Usage:
    python scripts/ingest.py [--doc-types product bundle policy support] [--verbose]
    python scripts/ingest.py  # builds all doc types
"""
import json
import re
import sys
import click
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Analytics source paths (read-only)
ANALYTICS_ROOT = PROJECT_ROOT.parent / "RAG_ANALYTICS" / "output"
OUTPUT_DIR = PROJECT_ROOT / "data"

PLACEHOLDERS = {
    "__MISSING_CLIENT__", "__MISSING_DIRECTION__",
    "__MANUAL_OR_UNCATEGORIZED__", "__NOT_APPLICABLE__"
}


def sanitize(value: Any) -> Any:
    """Remove pipeline placeholders and NaN values."""
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    if isinstance(value, str) and value in PLACEHOLDERS:
        return None
    return value


def safe_float(value: Any) -> float | None:
    """Convert to float safely."""
    try:
        v = float(value)
        return v if not pd.isna(v) else None
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    """Convert to int safely."""
    try:
        v = int(float(value))
        return v if not pd.isna(float(value)) else None
    except (TypeError, ValueError):
        return None


def load_csv(path: Path) -> pd.DataFrame:
    """Load semicolon-delimited UTF-8 CSV."""
    return pd.read_csv(path, sep=";", encoding="utf-8", dtype=str, low_memory=False)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Document Type A: product_docs ──────────────────────────────────────────

def build_product_docs() -> list[dict]:
    """Build product documents from product_facts + pricing_recommendations."""
    click.echo("Loading product_facts.csv...")
    pf = load_csv(ANALYTICS_ROOT / "facts" / "product_facts.csv")

    click.echo("Loading pricing_recommendations.csv...")
    pr = load_csv(ANALYTICS_ROOT / "pricing" / "pricing_recommendations.csv")

    # Join on PRODUCT_KEY
    merged = pf.merge(pr[["PRODUCT_KEY", "RECOMMENDED_PRICE", "SUGGESTED_MIN_PRICE",
                           "SUGGESTED_MAX_PRICE", "NEAREST_ANALOG_1", "NEAREST_ANALOG_2",
                           "NEAREST_ANALOG_3"]], on="PRODUCT_KEY", how="left")

    docs = []
    for _, row in merged.iterrows():
        product_key = str(row.get("PRODUCT_KEY", "")).strip()
        product_id = str(row.get("PRODUCT_ID", "")).strip()
        product_name = sanitize(row.get("PRODUCT_NAME")) or sanitize(row.get("CURRENT_CATALOG_NAME")) or ""
        direction = sanitize(row.get("NORMALIZED_DIRECTION")) or ""
        price_mode = sanitize(row.get("PRICE_MODE")) or "manual"
        confidence_tier = sanitize(row.get("CONFIDENCE_TIER")) or "low"
        order_rows = safe_int(row.get("ORDER_ROWS")) or 0
        offer_rows = safe_int(row.get("OFFER_ROWS")) or 0
        total_order_qty = safe_int(row.get("TOTAL_ORDER_QTY")) or 0
        price_ratio_range = safe_float(row.get("PRICE_RATIO_RANGE"))
        order_qty_p50 = safe_float(row.get("ORDER_QTY_P50"))
        base_price = safe_float(row.get("CURRENT_BASE_PRICE"))
        p25 = safe_float(row.get("ORDER_PRICE_P25"))
        p50 = safe_float(row.get("ORDER_PRICE_P50"))
        p75 = safe_float(row.get("ORDER_PRICE_P75"))
        recommended_price = safe_float(row.get("RECOMMENDED_PRICE"))
        suggested_min = safe_float(row.get("SUGGESTED_MIN_PRICE"))
        suggested_max = safe_float(row.get("SUGGESTED_MAX_PRICE"))
        qty_ladder = sanitize(row.get("OFFER_QTY_LADDER")) or ""
        manual_reason = sanitize(row.get("MANUAL_REVIEW_REASON")) or ""
        section_name = sanitize(row.get("SECTION_NAME")) or ""
        parent_section = sanitize(row.get("PARENT_SECTION")) or ""
        cost_price = safe_float(row.get("COST_PRICE"))
        markup_ratio = safe_float(row.get("MARKUP_RATIO"))

        analogs_raw = [
            sanitize(row.get("NEAREST_ANALOG_1")),
            sanitize(row.get("NEAREST_ANALOG_2")),
            sanitize(row.get("NEAREST_ANALOG_3")),
        ]
        # Filter out self-referencing analogs (e.g., "13612:Name" for product 13612)
        analogs = [
            a for a in analogs_raw
            if a and not a.startswith(f"{product_key}:")
        ]

        # Build searchable text (natural language for better embeddings)
        parts = [product_name]
        if section_name:
            parts.append(f"раздел {section_name}")
        if direction:
            parts.append(f"направление {direction}")
        if base_price is not None:
            parts.append(f"цена {base_price:.0f} руб")
        if recommended_price is not None:
            parts.append(f"рекомендованная {recommended_price:.0f} руб")
        if p50 is not None:
            parts.append(f"медиана заказов {p50:.0f} руб")
        if order_rows:
            parts.append(f"{order_rows} заказов")
        if total_order_qty:
            parts.append(f"всего заказано {total_order_qty} шт")
        parts.append(f"режим {price_mode}")
        if analogs:
            analog_names = [re.sub(r"^\d+:", "", a).strip() for a in analogs if a]
            if analog_names:
                parts.append(f"аналоги: {', '.join(analog_names)}")
        if qty_ladder:
            parts.append(f"количество: {qty_ladder}")
        searchable_text = ", ".join(parts)

        doc = {
            "doc_id": f"product_{product_key}",
            "doc_type": "product",
            "searchable_text": searchable_text,
            "metadata": {
                "product_key": product_key,
                "product_id": product_id,
                "product_name": product_name,
                "direction": direction,
                "price_mode": price_mode,
                "confidence_tier": confidence_tier,
                "current_base_price": base_price,
                "recommended_price": recommended_price,
                "suggested_min": suggested_min,
                "suggested_max": suggested_max,
                "order_rows": order_rows,
                "offer_rows": offer_rows,
                "total_order_qty": total_order_qty,
                "price_ratio_range": price_ratio_range,
                "order_qty_p50": order_qty_p50,
                "order_price_p25": p25,
                "order_price_p50": p50,
                "order_price_p75": p75,
                "manual_review_reason": manual_reason,
                "nearest_analogs": analogs,
                "qty_ladder": qty_ladder,
                "section_name": section_name,
                "parent_section": parent_section,
                "cost_price": cost_price,
                "markup_ratio": markup_ratio,
            },
            "provenance": {
                "sources": ["product_facts.csv", "pricing_recommendations.csv"],
                "generated_at": now_iso(),
            }
        }
        docs.append(doc)

    click.echo(f"  Built {len(docs)} product docs")
    return docs


# ─── Document Type B: bundle_docs ───────────────────────────────────────────

def build_bundle_docs() -> list[dict]:
    """Build bundle documents from bundle_facts + deal_facts + template_match_facts."""
    click.echo("Loading bundle_facts.csv...")
    bf = load_csv(ANALYTICS_ROOT / "facts" / "bundle_facts.csv")

    click.echo("Loading template_match_facts.csv...")
    tmf = load_csv(ANALYTICS_ROOT / "facts" / "template_match_facts.csv")

    # Build template match lookup keyed by TEMPLATE_BUNDLE_KEY
    template_lookup: dict[str, dict] = {}
    for _, row in tmf.iterrows():
        key = str(row.get("TEMPLATE_BUNDLE_KEY", "")).strip()
        if key:
            template_lookup[key] = {
                "exact_matches": safe_int(row.get("EXACT_ORDER_MATCH_COUNT")) or 0,
                "contained_matches": safe_int(row.get("CONTAINED_ORDER_MATCH_COUNT")) or 0,
                "matched_value_median": safe_float(row.get("MATCHED_ORDER_VALUE_MEDIAN")),
                "match_rate": safe_float(row.get("MATCH_RATE_PER_TEMPLATE_DEAL")),
                "sample_order_title": sanitize(row.get("SAMPLE_ORDER_TITLE")) or "",
            }

    docs = []
    for _, row in bf.iterrows():
        dataset = sanitize(row.get("DATASET")) or ""
        bundle_key = str(row.get("BUNDLE_KEY", "")).strip()
        product_count = safe_int(row.get("PRODUCT_COUNT")) or 0
        deal_count = safe_int(row.get("DEAL_COUNT")) or 0
        median_value = safe_float(row.get("MEDIAN_DEAL_VALUE"))
        avg_value = safe_float(row.get("AVG_DEAL_VALUE"))
        duration_p50 = safe_int(row.get("DURATION_P50"))
        direction = sanitize(row.get("PRIMARY_DIRECTION")) or ""
        sample_title = sanitize(row.get("SAMPLE_TITLE")) or ""
        sample_products = sanitize(row.get("SAMPLE_PRODUCTS")) or ""
        letter_height_cm = safe_int(row.get("LETTER_HEIGHT_CM"))
        description = sanitize(row.get("DESCRIPTION")) or ""

        # Get template match signals
        tmatch = template_lookup.get(bundle_key, {})

        # Build searchable text
        parts = []
        if sample_title:
            parts.append(f"Набор: {sample_title}")
        if direction:
            parts.append(direction)
        if dataset:
            parts.append(f"тип: {dataset}")
        if product_count:
            parts.append(f"продуктов: {product_count}")
        if median_value is not None:
            parts.append(f"медиана сделки: {median_value:.0f} руб")
        if duration_p50 is not None:
            parts.append(f"срок изготовления: ~{duration_p50} дней")
        if tmatch.get("contained_matches"):
            parts.append(f"встречалось в заказах: {tmatch['contained_matches']} раз")
        if sample_products:
            parts.append(f"состав: {sample_products}")
        if tmatch.get("sample_order_title"):
            parts.append(f"пример заказа: {tmatch['sample_order_title']}")
        if description:
            parts.append(description[:300])
        if letter_height_cm:
            parts.append(f"высота букв: {letter_height_cm} см")
        searchable_text = " | ".join(parts) if parts else f"bundle {bundle_key}"

        doc = {
            "doc_id": f"bundle_{bundle_key}",
            "doc_type": "bundle",
            "searchable_text": searchable_text,
            "metadata": {
                "bundle_key": bundle_key,
                "dataset_type": dataset,
                "product_count": product_count,
                "deal_count": deal_count,
                "median_deal_value": median_value,
                "avg_deal_value": avg_value,
                "direction": direction,
                "median_duration_days": duration_p50,
                "sample_title": sample_title,
                "sample_products": sample_products,
                "exact_matches": tmatch.get("exact_matches", 0),
                "contained_matches": tmatch.get("contained_matches", 0),
                "matched_value_median": tmatch.get("matched_value_median"),
                "match_rate": tmatch.get("match_rate"),
                "sample_order_title": tmatch.get("sample_order_title", ""),
                "letter_height_cm": letter_height_cm,
                "description": description[:500] if description else "",
            },
            "provenance": {
                "sources": ["bundle_facts.csv", "template_match_facts.csv"],
                "generated_at": now_iso(),
            }
        }
        docs.append(doc)

    click.echo(f"  Built {len(docs)} bundle docs")
    return docs


# ─── Document Type C: pricing_policy_docs ───────────────────────────────────

def build_pricing_policy_docs() -> list[dict]:
    """Build pricing policy documents from pricing modes and QA data."""
    click.echo("Loading pricing_summary.json and qa_report.json...")

    try:
        with open(ANALYTICS_ROOT / "pricing" / "pricing_summary.json", encoding="utf-8") as f:
            pricing_summary = json.load(f)
    except Exception:
        pricing_summary = {}

    try:
        with open(ANALYTICS_ROOT / "qa_report.json", encoding="utf-8") as f:
            qa_report = json.load(f)
    except Exception:
        qa_report = {}

    try:
        with open(ANALYTICS_ROOT / "kpis" / "kpi_summary.json", encoding="utf-8") as f:
            kpi_summary = json.load(f)
    except Exception:
        kpi_summary = {}

    docs = []

    # Policy docs for each pricing mode
    auto_count = pricing_summary.get("autoProducts", 23)
    guided_count = pricing_summary.get("guidedProducts", 473)
    manual_count = pricing_summary.get("manualProducts", 12130)

    policy_modes = [
        {
            "doc_id": "policy_auto_high",
            "price_mode": "auto",
            "confidence_tier": "high",
            "product_count": auto_count,
            "rule": "Стабильное ценообразование: ≥30 строк заказов, диапазон отклонения цены ≤15%. Рекомендованная цена = медиана P50 фактических цен заказов. Ценовой диапазон: P25–P75.",
            "when_to_use": "Когда продукт имеет много заказов с устойчивой ценой. Можно давать точную рекомендацию без согласования.",
            "searchable_text": "автоматическое ценообразование высокая уверенность стабильная цена медиана p50 p25 p75 много заказов",
        },
        {
            "doc_id": "policy_guided_medium",
            "price_mode": "guided",
            "confidence_tier": "medium",
            "product_count": guided_count,
            "rule": "Направляемое ценообразование: 10-29 строк заказов или диапазон отклонения 15-50%. Ориентировочная цена требует подтверждения менеджера.",
            "when_to_use": "Когда есть достаточно данных для ориентира, но высокая вариативность. Указать диапазон и рекомендовать уточнить у менеджера.",
            "searchable_text": "направляемое ценообразование средняя уверенность ориентировочная цена подтверждение менеджера диапазон",
        },
        {
            "doc_id": "policy_manual_review",
            "price_mode": "manual",
            "confidence_tier": "low",
            "product_count": manual_count,
            "rule": "Ручная проверка обязательна. Причины: недостаточно данных (insufficient_history), высокая вариативность (high_price_variance), финансовый модификатор (financial_modifier), нестандартный продукт (manual_or_missing_catalog).",
            "when_to_use": "Большинство продуктов (94%). Всегда сообщать клиенту что цена требует подтверждения.",
            "searchable_text": "ручная проверка низкая уверенность нестандартный продукт высокая вариативность мало данных финансовый модификатор",
        },
        {
            "doc_id": "policy_financial_modifier",
            "price_mode": "manual",
            "confidence_tier": "special",
            "product_count": None,
            "rule": "Финансовые модификаторы (Безнал 10%, скидки, надбавки за срочность) не являются самостоятельными продуктами. Не указывать их как основу для ценового расчёта.",
            "when_to_use": "Когда запрашивается цена на 'безнал', 'скидку', 'надбавку' — сообщать что это модификатор условий оплаты, а не товар.",
            "searchable_text": "безнал финансовый модификатор скидка надбавка условия оплаты не продукт",
        },
        {
            "doc_id": "policy_analog_pricing",
            "price_mode": "analog",
            "confidence_tier": "low",
            "product_count": pricing_summary.get("manualWithAnalogs", 11702),
            "rule": "Для продуктов без истории заказов используются ценовые аналоги (NEAREST_ANALOG_1/2/3). Цена аналога — ориентировочный референс, не точная рекомендация.",
            "when_to_use": "Когда нет прямых данных о цене — использовать аналог с пометкой 'цена основана на аналоге'.",
            "searchable_text": "аналог похожий продукт ориентировочная цена без истории цена аналога",
        },
    ]

    for policy in policy_modes:
        doc = {
            "doc_id": policy["doc_id"],
            "doc_type": "pricing_policy",
            "searchable_text": policy["searchable_text"],
            "metadata": {
                "price_mode": policy["price_mode"],
                "confidence_tier": policy["confidence_tier"],
                "product_count": policy.get("product_count"),
                "rule": policy["rule"],
                "when_to_use": policy.get("when_to_use", ""),
            },
            "provenance": {
                "sources": ["pricing_summary.json", "qa_report.json"],
                "generated_at": now_iso(),
            }
        }
        docs.append(doc)

    # QA caveat docs
    qa_issues = qa_report.get("issues", [])
    for issue in qa_issues:
        code = issue.get("code", "unknown")
        severity = issue.get("severity", "warning")
        description = issue.get("description", "")
        doc = {
            "doc_id": f"policy_qa_{code}",
            "doc_type": "pricing_policy",
            "searchable_text": f"предупреждение данные {code} {description}",
            "metadata": {
                "price_mode": "caveat",
                "confidence_tier": severity,
                "qa_code": code,
                "description": description,
                "mitigation": issue.get("mitigation", ""),
            },
            "provenance": {
                "sources": ["qa_report.json"],
                "generated_at": now_iso(),
            }
        }
        docs.append(doc)

    # KPI summary doc
    if kpi_summary:
        kpi_text = (
            f"Всего заказов: {kpi_summary.get('orderDeals', 0)}, "
            f"предложений: {kpi_summary.get('offerDeals', 0)}, "
            f"клиентов: {kpi_summary.get('knownOrderClients', 0)}, "
            f"повторных клиентов: {kpi_summary.get('repeatOrderClientShare', 0)*100:.0f}%, "
            f"точных совпадений шаблонов: {kpi_summary.get('exactTemplateMatches', 0)}"
        )
        docs.append({
            "doc_id": "policy_kpi_summary",
            "doc_type": "pricing_policy",
            "searchable_text": f"сводка показатели {kpi_text}",
            "metadata": {
                "price_mode": "summary",
                "confidence_tier": "info",
                "kpi_data": kpi_summary,
                "description": kpi_text,
            },
            "provenance": {
                "sources": ["kpi_summary.json"],
                "generated_at": now_iso(),
            }
        })

    click.echo(f"  Built {len(docs)} pricing policy docs")
    return docs


# ─── Document Type D: retrieval_support_docs ────────────────────────────────

def build_retrieval_support_docs() -> list[dict]:
    """Build direction aliases and terminology docs."""
    DIRECTION_ALIASES = {
        "Цех": {
            "description": "Изготовление и монтаж наружной рекламы и вывесок",
            "keywords": [
                "вывеска", "световые буквы", "объёмные буквы", "лайтбокс",
                "монтаж вывески", "демонтаж", "сборка букв", "оклейка пленкой",
                "фасадная реклама", "световая конструкция", "неоновая вывеска",
                "led буквы", "короб", "подсветка", "конструкция",
            ]
        },
        "Сольвент": {
            "description": "Широкоформатная печать на баннерной ткани и пленке",
            "keywords": [
                "баннер", "банер", "растяжка", "широкоформатная печать",
                "рекламный щит", "виниловый баннер", "баннерная ткань",
                "сольвентная печать", "интерьерный баннер", "ролл-ап",
            ]
        },
        "Печатная": {
            "description": "Стандартная полиграфия: листовки, буклеты, наклейки",
            "keywords": [
                "листовка", "флаер", "буклет", "визитка", "открытка",
                "наклейка", "самоклейка", "этикетка", "стикер",
                "цифровая печать", "а4 печать", "а3 печать", "брошюра",
            ]
        },
        "Дизайн": {
            "description": "Дизайн-услуги: макеты, иллюстрации, анимация, съемка",
            "keywords": [
                "дизайн макет", "верстка", "иллюстрация", "логотип",
                "фирменный стиль", "анимация", "видеоролик", "съемка",
                "презентация", "слайд", "дизайн сайта", "почасовой дизайн",
                "художник", "дизайнер час",
            ]
        },
        "РИК": {
            "description": "Размещение и изготовление рекламных конструкций",
            "keywords": [
                "рик", "размещение рекламы", "аренда щита", "билборд",
                "биллборд", "рекламная конструкция", "рекламное место",
                "реклама на щите", "наружная реклама", "оператор наружной",
            ]
        },
        "Мерч": {
            "description": "Мерчандайзинг: брендированные товары, сувениры",
            "keywords": [
                "мерч", "сувенир", "брендированный товар", "промо продукция",
                "корпоративный подарок", "футболка с логотипом", "кружка",
                "ручка с логотипом", "блокнот",
            ]
        },
        "Безнал": {
            "description": "Финансовый модификатор (не продукт): надбавка за безналичный расчёт",
            "keywords": [
                "безналичный расчёт", "безнал надбавка", "безнал 10%",
                "оплата по счету", "финансовый модификатор",
            ]
        },
    }

    docs = []
    for direction, info in DIRECTION_ALIASES.items():
        keywords_text = ", ".join(info["keywords"])
        searchable_text = f"{direction} | {info['description']} | {keywords_text}"
        doc = {
            "doc_id": f"support_direction_{direction}",
            "doc_type": "retrieval_support",
            "searchable_text": searchable_text,
            "metadata": {
                "direction": direction,
                "description": info["description"],
                "keywords": info["keywords"],
            },
            "provenance": {
                "sources": ["domain_knowledge"],
                "generated_at": now_iso(),
            }
        }
        docs.append(doc)

    # Common request pattern docs
    request_patterns = [
        {
            "id": "request_quote_billboard",
            "text": "световая вывеска под ключ монтаж сборка оклейка буквы подсветка",
            "direction": "Цех",
            "pattern": "sign_assembly",
        },
        {
            "id": "request_print_run",
            "text": "тираж листовок флаеров печать цифровая полиграфия",
            "direction": "Печатная",
            "pattern": "print_run",
        },
        {
            "id": "request_design_hourly",
            "text": "дизайн час работа дизайнера макет верстка",
            "direction": "Дизайн",
            "pattern": "design_hourly",
        },
        {
            "id": "request_banner_large",
            "text": "баннер размер метр широкоформатная печать растяжка",
            "direction": "Сольвент",
            "pattern": "banner_large_format",
        },
    ]
    for pattern in request_patterns:
        doc = {
            "doc_id": f"support_{pattern['id']}",
            "doc_type": "retrieval_support",
            "searchable_text": pattern["text"],
            "metadata": {
                "direction": pattern["direction"],
                "pattern_type": pattern["pattern"],
            },
            "provenance": {
                "sources": ["domain_knowledge"],
                "generated_at": now_iso(),
            }
        }
        docs.append(doc)

    click.echo(f"  Built {len(docs)} retrieval support docs")
    return docs


# ─── Document Type E: deal_profile_docs ──────────────────────────────────────

def build_deal_profile_docs() -> list[dict]:
    """Build deal profile documents from deal_profiles.csv — real deal cases."""
    profiles_path = ANALYTICS_ROOT / "facts" / "deal_profiles.csv"
    if not profiles_path.exists():
        click.echo("  deal_profiles.csv not found, skipping")
        return []

    click.echo("Loading deal_profiles.csv...")
    df = load_csv(profiles_path)

    image_lookup = {}
    deals_json_path = PROJECT_ROOT.parent / "RAG_DATA" / "deals.json"
    if deals_json_path.exists():
        with open(deals_json_path, "r", encoding="utf-8") as f:
            for d in json.load(f):
                if d.get("ID"):
                    image_lookup[str(d["ID"])] = d.get("IMAGE_URLS", [])

    docs = []
    for _, row in df.iterrows():
        deal_id = sanitize(row.get("DEAL_ID")) or ""
        title = sanitize(row.get("TITLE")) or ""
        direction = sanitize(row.get("DIRECTION")) or ""
        line_total = safe_float(row.get("LINE_TOTAL"))
        duration = safe_int(row.get("DEAL_DURATION_DAYS"))
        description = sanitize(row.get("DESCRIPTION")) or ""
        comments = sanitize(row.get("COMMENTS")) or ""
        product_count = safe_int(row.get("UNIQUE_PRODUCT_COUNT")) or 0
        direction_count = safe_int(row.get("DIRECTION_COUNT")) or 1
        direction_breakdown = sanitize(row.get("DIRECTION_BREAKDOWN")) or ""
        component_summary = sanitize(row.get("COMPONENT_SUMMARY")) or ""
        materials = sanitize(row.get("MATERIALS")) or ""
        sample_products = sanitize(row.get("SAMPLE_PRODUCTS")) or ""

        # Build searchable text
        image_urls = image_lookup.get(deal_id, [])

        parts = [title]
        if image_urls:
            parts.append("включает фотографии проекта")
        if direction:
            parts.append(f"направление {direction}")
        if line_total is not None:
            parts.append(f"стоимость {line_total:.0f} руб")
        if duration is not None:
            parts.append(f"срок {duration} дней")
        if direction_breakdown:
            parts.append(f"разбивка: {direction_breakdown[:300]}")
        if materials:
            parts.append(f"материалы: {materials}")
        if component_summary:
            parts.append(f"состав: {component_summary[:500]}")
        if description:
            parts.append(f"описание: {description[:300]}")
        if comments:
            parts.append(f"комментарий: {comments[:300]}")
        searchable_text = ", ".join(parts)

        doc = {
            "doc_id": f"deal_profile_{deal_id}",
            "doc_type": "deal_profile",
            "searchable_text": searchable_text,
            "metadata": {
                "deal_id": deal_id,
                "title": title,
                "direction": direction,
                "line_total": line_total,
                "deal_duration_days": duration,
                "description": description,
                "comments": comments,
                "unique_product_count": product_count,
                "direction_count": direction_count,
                "direction_breakdown": direction_breakdown,
                "component_summary": component_summary,
                "materials": materials,
                "sample_products": sample_products,
                "image_urls": image_urls,
            },
            "provenance": {
                "sources": ["deal_profiles.csv", "deals.json"],
                "generated_at": now_iso(),
            }
        }
        docs.append(doc)

    click.echo(f"  Built {len(docs)} deal profile docs")
    return docs


# ─── Document Type F: service_composition_docs ──────────────────────────────

def build_service_composition_docs() -> list[dict]:
    """Build service composition reference from service_composition.csv."""
    comp_path = ANALYTICS_ROOT / "facts" / "service_composition.csv"
    if not comp_path.exists():
        click.echo("  service_composition.csv not found, skipping")
        return []

    click.echo("Loading service_composition.csv...")
    df = load_csv(comp_path)

    docs = []
    for _, row in df.iterrows():
        direction = sanitize(row.get("DIRECTION")) or ""
        deal_count = safe_int(row.get("DEAL_COUNT")) or 0
        median_value = safe_float(row.get("MEDIAN_VALUE"))
        avg_value = safe_float(row.get("AVG_VALUE"))
        core_products = sanitize(row.get("CORE_PRODUCTS")) or ""
        optional_products = sanitize(row.get("OPTIONAL_PRODUCTS")) or ""
        materials = sanitize(row.get("MATERIALS")) or ""
        cross_directions = sanitize(row.get("CROSS_DIRECTIONS")) or ""

        parts = [f"Типовой состав услуг направления {direction}"]
        if deal_count:
            parts.append(f"на основе {deal_count} завершённых сделок")
        if median_value is not None:
            parts.append(f"медиана стоимости {median_value:.0f} руб")
        if core_products:
            parts.append(f"основные компоненты: {core_products[:600]}")
        if optional_products:
            parts.append(f"дополнительные компоненты: {optional_products[:400]}")
        if materials:
            parts.append(f"материалы: {materials}")
        if cross_directions:
            parts.append(f"часто сочетается с: {cross_directions}")
        searchable_text = ", ".join(parts)

        doc = {
            "doc_id": f"service_comp_{direction}",
            "doc_type": "service_composition",
            "searchable_text": searchable_text,
            "metadata": {
                "direction": direction,
                "deal_count": deal_count,
                "median_value": median_value,
                "avg_value": avg_value,
                "core_products": core_products,
                "optional_products": optional_products,
                "materials": materials,
                "cross_directions": cross_directions,
            },
            "provenance": {
                "sources": ["service_composition.csv"],
                "generated_at": now_iso(),
            }
        }
        docs.append(doc)

    click.echo(f"  Built {len(docs)} service composition docs")
    return docs


# ─── Document Type G: timeline_docs ─────────────────────────────────────────

def build_timeline_docs() -> list[dict]:
    """Build timeline fact documents from timeline_facts.csv."""
    timeline_path = ANALYTICS_ROOT / "facts" / "timeline_facts.csv"
    if not timeline_path.exists():
        click.echo("  timeline_facts.csv not found, skipping")
        return []

    click.echo("Loading timeline_facts.csv...")
    df = load_csv(timeline_path)

    docs = []
    for _, row in df.iterrows():
        group_type = sanitize(row.get("GROUP_TYPE")) or ""
        group_key = sanitize(row.get("GROUP_KEY")) or ""
        deal_count = safe_int(row.get("DEAL_COUNT")) or 0
        d_min = safe_int(row.get("DURATION_MIN"))
        d_p25 = safe_int(row.get("DURATION_P25"))
        d_p50 = safe_int(row.get("DURATION_P50"))
        d_p75 = safe_int(row.get("DURATION_P75"))
        d_max = safe_int(row.get("DURATION_MAX"))
        sample_title = sanitize(row.get("SAMPLE_TITLE")) or ""

        if group_type == "direction":
            label = f"направление {group_key}"
        else:
            label = f"набор {sample_title}" if sample_title else f"bundle {group_key}"

        parts = [f"Срок изготовления {label}"]
        if d_p50 is not None:
            parts.append(f"медиана {d_p50} дней")
        if d_p25 is not None and d_p75 is not None:
            parts.append(f"диапазон {d_p25}–{d_p75} дней")
        if d_min is not None and d_max is not None:
            parts.append(f"от {d_min} до {d_max} дней")
        parts.append(f"на основе {deal_count} заказов")
        searchable_text = ", ".join(parts)

        doc = {
            "doc_id": f"timeline_{group_type}_{group_key[:80]}",
            "doc_type": "timeline_fact",
            "searchable_text": searchable_text,
            "metadata": {
                "group_type": group_type,
                "group_key": group_key,
                "deal_count": deal_count,
                "duration_min": d_min,
                "duration_p25": d_p25,
                "duration_p50": d_p50,
                "duration_p75": d_p75,
                "duration_max": d_max,
                "sample_title": sample_title,
            },
            "provenance": {
                "sources": ["timeline_facts.csv"],
                "generated_at": now_iso(),
            }
        }
        docs.append(doc)

    click.echo(f"  Built {len(docs)} timeline docs")
    return docs


# ─── Document Type H: photo_analysis_docs ────────────────────────────────────

MAX_CHUNK_CHARS = 1800  # BGE-M3 optimal ~512 tokens ≈ 1500-1800 chars


def _chunk_photo_doc(doc: dict) -> list[dict]:
    """Split a long photo_analysis doc into semantic chunks.

    Chunks: 1) header+description+ROI  2) bundle composition  3) vision analysis
    Each chunk shares the same metadata but gets its own searchable_text and doc_id.
    """
    text = doc.get("searchable_text", "")
    if len(text) <= MAX_CHUNK_CHARS:
        return [doc]

    # Split by known section markers
    sections = []
    current_label = "header"
    current_lines = []

    for line in text.split("\n"):
        if line.startswith("Описание товара/комплекта:"):
            if current_lines:
                sections.append((current_label, "\n".join(current_lines).strip()))
            current_label = "description"
            current_lines = [line]
        elif line.startswith("Состав комплекта"):
            if current_lines:
                sections.append((current_label, "\n".join(current_lines).strip()))
            current_label = "bundle"
            current_lines = [line]
        elif line.startswith("Визуальный анализ фотографий:"):
            if current_lines:
                sections.append((current_label, "\n".join(current_lines).strip()))
            current_label = "vision"
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_label, "\n".join(current_lines).strip()))

    # If we couldn't split meaningfully, return as-is
    if len(sections) <= 1:
        return [doc]

    # Sub-chunk long vision sections by "Фото N:" markers
    expanded = []
    for label, text_part in sections:
        if label == "vision" and len(text_part) > MAX_CHUNK_CHARS:
            # Split by photo markers
            photo_parts = re.split(r"(?=(?:^|\n)Фото \d+:)", text_part)
            # First part is the header "Визуальный анализ фотографий:"
            for j, pp in enumerate(photo_parts):
                pp = pp.strip()
                if pp:
                    expanded.append((f"vision_photo{j}", pp))
        else:
            expanded.append((label, text_part))
    sections = expanded

    # Merge small sections: header+description stay together if short
    merged = []
    buf_label = ""
    buf_text = ""
    for label, text_part in sections:
        if not text_part or len(text_part) < 30:
            continue
        if buf_text and len(buf_text) + len(text_part) <= MAX_CHUNK_CHARS:
            buf_text += "\n\n" + text_part
            buf_label = buf_label or label
        else:
            if buf_text:
                merged.append((buf_label, buf_text))
            buf_label = label
            buf_text = text_part
    if buf_text:
        merged.append((buf_label, buf_text))

    if len(merged) <= 1:
        return [doc]

    # Create chunked docs
    base_id = doc["doc_id"]
    chunks = []
    for i, (label, chunk_text) in enumerate(merged):
        chunk_doc = {
            "doc_id": f"{base_id}_chunk{i}",
            "doc_type": doc["doc_type"],
            "searchable_text": chunk_text,
            "metadata": dict(doc.get("metadata", {})),
            "provenance": doc.get("provenance", {}),
        }
        chunk_doc["metadata"]["chunk_index"] = i
        chunk_doc["metadata"]["chunk_label"] = label
        chunk_doc["metadata"]["chunk_total"] = len(merged)
        chunks.append(chunk_doc)

    return chunks


def build_photo_analysis_docs() -> list[dict]:
    """Load photo analysis docs, filter empties, chunk long texts."""
    jsonl_path = OUTPUT_DIR / "photo_analysis_raw.jsonl"
    if not jsonl_path.exists():
        click.echo("  photo_analysis_raw.jsonl not found, skipping (run vision_analysis_local.py first)")
        return []

    raw_docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_docs.append(json.loads(line))

    # Filter: skip docs with no products AND no deal_total (empty deals)
    filtered = []
    skipped = 0
    for doc in raw_docs:
        meta = doc.get("metadata", {})
        has_products = bool(meta.get("products"))
        has_total = bool(meta.get("deal_total"))
        if not has_products and not has_total:
            skipped += 1
            continue
        filtered.append(doc)

    # Chunk long texts
    docs = []
    chunked_count = 0
    for doc in filtered:
        chunks = _chunk_photo_doc(doc)
        if len(chunks) > 1:
            chunked_count += 1
        docs.extend(chunks)

    click.echo(f"  Loaded {len(raw_docs)} raw, filtered {skipped} empty, "
               f"chunked {chunked_count} long, {len(docs)} final docs")
    return docs


# ─── Main CLI ────────────────────────────────────────────────────────────────

DOC_BUILDERS = {
    "product": ("product_docs.jsonl", build_product_docs),
    "bundle": ("bundle_docs.jsonl", build_bundle_docs),
    "policy": ("pricing_policy_docs.jsonl", build_pricing_policy_docs),
    "support": ("retrieval_support_docs.jsonl", build_retrieval_support_docs),
    "deal_profile": ("deal_profile_docs.jsonl", build_deal_profile_docs),
    "service_comp": ("service_composition_docs.jsonl", build_service_composition_docs),
    "timeline": ("timeline_docs.jsonl", build_timeline_docs),
    "photo_analysis": ("photo_analysis_docs.jsonl", build_photo_analysis_docs),
}

ALL_DOC_TYPES = list(DOC_BUILDERS.keys())


@click.command()
@click.option("--doc-types", "-t", multiple=True,
              type=click.Choice(ALL_DOC_TYPES),
              default=ALL_DOC_TYPES,
              help="Doc types to build (default: all)")
@click.option("--output-dir", default=str(OUTPUT_DIR), help="Output directory for JSONL files")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def main(doc_types, output_dir, verbose):
    """Build canonical RAG documents from analytics artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    generated = now_iso()

    for doc_type in doc_types:
        filename, builder = DOC_BUILDERS[doc_type]
        click.echo(f"\n[{doc_type}] Building {filename}...")

        try:
            docs = builder()
            out_file = output_path / filename
            with open(out_file, "w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            click.echo(f"  -> Wrote {len(docs)} docs to {out_file}")
            total_docs += len(docs)
        except Exception as e:
            click.echo(f"  ERROR building {doc_type}: {e}", err=True)
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    click.echo(f"\nIngestion complete: {total_docs} total docs written to {output_dir}/")


if __name__ == "__main__":
    main()
