#!/usr/bin/env python3
"""
Vision Analysis Script for SALES_RAG.
Downloads image URLs from deal profiles and uses Vision API (OpenAI-compatible)
to extract technical details, pricing hints, and unique value propositions for RAG embeddings.

Enriches each document with product composition from offers.csv (GOOD_IDs, materials,
pricing, sections) for precise matching in RAG retrieval.

Usage:
    python scripts/vision_analysis.py [--limit N] [--migrate]
"""
import csv
import json
import base64
import requests
import click
import time
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).parent.parent
RAG_DATA = PROJECT_ROOT.parent / "RAG_DATA"
DEALS_JSON_PATH = RAG_DATA / "deals.json"
OFFERS_CSV_PATH = RAG_DATA / "offers.csv"
GOODS_CSV_PATH = RAG_DATA / "goods.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "photo_analysis_docs.jsonl"
ENV_FILE = PROJECT_ROOT / "configs" / ".env"

# URL path segments that always return 404 — skip instantly
BLACKLISTED_URL_CATEGORIES = frozenset([
    "menu", "Billboards", "Kartochka_tovara", "beidj", "krujky", "otkritka",
])


def _is_url_blacklisted(url: str) -> bool:
    """Check if URL belongs to a known-broken image category."""
    return any(f"/images/{cat}/" in url for cat in BLACKLISTED_URL_CATEGORIES)


def load_env() -> dict[str, str]:
    """Load .env file into a dict."""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


# ---------------------------------------------------------------------------
# Offers / Goods enrichment
# ---------------------------------------------------------------------------

def load_offers_by_deal(csv_path: Path) -> dict[str, dict]:
    """Load offers.csv and group product rows by deal ID.

    Returns: {deal_id: {
        "products": [{product_id, good_id, product_name, price, quantity,
                       section_name, parent_section}, ...],
        "direction": str,
        "opportunity": float,
        "description": str,     # marketing/technical description of the deal
        "company_id": str,
        "title": str,
    }}
    """
    deals: dict[str, dict] = {}
    if not csv_path.exists():
        return deals

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            did = row.get("ID", "").strip()
            if not did:
                continue

            if did not in deals:
                deals[did] = {
                    "products": [],
                    "direction": row.get("DIRECTION", "").strip() or row.get("Направление", "").strip(),
                    "opportunity": _safe_float(row.get("OPPORTUNITY", "")),
                    "description": row.get("DESCRIPTION", "").strip(),
                    "company_id": row.get("COMPANY_ID", "").strip(),
                    "title": row.get("TITLE", "").strip(),
                }

            product = {
                "product_id": row.get("PRODUCT_ID", "").strip(),
                "good_id": row.get("GOOD_ID", "").strip(),
                "product_name": row.get("PRODUCT_NAME", "").strip(),
                "price": _safe_float(row.get("PRICE", "")),
                "quantity": _safe_float(row.get("QUANTITY", "")),
                "section_name": row.get("SECTION_NAME", "").strip(),
                "parent_section": row.get("PARENT_SECTION", "").strip(),
            }
            deals[did]["products"].append(product)

    return deals


def load_goods_index(csv_path: Path) -> dict[str, dict]:
    """Load goods.csv → {PRODUCT_ID: {name, base_price, cost_price, section, parent, direction}}.

    Provides catalog-level data for each product: pricing economics,
    category hierarchy, and direction — enriching beyond what offers.csv has.
    """
    index: dict[str, dict] = {}
    if not csv_path.exists():
        return index
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            pid = row.get("PRODUCT_ID", "").strip()
            if not pid:
                continue
            index[pid] = {
                "name": row.get("NAME", "").strip() or row.get("PRODUCT_NAME", "").strip(),
                "base_price": _safe_float(row.get("BASE_PRICE", "")),
                "cost_price": _safe_float(row.get("COST_PRICE", "")),
                "coefficient": _safe_float(row.get("COEFFICIENT", "")),
                "section_name": row.get("SECTION_NAME", "").strip(),
                "parent_section": row.get("PARENT_SECTION", "").strip(),
                "direction": row.get("Направление", "").strip(),
            }
    return index


def _safe_float(val: str) -> float:
    try:
        return float(val.replace(",", ".").strip())
    except (ValueError, AttributeError):
        return 0.0


def enrich_metadata(deal_id: str, deal_title: str, urls: list[str],
                    analysis: str, model: str,
                    offers_map: dict, goods_index: dict) -> dict:
    """Build a fully enriched photo_analysis doc with product composition.

    Cross-references:
    - offers.csv: deal-level (DESCRIPTION, DIRECTION, OPPORTUNITY) + per-product
      (PRODUCT_ID, GOOD_ID, PRODUCT_NAME, PRICE, QUANTITY, SECTION_NAME)
    - goods.csv: catalog-level per product (BASE_PRICE, COST_PRICE, SECTION hierarchy, direction)
    """

    deal_data = offers_map.get(deal_id, {})
    products = deal_data.get("products", []) if isinstance(deal_data, dict) else deal_data

    # --- Deal description from offers.csv (marketing/technical text) ---
    deal_description = ""
    if isinstance(deal_data, dict):
        deal_description = deal_data.get("description", "")

    # --- Direction (from offers or title heuristics) ---
    direction = ""
    if isinstance(deal_data, dict):
        direction = deal_data.get("direction", "")
    if not direction and products:
        # Fallback: direction from first product's goods.csv catalog entry
        for p in products:
            cat = goods_index.get(p.get("product_id", ""), {})
            if isinstance(cat, dict) and cat.get("direction"):
                direction = cat["direction"]
                break
    if not direction:
        direction = _detect_direction_from_title(deal_title)

    # --- Deal total ---
    deal_total = 0.0
    if isinstance(deal_data, dict):
        deal_total = deal_data.get("opportunity", 0.0)
    if not deal_total and products:
        deal_total = sum(p["price"] * p["quantity"] for p in products)

    # --- Bundle key (pipe-separated PRODUCT_IDs) ---
    product_ids = []
    for p in products:
        pid = p.get("product_id", "")
        if pid and pid not in product_ids:
            product_ids.append(pid)
    bundle_key = "|".join(product_ids)

    # --- Product composition: enriched with goods.csv catalog data ---
    composition_lines = []
    categories = set()
    catalog_products = []  # structured per-product data for metadata

    for p in products:
        pid = p.get("product_id", "")
        gid = p.get("good_id", "")
        name = p["product_name"]
        qty = p["quantity"]
        price = p["price"]
        section = p["section_name"]
        parent = p["parent_section"]
        total = price * qty

        # Goods.csv catalog data
        cat = goods_index.get(pid, {})
        cat_data = cat if isinstance(cat, dict) else {}
        base_price = cat_data.get("base_price", 0.0)
        cost_price = cat_data.get("cost_price", 0.0)
        cat_section = cat_data.get("section_name", "") or section
        cat_parent = cat_data.get("parent_section", "") or parent

        if cat_section:
            categories.add(cat_section)
        if cat_parent:
            categories.add(cat_parent)

        # Composition line with GOOD_ID and category
        cat_label = f" [{cat_section}]" if cat_section else ""
        if total > 0:
            composition_lines.append(
                f"  • {name}{cat_label}: {qty:g} × {price:,.0f} = {total:,.0f} руб"
                f" (GOOD_ID:{gid})"
            )
        else:
            composition_lines.append(f"  • {name}{cat_label}: {qty:g} шт (GOOD_ID:{gid})")

        # Structured product entry for metadata
        product_entry = {
            "product_id": pid,
            "good_id": gid,
            "name": name,
            "price": price,
            "quantity": qty,
            "total": round(total, 2),
            "section": cat_section,
            "parent_section": cat_parent,
        }
        if base_price:
            product_entry["catalog_base_price"] = base_price
        # NOTE: cost_price (себестоимость) — конфиденциальная информация, НЕ включаем в RAG
        catalog_products.append(product_entry)

    composition_text = "\n".join(composition_lines) if composition_lines else ""
    product_count = len(products)

    # --- Extract fields from vision analysis ---
    product_type = _extract_structured_field(analysis, "тип изделия") or ""
    visible_text = _extract_structured_field(analysis, "текст на изделии") or ""
    dimensions = _extract_structured_field(analysis, "размеры (оценка)") or _extract_structured_field(analysis, "размеры") or ""
    application = _extract_structured_field(analysis, "применение") or ""
    raw_practical_value = _extract_structured_field(analysis, "практическая ценность") or ""
    practical_value = _validate_practical_value(raw_practical_value, direction)

    # --- ROI/ROMI from benchmarks table (by direction, NOT from model) ---
    roi_data = ROI_BENCHMARKS.get(direction, {})
    roi_description = roi_data.get("description", "")
    roi_romi_avg = roi_data.get("romi_avg", 0)

    # --- Searchable text: structured, cross-referenced, information-dense ---
    # NOTE: bloat fields removed (Цвета, Шрифт, Форма, Подсветка, Материалы, etc.)
    search_parts = [
        f"[Фото-анализ] {deal_title}",
        f"Сделка: ID {deal_id} из offers.csv",
    ]
    if direction:
        search_parts.append(f"Направление: {direction}")
    if product_type:
        search_parts.append(f"Тип изделия: {product_type}")
    if visible_text:
        search_parts.append(f"Текст на изделии: {visible_text}")
    if dimensions:
        search_parts.append(f"Размеры: {dimensions}")
    if application:
        search_parts.append(f"Применение: {application}")
    if deal_total > 0:
        search_parts.append(f"Стоимость комплекта: {deal_total:,.0f} руб")
    if practical_value:
        search_parts.append(f"Практическая ценность: {practical_value}")

    # ROI/ROMI from benchmarks (script-generated, not from model)
    if roi_description:
        search_parts.append(f"ROI/ROMI ({direction}): ROMI ~{roi_romi_avg}%. {roi_description}")

    # Marketing description from offers.csv DESCRIPTION field
    if deal_description:
        search_parts.append("")
        search_parts.append("Описание товара/комплекта:")
        search_parts.append(deal_description[:1500])

    # Product composition with goods.csv cross-references
    if product_count > 0:
        search_parts.append("")
        search_parts.append(f"Состав комплекта ({product_count} товаров из goods.csv):")
        search_parts.append(composition_text)

    # Vision API analysis
    search_parts.append("")
    search_parts.append("Визуальный анализ фотографий:")
    search_parts.append(analysis)

    searchable_text = "\n".join(search_parts)

    return {
        "doc_id": f"photo_vision_{deal_id}",
        "doc_type": "photo_analysis",
        "searchable_text": searchable_text,
        "metadata": {
            # --- Связь с offers.csv ---
            "deal_id": deal_id,
            "deal_title": deal_title,
            "deal_description": deal_description[:2000] if deal_description else "",
            "direction": direction,
            "deal_total": round(deal_total, 2) if deal_total else None,
            # --- Связь с goods.csv ---
            "bundle_key": bundle_key,
            "product_count": product_count,
            "products": catalog_products,  # full per-product data with GOOD_ID + catalog prices
            "categories": sorted(categories),
            # --- Визуальные характеристики (из Vision API) ---
            "product_type": product_type,
            "visible_text": visible_text,
            "dimensions": dimensions,
            "application": application,
            "practical_value": practical_value,
            # --- ROI/ROMI (из knowledge_docs, не от модели) ---
            "roi_romi_avg": roi_romi_avg if roi_romi_avg else None,
            "roi_description": roi_description if roi_description else None,
            # --- Фотографии ---
            "image_urls": urls,
            "photo_count": len(urls),
            # --- Исходный анализ Vision API ---
            "vision_analysis": analysis,
        },
        "provenance": {
            "sources": urls,
            "vision_model": model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _detect_direction_from_title(title: str) -> str:
    """Heuristic direction detection from deal title."""
    t = title.lower()
    if any(kw in t for kw in ["буквы", "вывеск", "короб", "неон", "табличк",
                                "штендер", "панель", "кронштейн", "крышн"]):
        return "Цех"
    if any(kw in t for kw in ["монтаж", "демонтаж", "установк"]):
        return "РИК"
    if any(kw in t for kw in ["баннер", "пленк", "широкоформат", "сольвент"]):
        return "Сольвент"
    if any(kw in t for kw in ["визитк", "буклет", "листовк", "флаер", "блокнот",
                                "каталог", "календар", "стикер", "этикетк",
                                "бланк", "брошюр", "открытк"]):
        return "Печатная"
    if any(kw in t for kw in ["дизайн", "логотип", "макет", "фирменн", "брендбук"]):
        return "Дизайн"
    if any(kw in t for kw in ["кружк", "ручк", "шоппер", "футболк", "бейсболк",
                                "термос", "магнит", "шоколад", "блокнот", "ежедневник",
                                "повербанк", "пакет"]):
        return "Мерч"
    return ""



def _extract_structured_field(analysis: str, field_name: str) -> str:
    """Extract a value from structured Vision API output by field label.

    Handles format like:
        ТИП ИЗДЕЛИЯ: объёмные буквы
        ПОДСВЕТКА: лицевая LED
    """
    for line in analysis.splitlines():
        line_stripped = line.strip()
        # Match "FIELD_NAME:" or "- Field_name:" patterns
        low = line_stripped.lower()
        target = field_name.lower()
        if low.startswith(target + ":") or low.startswith("- " + target + ":"):
            value = line_stripped.split(":", 1)[1].strip()
            # Clean up brackets and "не определяется"
            if value.lower() in ("не определяется", "не определяется.", "n/a", "—", "-"):
                return ""
            # Remove surrounding brackets if present
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1].strip()
            return value
    return ""


_EN_RU_REPLACEMENTS = {
    "promotional": "рекламный",
    "logo design": "разработка логотипа",
    "turquoise": "бирюзовый",
    "gift set": "подарочный набор",
    "business card": "визитка",
    "business cards": "визитки",
    "light box": "световой короб",
    "lightbox": "световой короб",
    "press wall": "пресс-волл",
    "press-wall": "пресс-волл",
    "branding": "брендирование",
    "strategic lighting": "подсветка",
    "statement": "заявление о бренде",
    "banner": "баннер",
    "signage": "вывеска",
    "neon sign": "неоновая вывеска",
    "channel letters": "объёмные буквы",
    "acrylic": "акрил",
    "stainless steel": "нержавеющая сталь",
    "aluminum": "алюминий",
    "vinyl": "виниловая плёнка",
    "backlit": "контражурная подсветка",
    "front-lit": "лицевая подсветка",
    "outdoor": "наружный",
    "indoor": "интерьерный",
    "corporate identity": "фирменный стиль",
    "brand identity": "фирменный стиль",
    "visibility": "видимость",
    "foot traffic": "пешеходный трафик",
    "brand awareness": "узнаваемость бренда",
    "brand recognition": "узнаваемость бренда",
    "customer loyalty": "лояльность клиентов",
    "wayfinding": "навигация",
    "point of sale": "точка продаж",
    "flat": "плоский",
    "planar": "плоский",
    "digital printing": "цифровая печать",
    "printing": "печать",
    "without lighting": "без подсветки",
    "activities": "деятельность",
    "visitors": "посетители",
    "economic": "экономичный",
    "universal": "универсальный",
    "professional": "профессиональный",
    "reliable": "надёжный",
    "reliability": "надёжность",
    "suitable": "подходящий",
    "reminiscent": "напоминающий",
    "clean": "чистый",
    "image": "образ",
    "convey": "передать",
    "businesses": "бизнесы",
    "want": "хотят",
}


def _clean_model_output(text: str) -> str:
    """Clean up raw local model output: strip <think> blocks, English/Chinese text, code."""
    import re

    # 1. Strip <think>...</think> blocks (model chain-of-thought leakage)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. Strip any remaining XML-like tags
    text = re.sub(r"<[^>]+>", "", text)

    # 3. Remove Chinese characters (replace with empty string)
    text = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf\u2e80-\u2eff\u3000-\u303f\uff00-\uffef]+", "", text)

    # 4. EN→RU word replacements (case-insensitive, longest-first)
    for en, ru in sorted(_EN_RU_REPLACEMENTS.items(), key=lambda x: -len(x[0])):
        text = re.sub(re.escape(en), ru, text, flags=re.IGNORECASE)

    # 5. Remove "EN / RU" slash-duplicates (e.g. "visitors cards / визитки")
    text = re.sub(r"[a-zA-Z][a-zA-Z\s]+\s*/\s*(?=[а-яА-ЯёЁ])", "", text)
    # Remove "(english_word)" in parentheses
    text = re.sub(r"\(\s*[a-zA-Z][a-zA-Z\s]*\s*\)", "", text)
    # Remove stray English meta-words (model artifacts, NOT brands/terms)
    _EN_JUNK_WORDS = {
        "consists", "titled", "capitalized", "readable", "finished",
        "uppercase", "standard", "item", "product", "none",
        "sided", "conveniently", "deal",
    }
    text = re.sub(
        r"\b(" + "|".join(_EN_JUNK_WORDS) + r")\b",
        "", text, flags=re.IGNORECASE,
    )

    # 6. Remove English sentences embedded in mixed lines
    # Matches sequences of 4+ English words (articles, prepositions, nouns)
    text = re.sub(
        r"(?<![a-zA-Z])"  # not preceded by letter
        r"(?:[A-Z][a-z]+|[a-z]+)"  # first word
        r"(?:\s+(?:the|a|an|is|are|was|were|of|for|in|on|to|and|or|with|that|this|it|not|can|has|have|be|by|from|at|as|but|so|if|its|do|does|will|would|could|should|shall|may|might|which|who|whom|what|where|when|how|than|very|more|most|also|just|about|into|over|after|before|between|through|during|without|within|such|each|every|both|few|many|much|some|any|no|all|own)\s+)*"
        r"(?:[a-zA-Z]+\s+){2,}[a-zA-Z]+\.?"  # 3+ more English words
        r"(?![a-zA-Z])",  # not followed by letter
        "", text
    )

    # 6. Remove lines that are obviously code (C++, Python, etc.)
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Skip code lines
        if any(stripped.startswith(pat) for pat in [
            "#include", "using namespace", "int main", "cout <<",
            "import ", "def ", "class ", "return ", "print(",
            "```", "//", "/*",
        ]):
            continue
        # Skip lines that are mostly non-Cyrillic (English garbage)
        if stripped and len(stripped) > 10:
            cyrillic_count = len(re.findall(r"[а-яА-ЯёЁ]", stripped))
            total_alpha = len(re.findall(r"[a-zA-Zа-яА-ЯёЁ]", stripped))
            if total_alpha > 0 and cyrillic_count / total_alpha < 0.3:
                continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 7. Remove "This code is not related..." type disclaimers
    text = re.sub(r"This (code|program|text) is not related.*", "", text, flags=re.IGNORECASE)

    # 8. Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# ROI/ROMI benchmarks by direction (from knowledge_docs roi_benchmarks)
# ---------------------------------------------------------------------------

ROI_BENCHMARKS = {
    "Цех": {"category": "Наружная реклама", "romi_min": 150, "romi_avg": 300, "romi_opt": 500,
             "description": "Наружная реклама обеспечивает 30 000–70 000 визуальных контактов в день. "
                            "Срок окупаемости вывески: 3–6 месяцев при потоке от 500 чел/день."},
    "РИК": {"category": "Наружная реклама / Монтаж", "romi_min": 150, "romi_avg": 250, "romi_opt": 400,
             "description": "Профессиональный монтаж повышает срок службы конструкции на 30–50%. "
                            "Правильная установка обеспечивает безопасность и соответствие нормам."},
    "Сольвент": {"category": "Широкоформатная печать", "romi_min": 120, "romi_avg": 200, "romi_opt": 350,
                  "description": "Баннерная реклама — один из самых бюджетных каналов наружной рекламы. "
                                 "Стоимость контакта: от 0.01 руб. Охват до 50 000 чел/день."},
    "Печатная": {"category": "Полиграфия", "romi_min": 100, "romi_avg": 180, "romi_opt": 500,
                  "description": "Визитки обеспечивают ROMI до 400%. Листовки: конверсия 1–3% при правильной раздаче. "
                                 "Буклеты увеличивают доверие и средний чек на 15–25%."},
    "Дизайн": {"category": "Дизайн-услуга", "romi_min": 100, "romi_avg": 300, "romi_opt": 750,
                "description": "Логотип — фундамент бренда с ROMI 100–600%. Фирменный стиль повышает узнаваемость "
                               "на 80% и увеличивает повторные обращения."},
    "Мерч": {"category": "Сувениры и мерч", "romi_min": 200, "romi_avg": 450, "romi_opt": 600,
              "description": "Брендированные кружки: ROMI до 550%, повербанки: до 500%. "
                             "Мерч работает как напоминание о бренде при каждом использовании."},
}
# Aliases: direction variants that map to the same benchmarks
ROI_BENCHMARKS["Макет"] = ROI_BENCHMARKS["Дизайн"]
ROI_BENCHMARKS["Офсет"] = ROI_BENCHMARKS["Печатная"]  # офсетная печать = полиграфия

PRACTICAL_VALUE_TEMPLATES = {
    "Цех": "Вывеска обеспечивает видимость бизнеса на фасаде, привлекает пешеходный и автомобильный трафик, формирует первое впечатление о компании.",
    "Печатная": "Полиграфическая продукция обеспечивает передачу контактной информации, повышает узнаваемость бренда и поддерживает деловую коммуникацию.",
    "Дизайн": "Дизайн-продукт формирует визуальную идентичность бренда, повышает узнаваемость и создаёт единый фирменный стиль.",
    "Мерч": "Брендированная продукция работает как постоянное напоминание о компании, повышает лояльность клиентов и партнёров.",
    "Сольвент": "Широкоформатная реклама обеспечивает максимальный охват аудитории при минимальной стоимости контакта.",
    "РИК": "Профессиональный монтаж обеспечивает долговечность конструкции, безопасность и соответствие нормам размещения.",
}
PRACTICAL_VALUE_TEMPLATES["Макет"] = PRACTICAL_VALUE_TEMPLATES["Дизайн"]
PRACTICAL_VALUE_TEMPLATES["Офсет"] = PRACTICAL_VALUE_TEMPLATES["Печатная"]


def _validate_practical_value(value: str, direction: str) -> str:
    """Return practical_value from model if valid, otherwise fallback template."""
    import re
    if not value or len(value.strip()) < 10:
        return PRACTICAL_VALUE_TEMPLATES.get(direction, "")
    # Check Cyrillic ratio — if <30% Cyrillic, it's garbage
    cyrillic = len(re.findall(r"[а-яА-ЯёЁ]", value))
    total_alpha = len(re.findall(r"[a-zA-Zа-яА-ЯёЁ]", value))
    if total_alpha > 0 and cyrillic / total_alpha < 0.3:
        return PRACTICAL_VALUE_TEMPLATES.get(direction, "")
    return value.strip()


# ---------------------------------------------------------------------------
# Vision API
# ---------------------------------------------------------------------------

VISION_PROMPT = """Ты — технолог рекламно-производственной компании. Отвечай СТРОГО на русском языке.
Проанализируй фотографию и опиши рекламный продукт (сделка ID: {deal_id}, «{title}»).
Заполни КАЖДЫЙ раздел. Если не видно — напиши "не определяется". Не выдумывай то, чего нет на фото.

КАТЕГОРИЯ: [объёмные буквы / световой короб / лайтбокс / баннер / штендер / табличка / неоновая вывеска / крышная установка / панель-кронштейн / световая панель / пресс-волл / стенд / визитки / листовки / буклет / стикеры / этикетки / меню / плакат / бланк / блокнот / календарь / открытка / папка / бейдж / кружка / футболка / шоппер / ручка / бейсболка / повербанк / термос / подарочный набор / логотип / другое]

ТИП ИЗДЕЛИЯ: [конкретный тип из списка выше или уточнить]

ВНЕШНИЙ ВИД:
- Текст на изделии: [какой текст написан, если читается]
- Размеры (оценка): [примерные размеры в см/м если можно оценить]
- Визуальное описание: [2-3 предложения — что изображено, конструкция, цвета, материалы]
- Геометрия: [плоское / объёмное / короб / конструкция с креплениями / другое]

ТЕХНОЛОГИЯ:
- Подсветка: [лицевая LED / контражур / комбинированная / открытый неон / без подсветки / не определяется]
- Способ изготовления: [фрезеровка / лазерная резка / гибка / УФ-печать / сублимация / шелкография / цифровая печать / не определяется]

ПРИМЕНЕНИЕ: [фасад / интерьер / витрина / раздаточный материал / промо-подарок / другое]
ЭКСПЛУАТАЦИЯ: [уличная / интерьерная / универсальная]

ПРАКТИЧЕСКАЯ ЦЕННОСТЬ: [1-2 предложения — какую конкретную задачу решает продукт для бизнеса заказчика: привлечение пешеходного трафика, навигация, узнаваемость бренда, информирование о товарах/услугах, повышение лояльности и т.д.]"""


def download_image_as_base64(url: str, max_retries: int = 3) -> tuple[str, str] | None:
    """Download image from URL and return (base64_data, mime_type) or None.

    Retries up to max_retries times with exponential backoff on timeout/connection errors.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://labus.pro/",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 1000:
                mime = resp.headers.get("Content-Type", "image/jpeg")
                b64 = base64.b64encode(resp.content).decode("utf-8")
                if len(b64) > 5_500_000:
                    return None
                return b64, mime
            else:
                print(f"[DEBUG] Ошибка скачивания {url}: HTTP {resp.status_code}")
                return None  # HTTP error, no retry
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2, 4 sec
                print(f"[RETRY] {url} attempt {attempt+1}/{max_retries}, wait {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"[DEBUG] Ошибка соединения для {url} (все {max_retries} попытки): {e}")
        except Exception as e:
            print(f"[DEBUG] Ошибка скачивания {url}: {e}")
            return None
    return None






def analyze_deal_images(model: str, urls: list[str],
                        deal_id: str, title: str,
                        base_url: str = "") -> str | None:
    """Analyze ALL images for a deal using local Ollama vision model.

    Each image is analyzed separately with the full VISION_PROMPT.
    Returns concatenated descriptions: "Фото 1: ... Фото 2: ..."
    CSV enrichment is done later by enrich_metadata(), NOT here.
    """
    # Filter blacklisted URLs
    valid_urls = [u for u in urls if not _is_url_blacklisted(u)]
    if not valid_urls:
        return None

    # Determine Ollama API URL
    api_url = "http://localhost:11434/api/chat"
    if base_url and "localhost" in base_url:
        parsed = urlparse(base_url)
        if parsed.port and parsed.port != 11434:
            api_url = f"http://localhost:{parsed.port}/api/chat"

    prompt_text = VISION_PROMPT.format(deal_id=deal_id, title=title)

    descriptions = []

    for idx, url in enumerate(valid_urls, 1):
        img_result = download_image_as_base64(url)
        if img_result is None:
            continue

        b64_data, mime = img_result

        # Ollama /api/chat: images go INSIDE messages[0], stream=False for single JSON
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": [b64_data],
                }
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 2048},
        }

        try:
            response = requests.post(api_url, json=payload, timeout=180)
            if response.status_code == 200:
                data = response.json()
                raw_text = data.get("message", {}).get("content", "").strip()
                text = _clean_model_output(raw_text)
                if text and len(text) > 20:
                    descriptions.append(f"Фото {idx}: {text}")
                else:
                    print(f"  [WARN] Empty/short response for image {idx} of deal {deal_id}")
            else:
                print(f"  [WARN] Ollama HTTP {response.status_code} for image {idx} of deal {deal_id}")
        except requests.exceptions.Timeout:
            print(f"  [WARN] Timeout analyzing image {idx} for deal {deal_id}")
        except Exception as e:
            print(f"  [WARN] Error analyzing image {idx} for deal {deal_id}: {e}")

    if not descriptions:
        return None

    return "\n\n".join(descriptions)


# ---------------------------------------------------------------------------
# Migration: re-enrich existing docs without re-calling Vision API
# ---------------------------------------------------------------------------

def migrate_existing_docs(offers_map: dict, goods_index: dict, clean: bool = False):
    """Re-enrich all existing photo_analysis_docs.jsonl with offers.csv data.

    Preserves the original vision_analysis text, only adds/updates metadata
    and rebuilds searchable_text with structured product composition.

    If clean=True: also re-cleans vision_analysis text through _clean_model_output()
    to strip <think>, English/Chinese, code — and removes all legacy bloat fields.
    """
    if not OUTPUT_PATH.exists():
        print("No existing photo_analysis_docs.jsonl to migrate.")
        return

    print(f"Loading existing docs from {OUTPUT_PATH}...")
    existing_docs = []
    skipped = 0
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                existing_docs.append(doc)
            except json.JSONDecodeError:
                print(f"  WARNING: Skipping malformed line {line_num}")
                skipped += 1

    print(f"  Loaded {len(existing_docs)} docs ({skipped} skipped)")
    if clean:
        print("  --clean mode: re-cleaning analysis text + stripping legacy fields")

    # Re-enrich each doc
    enriched = []
    broken = 0
    for doc in existing_docs:
        meta = doc.get("metadata", {})
        deal_id = str(meta.get("deal_id", ""))
        deal_title = str(meta.get("deal_title", ""))
        urls = meta.get("image_urls", []) or meta.get("photo_files", [])
        analysis = meta.get("vision_analysis", "")

        # Fallback for legacy docs: extract analysis from searchable_text
        if not analysis and doc.get("searchable_text", ""):
            st = doc["searchable_text"]
            # Legacy format: "Фото выполненной работы: ... | ... \n---\n analysis"
            if "---" in st:
                analysis = st.split("---", 1)[1].strip()
            elif len(st) > 100:
                analysis = st

        # --clean: re-run _clean_model_output on existing analysis
        if clean and analysis:
            analysis = _clean_model_output(analysis)

        # Skip broken/truncated docs (< 50 chars of analysis)
        if len(analysis) < 50:
            broken += 1
            continue

        # Get vision model from provenance
        model = doc.get("provenance", {}).get("vision_model", "gemini-2.0-flash")

        enriched_doc = enrich_metadata(
            deal_id=deal_id,
            deal_title=deal_title,
            urls=urls,
            analysis=analysis,
            model=model,
            offers_map=offers_map,
            goods_index=goods_index,
        )
        enriched.append(enriched_doc)

    # Write back
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for doc in enriched:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Stats
    n = len(enriched)
    with_bundle = sum(1 for d in enriched if d["metadata"].get("bundle_key"))
    with_direction = sum(1 for d in enriched if d["metadata"].get("direction"))
    with_total = sum(1 for d in enriched if d["metadata"].get("deal_total"))
    with_desc = sum(1 for d in enriched if d["metadata"].get("deal_description"))
    with_products = sum(1 for d in enriched if d["metadata"].get("products"))
    with_pv = sum(1 for d in enriched if d["metadata"].get("practical_value"))
    with_roi = sum(1 for d in enriched if d["metadata"].get("roi_romi_avg"))
    avg_st = sum(len(d["searchable_text"]) for d in enriched) // n if n else 0

    print(f"\nMigration complete!")
    print(f"  Total enriched: {n} (dropped {broken} broken docs)")
    print(f"  With bundle_key:       {with_bundle}/{n}")
    print(f"  With direction:        {with_direction}/{n}")
    print(f"  With deal_total:       {with_total}/{n}")
    print(f"  With deal_description: {with_desc}/{n}")
    print(f"  With products detail:  {with_products}/{n}")
    print(f"  With practical_value:  {with_pv}/{n}")
    print(f"  With ROI/ROMI:         {with_roi}/{n}")
    print(f"  Avg searchable_text:   {avg_st} chars")
    print(f"  Written to: {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--limit", default=0, type=int, help="Limit number of deals to process (0 = all)")
@click.option("--migrate", is_flag=True, help="Re-enrich existing docs with offers.csv data (no API calls)")
@click.option("--clean", is_flag=True, help="Re-clean ALL existing docs: strip legacy fields, re-run text cleanup, rebuild metadata")
def main(limit, migrate, clean):
    """Analyze deal images via local Ollama Vision model and enrich with product composition."""

    # Load offers.csv for enrichment
    print("Loading offers.csv for product enrichment...")
    offers_map = load_offers_by_deal(OFFERS_CSV_PATH)
    print(f"  Loaded product data for {len(offers_map)} deals")

    # Load goods.csv for catalog-level product data
    print("Loading goods.csv for catalog index...")
    goods_index = load_goods_index(GOODS_CSV_PATH)
    print(f"  Loaded {len(goods_index)} catalog products")

    if migrate or clean:
        migrate_existing_docs(offers_map, goods_index, clean=clean)
        return

    # --- Normal mode: process new images via local Ollama ---
    env = load_env()
    base_url = env.get("VISION_BASE_URL", "http://localhost:11434")
    model = env.get("VISION_MODEL", "openbmb/minicpm-v4.5:latest")

    if not DEALS_JSON_PATH.exists():
        print(f"Data not found at {DEALS_JSON_PATH}. Run generateRagData.mjs first.")
        return

    print(f"Loading deals from {DEALS_JSON_PATH}...")
    with open(DEALS_JSON_PATH, "r", encoding="utf-8") as f:
        deals = json.load(f)

    with_images = [d for d in deals if d.get("IMAGE_URLS")]

    # --- Pre-filter: remove deals where ALL URLs are blacklisted ---
    def has_valid_urls(deal):
        return any(not _is_url_blacklisted(u) for u in deal.get("IMAGE_URLS", []))

    with_valid_images = [d for d in with_images if has_valid_urls(d)]
    blacklisted_deals = len(with_images) - len(with_valid_images)

    # --- Skip already processed ---
    processed_deal_ids = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    processed_deal_ids.add(str(doc.get("metadata", {}).get("deal_id")))
                except json.JSONDecodeError:
                    pass

    without_analysis = [d for d in with_valid_images if str(d.get("ID")) not in processed_deal_ids]

    print(f"\nTotal deals: {len(deals)}")
    print(f"  With images:              {len(with_images)}")
    print(f"  Blacklisted (all 404):    {blacklisted_deals}")
    print(f"  With valid images:        {len(with_valid_images)}")
    print(f"  Already processed:        {len(processed_deal_ids)}")
    print(f"  Need analysis:            {len(without_analysis)}")

    if not without_analysis:
        print("No new images to analyze.")
        return

    to_process = without_analysis
    if limit > 0:
        to_process = to_process[:limit]

    print(f"\nProcessing {len(to_process)} deals via Ollama ({model})...")
    print(f"Base URL: {base_url}")

    processed = 0
    errors = 0
    skipped_empty = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for deal in tqdm(to_process, desc="Analyzing images"):
        urls = [url for url in deal.get("IMAGE_URLS", []) if isinstance(url, str)]
        deal_id = str(deal.get("ID", "Unknown"))
        title = str(deal.get("TITLE", "Unknown"))

        if not urls:
            continue

        # Pass 1: Vision analysis (per-image via Ollama)
        analysis = analyze_deal_images(model, urls, deal_id, title, base_url=base_url)

        # Filter empty/short results
        if not analysis or len(analysis.strip()) < 50:
            skipped_empty += 1
            continue

        # Pass 2: Script aggregation via enrich_metadata()
        doc_payload = enrich_metadata(
            deal_id=deal_id,
            deal_title=title,
            urls=urls,
            analysis=analysis,
            model=model,
            offers_map=offers_map,
            goods_index=goods_index,
        )

        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc_payload, ensure_ascii=False) + "\n")
        processed += 1

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Processed:            {processed}")
    print(f"  Skipped (empty):      {skipped_empty}")
    print(f"  Blacklisted deals:    {blacklisted_deals}")
    print(f"  Output:               {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
