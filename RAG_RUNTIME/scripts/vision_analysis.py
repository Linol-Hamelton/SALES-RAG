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
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
RAG_DATA = PROJECT_ROOT.parent / "RAG_DATA"
DEALS_JSON_PATH = RAG_DATA / "deals.json"
OFFERS_CSV_PATH = RAG_DATA / "offers.csv"
GOODS_CSV_PATH = RAG_DATA / "goods.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "photo_analysis_raw.jsonl"
ENV_FILE = PROJECT_ROOT / "configs" / ".env"


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
    materials_detected = set()
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

        # Detect materials from product name
        for mat in ["акрил", "пвх", "композит", "нержавейк", "оцинков", "баннер",
                     "пленк", "бумаг", "картон", "холст", "ткан", "хлопок", "полиэстер",
                     "алюмини", "металл", "стекл", "поликарбонат"]:
            if mat in name.lower():
                materials_detected.add(name.split(" ")[0] if len(name.split(" ")) > 1 else name)

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
        if cost_price:
            product_entry["catalog_cost_price"] = cost_price
        catalog_products.append(product_entry)

    composition_text = "\n".join(composition_lines) if composition_lines else ""
    product_count = len(products)

    # --- Extract fields: try structured format first, then keyword fallback ---
    combined_text = f"{analysis}\n{deal_description}"

    # Category (new structured field)
    category = _extract_structured_field(analysis, "категория") or ""

    # Sign type / product type
    sign_type = (
        _extract_structured_field(analysis, "тип изделия")
        or _extract_field(combined_text, [
            "световой короб", "лайтбокс", "объёмные буквы", "объемные буквы",
            "баннер", "штендер", "табличка", "неоновая вывеска", "крышная установка",
            "панель-кронштейн", "световая панель", "брандмауэр",
            "листовка", "буклет", "визитка", "флаер", "стикер", "этикетка",
            "кружка", "ручка", "шоппер", "футболка", "бейсболка", "термос",
        ])
    )

    # Lighting
    lighting_type = (
        _extract_structured_field(analysis, "подсветка")
        or _extract_field(combined_text, [
            "лицевая led", "контражур", "задняя подсветка", "открытый неон",
            "комбинированная подсветка", "без подсветки", "внутренняя подсветка",
            "фронтальная подсветка", "светодиодная лента", "led подсветка",
        ])
    )

    # Materials (from structured + keyword)
    analysis_materials = (
        _extract_structured_field(analysis, "материалы (видимые)")
        or _extract_structured_field(analysis, "материалы")
        or _extract_field(combined_text, [
            "молочный акрил", "цветной акрил", "литой акрил", "акрил",
            "пвх", "композит", "нержавейка", "оцинковка", "алюминий",
            "мелованная бумага", "виниловая пленка", "самоклеящаяся бумага",
        ])
    )

    # New structured fields from improved prompt
    colors = _extract_structured_field(analysis, "цвета") or ""
    font_style = _extract_structured_field(analysis, "шрифт") or ""
    form_shape = _extract_structured_field(analysis, "форма") or ""
    visible_text = _extract_structured_field(analysis, "текст на изделии") or ""
    dimensions = _extract_structured_field(analysis, "размеры (оценка)") or _extract_structured_field(analysis, "размеры") or ""
    print_method = _extract_structured_field(analysis, "метод нанесения") or ""
    finishing = _extract_structured_field(analysis, "финишная обработка") or ""
    application = _extract_structured_field(analysis, "применение") or ""
    selling_point = _extract_structured_field(analysis, "ценность для клиента") or ""

    # --- Searchable text: structured, cross-referenced, information-dense ---
    search_parts = [
        f"[Фото-анализ] {deal_title}",
        f"Сделка: ID {deal_id} из offers.csv",
    ]
    if direction:
        search_parts.append(f"Направление: {direction}")
    if category:
        search_parts.append(f"Категория: {category}")
    if sign_type:
        search_parts.append(f"Тип изделия: {sign_type}")
    if colors:
        search_parts.append(f"Цвета: {colors}")
    if font_style:
        search_parts.append(f"Шрифт: {font_style}")
    if form_shape:
        search_parts.append(f"Форма: {form_shape}")
    if visible_text:
        search_parts.append(f"Текст на изделии: {visible_text}")
    if dimensions:
        search_parts.append(f"Размеры: {dimensions}")
    if lighting_type:
        search_parts.append(f"Подсветка: {lighting_type}")
    if analysis_materials:
        search_parts.append(f"Материалы: {analysis_materials}")
    if print_method:
        search_parts.append(f"Метод нанесения: {print_method}")
    if finishing:
        search_parts.append(f"Финишная обработка: {finishing}")
    if application:
        search_parts.append(f"Применение: {application}")
    if deal_total > 0:
        search_parts.append(f"Стоимость комплекта: {deal_total:,.0f} руб")
    if selling_point:
        search_parts.append(f"Ценность: {selling_point}")

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
            "category": category,
            "sign_type": sign_type,
            "colors": colors,
            "font_style": font_style,
            "form_shape": form_shape,
            "visible_text": visible_text,
            "dimensions": dimensions,
            "lighting_type": lighting_type,
            "materials": analysis_materials,
            "materials_from_products": sorted(materials_detected),
            "print_method": print_method,
            "finishing": finishing,
            "application": application,
            "selling_point": selling_point,
            # --- Фотографии ---
            "image_urls": urls,
            "photo_count": len(urls),
            # --- Исходный анализ Vision API ---
            "vision_analysis": analysis,
        },
        "provenance": {
            "sources": urls[:3],
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


def _extract_field(text: str, keywords: list[str]) -> str:
    """Extract first matching keyword from text (case-insensitive)."""
    t = text.lower()
    for kw in keywords:
        if kw in t:
            return kw.title() if len(kw) > 3 else kw.upper()
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


# ---------------------------------------------------------------------------
# Vision API
# ---------------------------------------------------------------------------

VISION_PROMPT = """Ты — главный технолог рекламно-производственной компании Лабус (labus.pro).
Перед тобой фотографии коммерческого предложения (сделка ID: {deal_id}, «{title}»).
{product_context}
Проанализируй фотографии и заполни КАЖДЫЙ раздел ниже. Если на фото не видно — напиши "не определяется".

КАТЕГОРИЯ: [одна из: Наружная реклама / Интерьерная реклама / Широкоформатная печать / Полиграфия / Сувениры и мерч / Дизайн-услуга]

ТИП ИЗДЕЛИЯ: [конкретный тип: объёмные буквы / световой короб / лайтбокс / баннер / штендер / табличка / неоновая вывеска / крышная установка / панель-кронштейн / визитки / листовки / буклет / стикеры / этикетки / кружка / футболка / шоппер / другое — указать]

ВНЕШНИЙ ВИД:
- Цвета: [перечислить основные цвета, которые видны на фото]
- Шрифт: [тип шрифта если различим: гротеск/антиква/рукописный/декоративный, жирность]
- Форма: [прямоугольная / фигурная / круглая / по контуру букв / другое]
- Текст на изделии: [какой текст написан, если читается]
- Размеры (оценка): [примерные размеры если можно оценить]

ТЕХНОЛОГИЯ:
- Подсветка: [лицевая LED / контражур / комбинированная / открытый неон / без подсветки / не определяется]
- Материалы (видимые): [акрил / ПВХ / композит / металл / бумага / плёнка / ткань / другое]
- Метод нанесения: [для полиграфии/мерча: УФ-печать / сублимация / шелкография / лазерная гравировка / тампопечать / цифровая печать / не определяется]
- Финишная обработка: [ламинация / лак / фольгирование / тиснение / скругление углов / биговка / не определяется]

ПРИМЕНЕНИЕ: [где устанавливается/используется: фасад / интерьер / витрина / стойка / раздаточный материал / промо-подарок / другое]

ЦЕННОСТЬ ДЛЯ КЛИЕНТА: [1-2 предложения — главная продающая фишка этого комплекта, чем менеджер должен убедить клиента]"""


def download_image_as_base64(url: str) -> tuple[str, str] | None:
    """Download image from URL and return (base64_data, mime_type) or None."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://labus.pro/",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200 and len(resp.content) > 1000:
            mime = resp.headers.get("Content-Type", "image/jpeg")
            b64 = base64.b64encode(resp.content).decode("utf-8")
            if len(b64) > 5_500_000:
                return None
            return b64, mime
        else:
            print(f"[DEBUG] Ошибка скачивания {url}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"[DEBUG] Ошибка соединения для {url}: {e}")
    return None


API_CLIENTS = []

def get_next_client():
    global API_CLIENTS
    available = [c for c in API_CLIENTS if not c["is_dead"] and c["cooldown_until"] < time.time()]

    if not available:
        alive = [c for c in API_CLIENTS if not c["is_dead"]]
        if not alive:
            return None
        sleep_time = min(c["cooldown_until"] for c in alive) - time.time()
        if sleep_time > 0:
            print(f"\n[!] All {len(alive)} active keys hit RPM limit. Sleeping {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        return get_next_client()

    available.sort(key=lambda c: c.get("last_used", 0))
    selected = available[0]
    selected["last_used"] = time.time()
    return selected


def _build_product_context(deal_data: dict, goods_index: dict) -> str:
    """Build product composition context string for the Vision API prompt."""
    products = deal_data.get("products", []) if isinstance(deal_data, dict) else []
    if not products:
        return ""

    lines = ["\nСостав этой сделки (товары из каталога):"]
    for p in products:
        pid = p.get("product_id", "")
        name = p["product_name"]
        qty = p["quantity"]
        price = p["price"]
        section = p["section_name"]
        cat = goods_index.get(pid, {})
        cat_section = cat.get("section_name", "") if isinstance(cat, dict) else ""
        label = cat_section or section
        total = price * qty
        if total > 0:
            lines.append(f"  - {name} [{label}]: {qty:g} × {price:,.0f} = {total:,.0f} руб")
        else:
            lines.append(f"  - {name} [{label}]: {qty:g} шт")

    deal_total = deal_data.get("opportunity", 0.0)
    if deal_total:
        lines.append(f"  Итого сделка: {deal_total:,.0f} руб")

    desc = deal_data.get("description", "")
    if desc:
        lines.append(f"\nОписание из CRM: {desc[:500]}")

    lines.append("\nИспользуй эту информацию чтобы точнее определить что изображено на фото.")
    return "\n".join(lines)


def analyze_deal_images(model: str, urls: list[str],
                        deal_id: str, title: str,
                        deal_data: dict = None,
                        goods_index: dict = None) -> str | None:
    """Analyze up to 3 images for a deal using vision API.

    Now includes product composition context from offers.csv/goods.csv
    so the model can correlate visual elements with actual products.
    """
    content = []
    product_context = ""
    if deal_data and goods_index:
        product_context = _build_product_context(deal_data, goods_index)
    prompt_text = VISION_PROMPT.format(deal_id=deal_id, title=title, product_context=product_context)
    content.append({"type": "text", "text": prompt_text})

    images_loaded = 0
    for url in urls[:3]:
        result = download_image_as_base64(url)
        if result:
            b64, mime = result
            data_url = f"data:{mime};base64,{b64}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
            images_loaded += 1

    if images_loaded == 0:
        return None

    max_attempts = len(API_CLIENTS) * 4
    for attempt in range(max_attempts):
        client_dict = get_next_client()
        if not client_dict:
            print(f"\nFATAL: All API keys have exhausted their DAILY quotas!")
            return None

        try:
            response = client_dict["client"].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=2048,
            )
            return response.choices[0].message.content or None
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                if any(kw in error_str for kw in ["PerDay", "limit: 20", "limit: 50",
                                                    "limit: 1500", "RESOURCE_EXHAUSTED"]):
                    print(f"\n[!] Key {client_dict['key_id']} exhausted DAILY quota.")
                    client_dict["is_dead"] = True
                else:
                    print(f"\n[!] Key {client_dict['key_id']} hit RPM limit. Cooldown 60s.")
                    client_dict["cooldown_until"] = time.time() + 60.0
                continue

            if "503" in error_str or "500" in error_str or "timeout" in error_str.lower():
                print(f"\n[!] Server error on Key {client_dict['key_id']}. Cooldown 15s.")
                client_dict["cooldown_until"] = time.time() + 15.0
                continue

            print(f"\nVision API error on deal {deal_id}: {e}")
            return None

    print(f"\nFailed to process deal {deal_id} after {max_attempts} retries.")
    return None


# ---------------------------------------------------------------------------
# Migration: re-enrich existing docs without re-calling Vision API
# ---------------------------------------------------------------------------

def migrate_existing_docs(offers_map: dict, goods_index: dict):
    """Re-enrich all existing photo_analysis_docs.jsonl with offers.csv data.

    Preserves the original vision_analysis text, only adds/updates metadata
    and rebuilds searchable_text with structured product composition.
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
    with_sign = sum(1 for d in enriched if d["metadata"].get("sign_type"))
    with_light = sum(1 for d in enriched if d["metadata"].get("lighting_type"))
    avg_st = sum(len(d["searchable_text"]) for d in enriched) // n if n else 0

    print(f"\nMigration complete!")
    print(f"  Total enriched: {n} (dropped {broken} broken docs)")
    print(f"  With bundle_key:       {with_bundle}/{n}")
    print(f"  With direction:        {with_direction}/{n}")
    print(f"  With deal_total:       {with_total}/{n}")
    print(f"  With deal_description: {with_desc}/{n}")
    print(f"  With products detail:  {with_products}/{n}")
    print(f"  With sign_type:        {with_sign}/{n}")
    print(f"  With lighting_type:    {with_light}/{n}")
    print(f"  Avg searchable_text:   {avg_st} chars")
    print(f"  Written to: {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--limit", default=0, type=int, help="Limit number of deals to process (0 = all)")
@click.option("--migrate", is_flag=True, help="Re-enrich existing docs with offers.csv data (no API calls)")
def main(limit, migrate):
    """Analyze deal images via Vision API and enrich with product composition."""

    # Load offers.csv for enrichment
    print("Loading offers.csv for product enrichment...")
    offers_map = load_offers_by_deal(OFFERS_CSV_PATH)
    print(f"  Loaded product data for {len(offers_map)} deals")

    # Load goods.csv for catalog-level product data
    print("Loading goods.csv for catalog index...")
    goods_index = load_goods_index(GOODS_CSV_PATH)
    print(f"  Loaded {len(goods_index)} catalog products")

    if migrate:
        migrate_existing_docs(offers_map, goods_index)
        return

    # --- Normal mode: process new images ---
    env = load_env()
    api_key_str = env.get("VISION_API_KEY", "")
    base_url = env.get("VISION_BASE_URL", "https://api.artemox.com/v1")
    model = env.get("VISION_MODEL", "gemini-2.0-flash")

    if not api_key_str:
        print("ERROR: VISION_API_KEY not set in configs/.env")
        return

    api_keys = [k.strip() for k in api_key_str.split(",") if k.strip()]
    if not api_keys:
        print("ERROR: No valid keys found in VISION_API_KEY")
        return

    clients_init = [OpenAI(api_key=ak, base_url=base_url) for ak in api_keys]

    global API_CLIENTS
    API_CLIENTS = [{"client": c, "key_id": i+1, "cooldown_until": 0,
                    "is_dead": False, "last_used": 0}
                   for i, c in enumerate(clients_init)]

    print(f"Vision API: loaded {len(API_CLIENTS)} keys / model: {model}")

    if not DEALS_JSON_PATH.exists():
        print(f"Data not found at {DEALS_JSON_PATH}. Run generateRagData.mjs first.")
        return

    print(f"Loading deals from {DEALS_JSON_PATH}...")
    with open(DEALS_JSON_PATH, "r", encoding="utf-8") as f:
        deals = json.load(f)

    with_images = [d for d in deals if d.get("IMAGE_URLS")]

    processed_deal_ids = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    processed_deal_ids.add(str(doc.get("metadata", {}).get("deal_id")))
                except json.JSONDecodeError:
                    pass

    without_analysis = [d for d in with_images if str(d.get("ID")) not in processed_deal_ids]
    print(f"Total deals: {len(deals)}, with images: {len(with_images)}, need analysis: {len(without_analysis)}")

    if not without_analysis:
        print("No new images to analyze.")
        return

    to_process = without_analysis
    if limit > 0:
        to_process = to_process[:limit]

    print(f"Processing {len(to_process)} deals via Gemini...")
    processed = 0
    errors = 0

    for deal in tqdm(to_process, desc="Analyzing images"):
        urls = [url for url in deal.get("IMAGE_URLS", []) if isinstance(url, str)]
        deal_id = str(deal.get("ID", "Unknown"))
        title = str(deal.get("TITLE", "Unknown"))

        if urls:
            deal_data = offers_map.get(deal_id, {})
            analysis = analyze_deal_images(model, urls, deal_id, title,
                                           deal_data=deal_data, goods_index=goods_index)
            if analysis:
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
            else:
                errors += 1

    print(f"Done! Processed: {processed}, Errors: {errors}")


if __name__ == "__main__":
    main()
