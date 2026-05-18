"""P21.A.1: subcategory inference rules for ingest + query parser.

Цель: разделить product-категории в семантически близких группах
(буклет vs листовка, объёмные буквы vs световой короб vs неон). Используется
в build_index.py (payload enrichment) и в query_parser.py (detect query subcategory).

Inference priority (highest to lowest):
1. `linked_smeta_category_ids` (если есть — explicit categorization из ingest)
2. `product_name` regex match
3. `searchable_text` regex match (last resort)

Returns: subcategory string или None если ничего не подходит.
NULL subcategory означает "ambiguous" — retriever filter пропустит через MatchAny([detected, None]).
"""
from __future__ import annotations

import re
from typing import Any

# Regex patterns с named groups для каждой subcategory. Порядок имеет значение —
# более специфичные стемы (визитк) перед общими (катало).
_SUBCATEGORY_PATTERNS: list[tuple[re.Pattern, str]] = [
    # === Print direction ===
    (re.compile(r"буклет\w*|брошюр\w*", re.IGNORECASE), "buklet"),
    (re.compile(r"листовк\w*|листовок\b|флаер\w*", re.IGNORECASE), "listovka"),
    (re.compile(r"визитк\w*", re.IGNORECASE), "visitka"),
    (re.compile(r"наклейк\w*|стикер\w*|этикетк\w*", re.IGNORECASE), "sticker"),
    (re.compile(r"календар\w*", re.IGNORECASE), "calendar"),
    (re.compile(r"каталог\w*", re.IGNORECASE), "catalog"),
    (re.compile(r"плакат\w*|постер\w*|афиш\w*", re.IGNORECASE), "poster"),
    (re.compile(r"\bменю\w*", re.IGNORECASE), "menu"),

    # === Signage / Outdoor direction ===
    (re.compile(r"объ[её]мн\w+\s+букв|объ[её]мн\w+\s+изделия", re.IGNORECASE), "signboard_volume"),
    (re.compile(r"светов\w+\s+короб|лайтбокс\w*", re.IGNORECASE), "signboard_box"),
    (re.compile(r"неон\w*", re.IGNORECASE), "signboard_neon"),
    (re.compile(r"брандмауэр\w*|брендмауэр\w*", re.IGNORECASE), "banner"),
    (re.compile(r"\bбаннер\w*|широкоформат\w*", re.IGNORECASE), "banner"),
    (re.compile(r"штендер\w*|стопер\w*", re.IGNORECASE), "shtender"),
    (re.compile(r"\bтабличк\w*", re.IGNORECASE), "tablichka"),

    # === Merch / Souvenir direction ===
    (re.compile(r"футболк\w*|футболок\b|поло\b", re.IGNORECASE), "merch_textile"),
    (re.compile(r"кружк\w*|кружек\b|термокружк\w*", re.IGNORECASE), "merch_mug"),
    (re.compile(r"бейсболк\w*|бейсболок\b|кепк\w*", re.IGNORECASE), "merch_cap"),
    (re.compile(r"\bручк\w*|ручек\b|карандаш\w*", re.IGNORECASE), "merch_pen"),
    (re.compile(r"флешк\w*|usb\b", re.IGNORECASE), "merch_usb"),
    (re.compile(r"пакет\w*|шопер\w*", re.IGNORECASE), "merch_bag"),

    # === Design direction ===
    (re.compile(r"логотип\w*|айдентик\w*", re.IGNORECASE), "design_logo"),
    (re.compile(r"брендбук\w*|фирстил\w*|фирменн\w+\s+стил", re.IGNORECASE), "design_brand"),
]


# Mapping из smeta_category_id (как они хранятся в linked_smeta_category_ids)
# к нашим subcategory enum. Эти ID берутся из RAG_ANALYTICS smeta categories.
_SMETA_CATEGORY_TO_SUBCAT = {
    "labus:Буклеты": "buklet",
    "labus:Брошюры": "buklet",
    "labus:Листовки": "listovka",
    "labus:Флаеры": "listovka",
    "labus:Визитки": "visitka",
    "labus:Наклейки": "sticker",
    "labus:Стикеры": "sticker",
    "labus:Этикетки": "sticker",
    "labus:Календари": "calendar",
    "labus:Каталоги": "catalog",
    "labus:Плакаты": "poster",
    "labus:Постеры": "poster",
    "labus:Афиши": "poster",
    "labus:Меню": "menu",
    "labus:Объёмные буквы": "signboard_volume",
    "labus:Объемные буквы": "signboard_volume",
    "labus:Световые короба": "signboard_box",
    "labus:Лайтбоксы": "signboard_box",
    "labus:Неон": "signboard_neon",
    "labus:Неоновые вывески": "signboard_neon",
    "labus:Брендмауэры": "banner",
    "labus:Баннеры": "banner",
    "labus:Штендеры": "shtender",
    "labus:Стоперы": "shtender",
    "labus:Таблички": "tablichka",
    "labus:Футболки": "merch_textile",
    "labus:Поло": "merch_textile",
    "labus:Кружки": "merch_mug",
    "labus:Термокружки": "merch_mug",
    "labus:Бейсболки": "merch_cap",
    "labus:Кепки": "merch_cap",
    "labus:Ручки": "merch_pen",
    "labus:USB": "merch_usb",
    "labus:Флешки": "merch_usb",
    "labus:Пакеты": "merch_bag",
    "labus:Шоперы": "merch_bag",
    "labus:Логотипы": "design_logo",
    "labus:Брендбук": "design_brand",
    "labus:Фирменный стиль": "design_brand",
    "labus:Айдентика": "design_brand",
}


def infer_subcategory(payload: dict[str, Any]) -> str | None:
    """Infer subcategory from document payload.

    Priority order:
    1. linked_smeta_category_ids (explicit category from ingest)
    2. product_name (regex match)
    3. searchable_text (regex match — last resort)

    Returns subcategory string or None если ambiguous/unknown.
    """
    # Priority 1: explicit smeta category ID
    smeta_ids = payload.get("linked_smeta_category_ids") or []
    if isinstance(smeta_ids, str):
        smeta_ids = [smeta_ids]
    for sid in smeta_ids:
        if sid in _SMETA_CATEGORY_TO_SUBCAT:
            return _SMETA_CATEGORY_TO_SUBCAT[sid]

    # Priority 2: product_name
    product_name = payload.get("product_name") or ""
    if product_name:
        for pattern, subcat in _SUBCATEGORY_PATTERNS:
            if pattern.search(product_name):
                return subcat

    # Priority 3: searchable_text (only first 500 chars to avoid false positives
    # from incidental mentions in long descriptions)
    text = payload.get("searchable_text") or ""
    if text:
        head = text[:500]
        for pattern, subcat in _SUBCATEGORY_PATTERNS:
            if pattern.search(head):
                return subcat

    return None


def detect_query_subcategory(query: str) -> str | None:
    """Detect what subcategory the user is asking about.

    Used by query_parser to set ParsedQuery.detected_subcategory, which then
    becomes a hard filter in retriever for pricing intents.

    Single subcategory wins on first match (most specific). If multiple
    subcategories mentioned (e.g. "буклет и листовка"), returns None to
    avoid filtering out either.
    """
    if not query or not query.strip():
        return None

    matches = []
    for pattern, subcat in _SUBCATEGORY_PATTERNS:
        if pattern.search(query):
            matches.append(subcat)

    # If multiple distinct subcategories matched — query mixes them, return None
    distinct = set(matches)
    if len(distinct) > 1:
        return None
    if len(distinct) == 1:
        return matches[0]
    return None


# Public list of all known subcategories (for QA / monitoring)
ALL_SUBCATEGORIES = sorted({s for _, s in _SUBCATEGORY_PATTERNS})


if __name__ == "__main__":
    # Quick smoke test
    test_payloads = [
        {"product_name": "Макет буклета Premium"},
        {"product_name": "Печать листовок A6 250шт"},
        {"product_name": "Визитки стандарт"},
        {"linked_smeta_category_ids": ["labus:Брендбук"]},
        {"linked_smeta_category_ids": ["labus:Объёмные буквы"]},
        {"searchable_text": "Каталог услуг 2024. Печать на мелованной бумаге..."},
        {"product_name": "Кружка хамелеон 330мл"},
        {"searchable_text": "Просто услуга без явной категории."},
    ]
    print("Subcategory inference smoke test:")
    for p in test_payloads:
        result = infer_subcategory(p)
        key = list(p.keys())[0]
        val = str(p[key])[:50]
        print(f"  {key}={val:<55} → {result}")

    print("\nQuery subcategory detection:")
    test_queries = [
        "Сколько стоят 1000 буклетов",
        "Виды фальцовки для буклетов",
        "Печать листовок А5",
        "Нужны и визитки и буклеты для клиники",
        "Световой короб для входа в магазин",
        "Сколько стоит логотип для кафе",
        "Что такое наружная реклама",
    ]
    for q in test_queries:
        result = detect_query_subcategory(q)
        print(f"  {q[:55]:<55} → {result}")
