#!/usr/bin/env python3
"""
Generate service_pricing_bridge documents that connect:
  Roadmap pricing packages ↔ Offer IDs ↔ Goods.csv product names ↔ Real deals

These bridge docs close the gap between the three knowledge layers
(transactional, knowledge, roadmaps) so the LLM can produce accurate,
package-based estimates with real catalog product names.

Usage:
    python scripts/ingest_bridges.py [--verbose]

Output:
    data/bridge_docs.jsonl
"""
import argparse
import json
import csv
import re
import unicodedata
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).parent.parent
RAG_DATA = PROJECT_ROOT.parent / "RAG_DATA"
GOODS_CSV = RAG_DATA / "goods.csv"
OFFERS_CSV = RAG_DATA.parent / "RAG_ANALYTICS" / "output" / "normalized" / "offers.normalized.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "bridge_docs.jsonl"
UNRESOLVED_PATH = PROJECT_ROOT / "data" / "bridge_unresolved.jsonl"
GENERATED_AT = datetime.now(timezone.utc).isoformat()


# ── P10.5-III: резолв product_name → PRODUCT_ID через goods.csv (G3) ──────────

_STOPWORDS = {
    "для", "из", "на", "с", "в", "и", "по", "к", "о", "от", "до", "же",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "18", "20", "25",
    "шт", "вариант", "варианта", "варианты",
    "слайдов", "слайда", "слайд",
    "интервью", "решение", "решения",
    "стороны", "сторона",
    "простой", "простая", "простое",
    "средней", "сложности",
}


def _norm_name(s: str) -> str:
    """Aggressive normalization for catalog matching.

    Critically: strips combining marks AFTER NFKD. Cyrillic 'й' (U+0439)
    decomposes into 'и' + U+0306 (combining breve); without Mn-stripping,
    the breve breaks \\w-regex and splits tokens in half.
    """
    s = unicodedata.normalize("NFKD", s or "").casefold()
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("ё", "е")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> set[str]:
    return {t for t in _norm_name(s).split() if t and t not in _STOPWORDS and len(t) > 1}


def load_goods_catalog() -> list[dict]:
    """Load goods.csv as list of {product_id, name, base_price, norm_tokens}."""
    out = []
    if not GOODS_CSV.exists():
        print(f"WARNING: {GOODS_CSV} not found — bridge product resolve disabled")
        return out
    with open(GOODS_CSV, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            pid = (row.get("PRODUCT_ID") or "").strip()
            name = (row.get("PRODUCT_NAME") or row.get("NAME") or "").strip()
            if not pid or not name:
                continue
            try:
                price = float(row.get("BASE_PRICE") or row.get("PRICE") or 0)
            except ValueError:
                price = 0.0
            out.append({
                "product_id": pid,
                "name": name,
                "base_price": price,
                "tokens": _tokens(name),
            })
    return out


def resolve_product(bridge_name: str, catalog: list[dict],
                    min_score: float = 0.55) -> dict | None:
    """Return best-match catalog entry, or None if none crosses threshold.

    Scoring combines token Jaccard (primary) and containment (helps when
    bridge name is a prefix/suffix of catalog name).
    """
    bn = _norm_name(bridge_name)
    bt = _tokens(bridge_name)
    if not bt:
        return None

    best = None
    best_score = 0.0
    for entry in catalog:
        ct = entry["tokens"]
        if not ct:
            continue
        inter = bt & ct
        if not inter:
            continue
        union = bt | ct
        jaccard = len(inter) / len(union) if union else 0.0
        cn = _norm_name(entry["name"])
        # Containment bonus: bridge_name strongly overlaps catalog name or vice versa
        contain = 0.0
        if bn and cn:
            if bn in cn:
                contain = len(bn) / max(len(cn), 1)
            elif cn in bn:
                contain = len(cn) / max(len(bn), 1)
        score = 0.7 * jaccard + 0.3 * contain
        # Prefer more-tokens-matched when scores tie
        if score > best_score or (score == best_score and best and len(inter) > len(bt & best["tokens"])):
            best_score = score
            best = entry

    if best and best_score >= min_score:
        return {
            "name": best["name"],
            "product_id": best["product_id"],
            "price": best["base_price"],
            "match_score": round(best_score, 3),
        }
    return None


# ── Bridge definitions: manually curated from roadmaps + offers analysis ──────

BRIDGE_DEFS = [
    # ═══════════════════════════════════════════════════════════════════════
    # ДИЗАЙН
    # ═══════════════════════════════════════════════════════════════════════
    {
        "service": "Дизайн логотипа",
        "direction": "Дизайн",
        "roadmap_source": "Дизайн логотипа.md",
        "roi_anchor": "ROMI 180%, при идеальной интеграции >600%",
        "prepayment": "50% предоплата, 50% после утверждения концепции",
        "urgency_surcharge": "Срочные заказы (48 часов): наценка 50-100%",
        "packages": [
            {
                "name": "Стандарт",
                "price_min": 27500,
                "price_max": 48500,
                "offer_ids": [21208],
                "offer_total": 27500,
                "description": "1–3 варианта креативно-графической идеи, ручная отрисовка одного эскиза в вектор, презентация с базовыми правилами использования",
                "products": [
                    "Заполнение брифа для разработки логотипа 1 интервью",
                    "Креативная графическая идея для логотипа с обоснованием",
                    "Отрисовка ручная с доработкой эскиза логотипа средней сложности",
                    "Правила использования логотипа 10 слайдов",
                ],
            },
            {
                "name": "Базовый",
                "price_min": 34000,
                "price_max": 58500,
                "offer_ids": [21210],
                "offer_total": 34000,
                "description": "3–5 концепций, отрисовка утверждённого варианта, подбор фирменных шрифтов и цветов, шаблонный логобук (12 слайдов)",
                "products": [
                    "Заполнение брифа для разработки логотипа 1 интервью",
                    "Разработка шрифтового логотипа Простая 1 вариант",
                    "Подбор шрифта для логотипа 1 решение",
                    "Отрисовка ручная с доработкой простого эскиза логотипа",
                    "Подбор фирменных цветов 2 решения",
                    "Логобук шаблонный 10 слайдов",
                ],
            },
            {
                "name": "Креативная группа",
                "price_min": 44000,
                "price_max": 75000,
                "offer_ids": [21214],
                "offer_total": 44000,
                "description": "До 5 концепций, индивидуальный логобук (12 слайдов), подбор шрифтов, цветов, корпоративный паттерн",
                "products": [
                    "Подбор шрифта для логотипа 1 решение",
                    "Разработка шрифтового логотипа Простая 1 вариант",
                    "Креативная графическая идея для логотипа с обоснованием",
                    "Отрисовка ручная с доработкой простого эскиза логотипа",
                    "Подбор фирменных цветов 2 решения",
                    "Логобук индивидуальный 10 слайдов",
                ],
            },
        ],
        "clarification_questions": [
            "Нужен только логотип или комплексный брендинг (фирменный стиль/брендбук)?",
            "Это новый логотип или редизайн существующего?",
            "Есть ли у вас бриф или визуальные референсы?",
        ],
        "extra_services": [
            "Дополнительный вариант концепции (вне договора): от 5 000 ₽ (8-10 часов работы)",
            "Восстановление/отрисовка старого логотипа без исходников: платная услуга",
        ],
    },

    {
        "service": "Дизайн фирменного стиля",
        "direction": "Дизайн",
        "roadmap_source": "Дизайн фирменного стиля.md",
        "roi_anchor": "",
        "prepayment": "50% предоплата",
        "packages": [
            {
                "name": "Стандарт",
                "price_min": 39500,
                "price_max": 65000,
                "offer_ids": [21216],
                "offer_total": 39500,
                "description": "Логотип + подбор фирменных цветов и шрифтов, базовые правила использования",
                "products": [
                    "Заполнение брифа для разработки логотипа 1 интервью",
                    "Креативная графическая идея для логотипа с обоснованием",
                    "Отрисовка ручная с доработкой эскиза логотипа средней сложности",
                    "Подбор фирменных цветов 2 решения",
                    "Подбор шрифта 2 решения",
                    "Правила использования логотипа 10 слайдов",
                ],
            },
            {
                "name": "Базовый",
                "price_min": 51500,
                "price_max": 80000,
                "offer_ids": [21218],
                "offer_total": 51500,
                "description": "Логотип + цвета/шрифты + корпоративный паттерн + логобук",
                "products": [
                    "Заполнение брифа для разработки логотипа 1 интервью",
                    "Креативная графическая идея для логотипа с обоснованием",
                    "Подбор фирменных цветов 2 решения",
                    "Подбор шрифта 2 решения",
                    "Отрисовка ручная с доработкой простого эскиза логотипа",
                    "Макет паттерна корпоративного Средней сложности",
                    "Логобук шаблонный 10 слайдов",
                ],
            },
            {
                "name": "Креативная группа",
                "price_min": 64500,
                "price_max": 100000,
                "offer_ids": [21220],
                "offer_total": 64500,
                "description": "Логотип + цвета/шрифты + паттерн + маскот + мудборд + индивидуальный логобук",
                "products": [
                    "Мудборд создание Простой 1 слайд",
                    "Заполнение брифа для разработки логотипа 1 интервью",
                    "Креативная графическая идея для логотипа с обоснованием",
                    "Подбор шрифта для логотипа 1 решение",
                    "Подбор фирменных цветов 2 решения",
                    "Отрисовка ручная с доработкой простого эскиза логотипа",
                    "Макет паттерна корпоративного Средней сложности",
                    "Разработка маскота Средней сложности 1 вариант",
                    "Логобук индивидуальный 10 слайдов",
                ],
            },
        ],
        "clarification_questions": [
            "У вас уже есть логотип или его тоже нужно разработать?",
            "Какие носители фирменного стиля вам нужны (визитки, бланки, презентации)?",
        ],
    },

    {
        "service": "Дизайн брендбука",
        "direction": "Дизайн",
        "roadmap_source": "Дизайн Брендбука.md",
        "roi_anchor": "",
        "prepayment": "50% предоплата",
        "packages": [
            {
                "name": "Стандарт",
                "price_min": 66000,
                "price_max": 90000,
                "offer_ids": [37272],
                "offer_total": 66000,
                "description": "Логотип + логобук + базовые макеты носителей (визитка, бланк, листовка)",
                "products": [
                    "Креативная графическая идея для логотипа с обоснованием",
                    "Отрисовка ручная с доработкой эскиза логотипа средней сложности",
                    "Логобук индивидуальный 10 слайдов",
                    "Макет визитки Средней сложности",
                    "Макет листовки Средней сложности 1 сторона",
                    "Макет фирменного бланка",
                ],
            },
            {
                "name": "Базовый",
                "price_min": 100000,
                "price_max": 150000,
                "offer_ids": [48444],
                "offer_total": 100000,
                "description": "Полный брендбук: логотип + фирменный стиль + макеты всех основных носителей (18+ позиций)",
                "products": [
                    "Логобук индивидуальный",
                    "Макет визитки",
                    "Макет бланка",
                    "Макет конверта",
                    "Макет папки",
                    "Макет титульной вывески",
                    "Макет флага",
                    "Макет ручки брендированной",
                ],
            },
            {
                "name": "Креативная группа",
                "price_min": 135000,
                "price_max": 300000,
                "offer_ids": [48452],
                "offer_total": 135000,
                "description": "Расширенный брендбук: 25+ носителей, орнаменты, мерч-макеты, интерьерное брендирование",
                "products": [
                    "Логобук индивидуальный",
                    "Ручная отрисовка с доработкой орнамента",
                    "Макет блокнота брендированного",
                    "Макет футболки брендированной",
                    "Макет таблички",
                    "Макет кружки брендированной",
                    "Макет бейджа",
                ],
            },
        ],
        "clarification_questions": [
            "Для какой отрасли нужен брендбук? (есть отраслевые шаблоны: ресторан, медцентр, салон красоты и др.)",
            "У вас уже есть логотип и фирменный стиль, или разработка с нуля?",
            "Сколько носителей фирменного стиля вам нужно (визитки, бланки, вывески, мерч)?",
        ],
        "extra_services": [
            "Отраслевые брендбуки: Ресторан (131.5К), Медцентр (110К), Строительство (90К), Салон красоты (76К)",
            "Брендбук для ивента: 55.5К (29 позиций)",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # МЕРЧ — P12.3.D: brendirovanie корпоративной атрибутики с нанесением.
    # Bucketing по тиражу, нижняя граница = себестоимость базового изделия
    # без дизайна, верхняя — премиум-ткань + комплексное нанесение.
    # Источник: fact-rollup по 69/216/70 сделкам соответственно.
    # ═══════════════════════════════════════════════════════════════════════
    {
        "service": "Футболки с нанесением",
        "direction": "Мерч",
        "roadmap_source": "merch.faq",
        "roi_anchor": "ROMI брендированного мерча: 120–350%",
        "prepayment": "50% предоплата, 50% перед отгрузкой",
        "packages": [
            {
                "name": "Малый тираж (10–30 шт)",
                "price_min": 8500,
                "price_max": 25000,
                "description": "Базовая хлопковая футболка + одностороннее нанесение (шелкография или термотрансфер). Цена за штуку: 850–900 ₽.",
                "products": [
                    "Футболка хлопок 160 г",
                    "Нанесение одностороннее термотрансфер",
                    "Дизайн-макет для мерча",
                ],
            },
            {
                "name": "Средний тираж (50–100 шт)",
                "price_min": 35000,
                "price_max": 95000,
                "description": "Хлопковая футболка 160–200 г + шелкография (1–2 цвета), двустороннее нанесение по желанию. Цена за штуку: 700–950 ₽.",
                "products": [
                    "Футболка хлопок 180 г",
                    "Нанесение шелкография 1–2 цвета",
                    "Дизайн-макет брендированной футболки",
                ],
            },
            {
                "name": "Корпоративный заказ (200–500+ шт)",
                "price_min": 140000,
                "price_max": 450000,
                "description": "Премиум-ткань, полноцветная шелкография или сублимация, упаковка, доставка. Цена за штуку: 700–900 ₽.",
                "products": [
                    "Футболка премиум хлопок 190–200 г",
                    "Нанесение полноцветная сублимация",
                    "Индивидуальная упаковка",
                    "Дизайн-макет брендированной футболки",
                ],
            },
        ],
        "clarification_questions": [
            "Какой тираж нужен (10, 50, 100, 500 шт)?",
            "Размерный ряд (какие размеры и сколько каждого)?",
            "Способ нанесения: шелкография / термотрансфер / вышивка / сублимация?",
            "Место нанесения (грудь / спина / рукав), одно или двустороннее?",
        ],
        "extra_services": [
            "Индивидуальная упаковка: +50–200 ₽/шт",
            "Вышивка логотипа: +150–400 ₽/шт (вместо шелкографии)",
            "Доставка по региону: расчёт по адресу",
        ],
    },

    {
        "service": "Кружки с нанесением",
        "direction": "Мерч",
        "roadmap_source": "merch.faq",
        "roi_anchor": "ROMI брендированного мерча: 120–350%",
        "prepayment": "50% предоплата",
        "packages": [
            {
                "name": "Малый тираж (10–30 шт)",
                "price_min": 3000,
                "price_max": 12000,
                "description": "Белая керамическая кружка 330 мл + сублимационное нанесение логотипа. Цена за штуку: 300–400 ₽.",
                "products": [
                    "Кружка керамическая белая 330 мл",
                    "Нанесение сублимация",
                    "Дизайн-макет кружки",
                ],
            },
            {
                "name": "Средний тираж (50–100 шт)",
                "price_min": 14000,
                "price_max": 45000,
                "description": "Цветная кружка (хамелеон / внутри цветная) + полноцветная сублимация. Цена за штуку: 280–450 ₽.",
                "products": [
                    "Кружка керамическая цветная 330 мл",
                    "Нанесение сублимация полноцвет",
                    "Дизайн-макет кружки",
                ],
            },
            {
                "name": "Корпоративный заказ (200–500+ шт)",
                "price_min": 50000,
                "price_max": 200000,
                "description": "Премиум-кружки (хамелеон, с гравировкой) + индивидуальная упаковка. Цена за штуку: 250–400 ₽.",
                "products": [
                    "Кружка премиум хамелеон 330 мл",
                    "Нанесение сублимация полноцвет",
                    "Индивидуальная упаковка подарочная",
                    "Дизайн-макет кружки",
                ],
            },
        ],
        "clarification_questions": [
            "Тираж (10, 50, 100, 500 шт)?",
            "Тип кружки: белая / цветная / хамелеон / с гравировкой?",
            "Нанесение: одностороннее / круговое?",
            "Нужна ли подарочная упаковка?",
        ],
        "extra_services": [
            "Гравировка вместо печати: +50–100 ₽/шт",
            "Подарочная упаковка: +80–150 ₽/шт",
        ],
    },

    {
        "service": "Бейсболки с нанесением",
        "direction": "Мерч",
        "roadmap_source": "merch.faq",
        "roi_anchor": "ROMI брендированного мерча: 120–350%",
        "prepayment": "50% предоплата",
        "packages": [
            {
                "name": "Малый тираж (10–30 шт)",
                "price_min": 5500,
                "price_max": 18000,
                "description": "Базовая хлопковая бейсболка + термотрансфер или вышивка логотипа. Цена за штуку: 550–600 ₽.",
                "products": [
                    "Бейсболка хлопок классическая",
                    "Нанесение термотрансфер лицевая сторона",
                    "Дизайн-макет бейсболки",
                ],
            },
            {
                "name": "Средний тираж (50–100 шт)",
                "price_min": 25000,
                "price_max": 70000,
                "description": "Классическая / снэпбэк + вышивка лицевой стороны, термотрансфер боковины. Цена за штуку: 500–700 ₽.",
                "products": [
                    "Бейсболка классическая/снэпбэк",
                    "Вышивка логотип лицевая сторона",
                    "Нанесение термотрансфер боковина",
                    "Дизайн-макет бейсболки",
                ],
            },
            {
                "name": "Корпоративный заказ (200–500+ шт)",
                "price_min": 100000,
                "price_max": 320000,
                "description": "Премиум-бейсболки + вышивка + полноцветный принт + индивидуальные этикетки. Цена за штуку: 500–650 ₽.",
                "products": [
                    "Бейсболка премиум с регулировкой",
                    "Вышивка фирменная",
                    "Нанесение полноцветное",
                    "Брендированная этикетка",
                    "Дизайн-макет бейсболки",
                ],
            },
        ],
        "clarification_questions": [
            "Тираж (10, 50, 100, 500 шт)?",
            "Тип: классическая / снэпбэк / премиум?",
            "Способ нанесения: вышивка / термотрансфер / их комбинация?",
            "Места нанесения: лицевая сторона / боковины / задник?",
        ],
        "extra_services": [
            "Индивидуальная этикетка: +40–80 ₽/шт",
            "Премиум-ткань (хлопок 230 г): +80–150 ₽/шт",
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # DISAMBIGUATION: отличия логотип / фирменный стиль / брендбук
    # ═══════════════════════════════════════════════════════════════════════
    {
        "service": "Логотип vs Фирменный стиль vs Брендбук",
        "direction": "Дизайн",
        "roadmap_source": "disambiguation",
        "is_disambiguation": True,
        "packages": [
            {
                "name": "Логотип",
                "price_min": 27500,
                "price_max": 75000,
                "description": "Только логотип: от эскизов до готового вектора + правила использования / логобук",
            },
            {
                "name": "Фирменный стиль",
                "price_min": 39500,
                "price_max": 100000,
                "description": "Логотип + фирменные цвета, шрифты, паттерн, маскот — всё для единого визуального языка бренда",
            },
            {
                "name": "Брендбук",
                "price_min": 66000,
                "price_max": 300000,
                "description": "Фирменный стиль + готовые макеты всех носителей (визитки, бланки, вывески, мерч). Полное руководство по бренду",
            },
        ],
        "clarification_questions": [
            "Вам нужен только сам логотип, или нужны ещё макеты носителей (визитки, бланки, вывески)?",
            "Если нужны макеты носителей — это фирменный стиль или полный брендбук с руководством?",
        ],
    },
]


def build_searchable_text(bridge: dict) -> str:
    """Build text blob optimized for embedding search."""
    parts = []
    service = bridge["service"]
    direction = bridge.get("direction", "")

    parts.append(f"[Пакеты «{service}»] Направление: {direction}")

    if bridge.get("is_disambiguation"):
        parts.append(f"Сравнение услуг: {service}")
        parts.append("Как выбрать между этими услугами? Чем они отличаются?")

    for pkg in bridge.get("packages", []):
        line = f"• {pkg['name']}: {pkg['price_min']:,} – {pkg['price_max']:,} ₽"
        if pkg.get("description"):
            line += f" ({pkg['description']})"
        parts.append(line)
        if pkg.get("products"):
            parts.append("  Состав: " + ", ".join(pkg["products"][:6]))

    if bridge.get("roi_anchor"):
        parts.append(f"ROI: {bridge['roi_anchor']}")

    if bridge.get("prepayment"):
        parts.append(f"Оплата: {bridge['prepayment']}")

    if bridge.get("clarification_questions"):
        parts.append("Уточняющие вопросы: " + "; ".join(bridge["clarification_questions"]))

    if bridge.get("extra_services"):
        parts.append("Доп. услуги: " + "; ".join(bridge["extra_services"]))

    return "\n".join(parts)


def slugify(text: str) -> str:
    """Simple slug from Russian text."""
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


def main():
    """Generate service_pricing_bridge JSONL documents."""
    parser = argparse.ArgumentParser(description="Generate service_pricing_bridge JSONL")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # P10.5-III: загружаем каталог один раз
    catalog = load_goods_catalog()
    print(f"Loaded {len(catalog)} catalog entries from {GOODS_CSV}")

    docs = []
    unresolved = []
    resolve_stats = {"total": 0, "resolved": 0}

    for bridge_def in BRIDGE_DEFS:
        service = bridge_def["service"]
        slug = slugify(service)
        doc_id = f"bridge_{slug}"

        # P10.5-III: резолвим products→PRODUCT_ID на уровне каждого пакета.
        # Изменяем объект package in-place, добавляя product_catalog_refs.
        for pkg in bridge_def.get("packages", []):
            refs = []
            for prod_name in pkg.get("products", []):
                resolve_stats["total"] += 1
                match = resolve_product(prod_name, catalog)
                if match:
                    refs.append(match)
                    resolve_stats["resolved"] += 1
                else:
                    refs.append({"name": prod_name, "product_id": None, "price": None})
                    unresolved.append({
                        "service": service,
                        "package": pkg.get("name"),
                        "product_name": prod_name,
                    })
            pkg["product_catalog_refs"] = refs

        # Collect all offer IDs and products across packages
        all_offer_ids = []
        all_products = []
        all_catalog_ids: list[str] = []
        for pkg in bridge_def.get("packages", []):
            all_offer_ids.extend(pkg.get("offer_ids", []))
            all_products.extend(pkg.get("products", []))
            for ref in pkg.get("product_catalog_refs", []):
                pid = ref.get("product_id")
                if pid:
                    all_catalog_ids.append(str(pid))

        searchable = build_searchable_text(bridge_def)
        # P10.5-III: добавить PRODUCT_ID'ы в searchable_text для sparse-хита
        if all_catalog_ids:
            searchable += f"\nКаталожные PRODUCT_ID: {', '.join(sorted(set(all_catalog_ids)))}"

        metadata = {
            "source": "bridge",
            "service": service,
            "direction": bridge_def.get("direction", ""),
            "roadmap_source": bridge_def.get("roadmap_source", ""),
            "packages": bridge_def.get("packages", []),
            "offer_ids": sorted(set(all_offer_ids)),
            "roi_anchor": bridge_def.get("roi_anchor", ""),
            "prepayment": bridge_def.get("prepayment", ""),
            "clarification_questions": bridge_def.get("clarification_questions", []),
        }

        if bridge_def.get("urgency_surcharge"):
            metadata["urgency_surcharge"] = bridge_def["urgency_surcharge"]
        if bridge_def.get("extra_services"):
            metadata["extra_services"] = bridge_def["extra_services"]
        if bridge_def.get("is_disambiguation"):
            metadata["is_disambiguation"] = True

        doc = {
            "doc_id": doc_id,
            "doc_type": "service_pricing_bridge",
            "searchable_text": searchable,
            "metadata": metadata,
            "provenance": {
                "source": "ingest_bridges.py",
                "generated_at": GENERATED_AT,
            },
        }
        docs.append(doc)

        if args.verbose:
            pkgs = len(bridge_def.get("packages", []))
            print(f"  {doc_id}: {service} [{bridge_def.get('direction','')}] — {pkgs} packages")

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nWritten: {OUTPUT_PATH} ({len(docs)} bridge documents)")
    print(f"  Services covered: {', '.join(d['metadata']['service'] for d in docs)}")

    # P10.5-III: отчёт по резолву
    total = resolve_stats["total"]
    resolved = resolve_stats["resolved"]
    pct = (100.0 * resolved / total) if total else 0.0
    print(f"\nCatalog resolve: {resolved}/{total} ({pct:.1f}%)")
    if unresolved:
        UNRESOLVED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(UNRESOLVED_PATH, "w", encoding="utf-8") as f:
            for u in unresolved:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
        print(f"  Unresolved names -> {UNRESOLVED_PATH} ({len(unresolved)} lines)")


if __name__ == "__main__":
    main()
