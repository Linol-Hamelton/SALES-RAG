"""
Query Decomposer: breaks complex signage/production queries into components
for multi-query retrieval and parametric pricing.

Handles queries like:
  "световая вывеска ДАГАВТОТРАНС 80 см монтаж спецтехника 8 часов"
  → fabrication (9.6 мп) + lighting (9.6 мп) + mounting + machinery (8 ч)
"""
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Component definitions: type → {keywords, label, sub_query, unit}
# ---------------------------------------------------------------------------
COMPONENT_DEFS: dict[str, dict] = {
    "fabrication": {
        "keywords": [
            "объемн", "объёмн", "световые буквы", "световая буква", "световых букв",
            "лицевая подсветка", "лицевой подсветкой", "лицевое", "изготовлени",
            "буквы из акрил", "буквы из нержав", "буквы пвх", "буквы композит",
            "неоновые буквы", "светящиеся буквы", "нeon", "акриловые буквы",
        ],
        "label": "Изготовление объёмных букв",
        "sub_query": "изготовление объемной буквы световая лицевая подсветка мп цех",
        "unit": "мп",
        "needs_linear_meters": True,
    },
    "lighting_face": {
        "keywords": [
            "лицевая подсветка", "прямая подсветка", "лицевое освещение",
            "прямой свет", "свет для букв прям",
        ],
        "label": "Лицевая подсветка (LED)",
        "sub_query": "свет для букв прямой лицевая подсветка LED мп",
        "unit": "мп",
        "needs_linear_meters": True,
    },
    "lighting_contra": {
        "keywords": [
            "контражур", "контражурн", "задняя подсветка", "задний свет",
            "halo", "halo effect", "ореол",
        ],
        "label": "Контражурная подсветка",
        "sub_query": "свет для букв контражур задняя подсветка мп",
        "unit": "мп",
        "needs_linear_meters": True,
    },
    "mounting": {
        "keywords": [
            "монтаж", "установк", "монтировать", "смонтировать",
            "вешать", "повесить", "крепить", "крепление вывески на",
        ],
        "label": "Монтаж вывески",
        "sub_query": "монтаж буквы сложной на выезде мп цех",
        "unit": "мп",
        "needs_linear_meters": True,
    },
    "frame": {
        "keywords": [
            "каркас", "рама", "крепление на крышу", "кронштейн",
            "несущая конструкция", "профильная труба", "монтажная рама",
            "крепление к стене", "крепление к фасаду",
        ],
        "label": "Каркас и крепление",
        "sub_query": "каркас трубный профильный сварка монтаж крыша",
        "unit": "шт",
        "needs_linear_meters": False,
    },
    "machinery": {
        "keywords": [
            "спецтехника", "автовышка", "манипулятор", "вышка", "кран",
            "гидравлик", "aerial platform", "люлька", "подъёмник", "подъемник",
        ],
        "label": "Спецтехника",
        "sub_query": "спецтехника аренда выезд манипулятор автовышка",
        "unit": "час",
        "needs_linear_meters": False,
    },
    "team": {
        "keywords": [
            "монтажник", "монтажники", "бригада монтаж", "выезд монтажн",
            "2 монтажника", "3 монтажника", "4 монтажника",
        ],
        "label": "Выезд монтажников",
        "sub_query": "выезд на монтаж монтажник бригада цех",
        "unit": "выезд",
        "needs_linear_meters": False,
    },
    "design": {
        "keywords": [
            "дизайн", "макет", "визуализация", "чертеж", "проект", "эскиз",
        ],
        "label": "Дизайн / макет",
        "sub_query": "дизайн макет визуализация вывески",
        "unit": "шт",
        "needs_linear_meters": False,
    },
    "removal": {
        "keywords": [
            "демонтаж", "снятие", "демонтировать", "снять", "убрать",
        ],
        "label": "Демонтаж",
        "sub_query": "демонтаж вывески буквы конструкции",
        "unit": "мп",
        "needs_linear_meters": True,
    },
}

# Keywords that signal a complex (multi-component) query
COMPLEXITY_SIGNALS = [
    r"\b(?:высот[аы]|height)\s*\d+",
    r"\bвысот[аы]\b",
    r"\bобъёмн|\bобъемн",
    r"\bсветовы[ех]\s*букв",
    r"\bлицевая\s*подсветка",
    r"\bконтражур",
    r"\bспецтехника\b",
    r"\bкаркас\b",
    r"\bпод\s*ключ\b",
    r"\bмонтаж\s*и\b",
    r"\b\d+\s*(?:часов|час[аа])\b",
    r"\bавтовышка\b",
    r"[А-ЯЁ]{3,}",        # all-caps Russian word (brand name)
    r"\b[A-Z]{3,}\b",     # all-caps Latin word (brand name: KINGCUT, DUBAI)
]

# Average letter width relative to height (for linear meters estimation)
LETTER_WIDTH_RATIO = 0.65  # 65% of height = typical letter width


@dataclass
class ComponentSpec:
    type: str
    label: str
    sub_query: str
    quantity: float
    unit: str
    is_estimated: bool = True
    quantity_basis: str = ""  # description of how quantity was derived


@dataclass
class QueryDecomposition:
    original: str
    is_complex: bool

    # Extracted parameters
    letter_text: str = ""       # "БАРДЕРШОП", "ДАГАВТОТРАНС"
    letter_count: int = 0
    height_cm: float = 0.0
    linear_meters: float = 0.0
    hours: float = 0.0
    workers: int = 0
    technology: str = ""        # "контражур", "лицевая", "неон", "без_подсветки", "нержавейка", ""

    components: list[ComponentSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parameter extraction helpers
# ---------------------------------------------------------------------------

def _extract_height(text: str) -> float:
    """Extract height in cm from text. Returns 0.0 if not found."""
    patterns = [
        r"высот[аы]\s*(\d+(?:[.,]\d+)?)\s*(?:см|cm|мм|mm)?",
        r"(\d+(?:[.,]\d+)?)\s*(?:см|cm)\s*высот",
        r"высот[аы]\s*(?:надписи|букв[ыа])?\s*(\d+(?:[.,]\d+)?)\s*(?:см|cm)",
        r"(\d+)\s*(?:см|cm)\s*(?:высота|размер)",
        r"h\s*=?\s*(\d+)\s*(?:см|cm)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(",", "."))
            # Sanity: if value > 500, likely in mm, convert
            if val > 500:
                val /= 10
            return val
    # Fallback: bare number before "см" that looks like a height (10-300 range)
    m = re.search(r"(\d{2,3})\s*см", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if 10 <= val <= 300:
            return val
    return 0.0


def _extract_hours(text: str) -> float:
    """Extract number of hours from text."""
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*час(?:ов|а)?", text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", "."))
    return 0.0


def _extract_workers(text: str) -> int:
    """Extract number of workers mentioned."""
    m = re.search(r"(\d)\s*монтажник", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


def _extract_letter_text(text: str) -> str:
    """Extract brand name / lettering text from query.

    Looks for: ALL-CAPS Russian words, quoted text, text after 'надпись' keyword.
    """
    # Quoted text
    m = re.search(r'["\u00ab\u00bb«»]([^"»«]+)["\u00ab\u00bb«»]', text)
    if m:
        return m.group(1).strip()

    # After keyword "надпись/вывеска/буквы" — accept any case after keyword
    # Captures the brand name, stops at stop-words or measurement tokens
    _LETTER_KEYWORDS = r"(?:надпис[ья]ю?|вывеск[аеуой])\s+"
    _STOP_WORDS = (
        r"(?:из|с|от|на|для|по|под|в|к|до|без|при|за|высот\w*|размер\w*|"
        r"объемн\w*|светов\w*|букв\w*|контраж\w*|неон\w*|подсвет\w*|"
        r"\d+\s*(?:см|мм|м\b)|стоим\w*|цен\w*|полн\w*|комплект\w*|сопутств\w*)"
    )
    m = re.search(_LETTER_KEYWORDS + r'["\u00ab«]?([А-ЯЁа-яёa-zA-Z][А-ЯЁа-яёa-zA-Z\s\-]{0,50})', text, re.IGNORECASE)
    if m:
        extracted = m.group(1).strip()
        # Trim at first stop-word boundary
        extracted = re.split(r"\s+" + _STOP_WORDS, extracted, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if len(extracted) >= 2:
            return extracted.upper()

    # After "буквы <name>" pattern — name follows "буквы" keyword
    m = re.search(r"букв[аыйе]\s+([А-ЯЁа-яёa-zA-Z][А-ЯЁа-яёa-zA-Z\s\-]{0,50})", text, re.IGNORECASE)
    if m:
        extracted = m.group(1).strip()
        extracted = re.split(r"\s+" + _STOP_WORDS, extracted, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if len(extracted) >= 2:
            return extracted.upper()

    # All-caps word (Russian OR Latin) of 3+ letters (brand name)
    # Covers: KINGCUT, RESTORAN DUBAI, БАРДЕРШОП, ДАГАВТОТРАНС
    # Try multi-word all-caps sequence first (e.g. "RESTORAN DUBAI")
    m_multi = re.search(r"\b([A-ZА-ЯЁ]{2,}(?:\s+[A-ZА-ЯЁ]{2,})+)\b", text)
    if m_multi:
        return m_multi.group(1).strip()

    # Single all-caps word (Russian)
    caps_ru = re.findall(r"[А-ЯЁ]{3,}", text)
    if caps_ru:
        return max(caps_ru, key=len)

    # Single all-caps word (Latin)
    caps_lat = re.findall(r"\b[A-Z]{3,}\b", text)
    if caps_lat:
        return max(caps_lat, key=len)

    return ""


def _estimate_linear_meters(letter_count: int, height_cm: float) -> float:
    """Estimate total linear meters of sign lettering.

    Formula: letter_count × (height_m × LETTER_WIDTH_RATIO)
    Assumes average letter width = 65% of height.
    """
    if letter_count == 0 or height_cm == 0:
        return 0.0
    height_m = height_cm / 100.0
    return round(letter_count * height_m * LETTER_WIDTH_RATIO, 2)


# ---------------------------------------------------------------------------
# Technology detection
# ---------------------------------------------------------------------------

_TECH_PATTERNS: list[tuple[str, list[str]]] = [
    ("контражур", ["контражур", "задняя подсветка", "задний свет", "halo", "ореол"]),
    ("лицевая", ["лицевая подсветка", "лицевой подсветкой", "прямая подсветка", "лицевое освещение"]),
    ("неон", ["неон", "неоновые", "гибкий неон", "neon"]),
    ("нержавейка", ["нержавейка", "нержавеющ", "металлические буквы", "металл букв", "латунь", "бронза"]),
    ("без_подсветки", ["без подсветки", "несветовые", "несветовых", "без света"]),
]

# Multipliers for per-letter market rates by technology
TECH_MULTIPLIERS: dict[str, float] = {
    "контражур": 1.0,
    "лицевая": 1.2,
    "неон": 1.7,
    "нержавейка": 1.8,
    "без_подсветки": 0.7,
}


def _detect_technology(text: str) -> str:
    """Detect letter technology from query text. Returns empty string if generic."""
    text_lower = text.lower()
    for tech, keywords in _TECH_PATTERNS:
        if any(kw in text_lower for kw in keywords):
            return tech
    return ""


# ---------------------------------------------------------------------------
# Component detection
# ---------------------------------------------------------------------------

def _detect_components(
    text: str,
    linear_meters: float,
    hours: float,
    workers: int,
) -> list[ComponentSpec]:
    """Detect which service components are mentioned in the query."""
    text_lower = text.lower()
    found: list[ComponentSpec] = []
    seen_types: set[str] = set()

    for comp_type, defn in COMPONENT_DEFS.items():
        if comp_type in seen_types:
            continue

        matched = any(kw in text_lower for kw in defn["keywords"])
        if not matched:
            continue

        # Determine quantity
        if defn["unit"] == "час" and hours > 0:
            qty = hours
            basis = f"{hours:.0f} часов (из запроса)"
        elif defn["needs_linear_meters"] and linear_meters > 0:
            qty = linear_meters
            basis = f"{linear_meters:.1f} мп (расчёт по буквам × высоте)"
        elif defn["needs_linear_meters"]:
            qty = 0.0  # unknown
            basis = "количество неизвестно — уточните у менеджера"
        elif defn["unit"] == "выезд":
            qty = 1.0
            basis = f"{workers or 2} монтажника"
        else:
            qty = 1.0
            basis = "1 шт"

        found.append(ComponentSpec(
            type=comp_type,
            label=defn["label"],
            sub_query=defn["sub_query"],
            quantity=qty,
            unit=defn["unit"],
            is_estimated=defn.get("needs_linear_meters", False),
            quantity_basis=basis,
        ))
        seen_types.add(comp_type)

        # Merge lighting_face and lighting_contra into single lighting if both triggered
        if comp_type in ("lighting_face", "lighting_contra") and "lighting_face" in seen_types and "lighting_contra" in seen_types:
            pass  # keep both — different products

    # If letter_text + height found but no fabrication — imply fabrication
    # (e.g. "ШОРОХ контражурная подсветка 40 см" → letters need to be made)
    if linear_meters > 0 and not any(c.type == "fabrication" for c in found):
        defn = COMPONENT_DEFS["fabrication"]
        found.append(ComponentSpec(
            type="fabrication",
            label=defn["label"],
            sub_query=defn["sub_query"],
            quantity=linear_meters,
            unit=defn["unit"],
            is_estimated=True,
            quantity_basis=f"{linear_meters:.1f} мп (расчёт по буквам × высоте)",
        ))
        seen_types.add("fabrication")

    # If "объемные буквы" detected but no mounting — assume mounting is needed
    if any(c.type == "fabrication" for c in found) and not any(c.type == "mounting" for c in found):
        # Add mounting implicitly (common workflow)
        defn = COMPONENT_DEFS["mounting"]
        found.append(ComponentSpec(
            type="mounting",
            label=defn["label"],
            sub_query=defn["sub_query"],
            quantity=linear_meters or 0.0,
            unit=defn["unit"],
            is_estimated=True,
            quantity_basis=f"{linear_meters:.1f} мп (расчёт)" if linear_meters else "уточните",
        ))

    # Sort by natural order
    ORDER = ["design", "fabrication", "lighting_face", "lighting_contra", "frame", "mounting", "team", "machinery", "removal"]
    found.sort(key=lambda c: ORDER.index(c.type) if c.type in ORDER else 99)

    return found


# ---------------------------------------------------------------------------
# Main decomposer
# ---------------------------------------------------------------------------

def _is_complex(text: str) -> bool:
    """Heuristic: does the query contain signals of a complex multi-component request?"""
    text_lower = text.lower()
    matches = sum(1 for pat in COMPLEXITY_SIGNALS if re.search(pat, text, re.IGNORECASE))
    return matches >= 2


def decompose(query: str) -> QueryDecomposition:
    """
    Decompose a complex query into components for multi-query retrieval.

    Returns QueryDecomposition with is_complex=False for simple queries
    (caller should fall back to standard retrieval).
    """
    is_complex = _is_complex(query)

    # Always extract parameters
    height_cm = _extract_height(query)
    hours = _extract_hours(query)
    workers = _extract_workers(query)
    letter_text = _extract_letter_text(query)
    letter_count = len(letter_text.replace(" ", "")) if letter_text else 0
    linear_meters = _estimate_linear_meters(letter_count, height_cm)
    technology = _detect_technology(query)

    if not is_complex:
        return QueryDecomposition(
            original=query,
            is_complex=False,
            letter_text=letter_text,
            letter_count=letter_count,
            height_cm=height_cm,
            linear_meters=linear_meters,
            hours=hours,
            technology=technology,
        )

    components = _detect_components(query, linear_meters, hours, workers)

    return QueryDecomposition(
        original=query,
        is_complex=True,
        letter_text=letter_text,
        letter_count=letter_count,
        height_cm=height_cm,
        linear_meters=linear_meters,
        hours=hours,
        workers=workers,
        technology=technology,
        components=components,
    )
