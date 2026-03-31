"""Russian text utilities - mirrors tokenizeText logic from RAG_ANALYTICS/lib/common.mjs."""
import re
from functools import lru_cache

# Direction detection aliases (prefix-based for Russian morphology)
# Key = prefix to match, value = canonical direction name
DIRECTION_ALIASES: dict[str, str] = {
    # Цех - workshop, signs
    "вывеск": "Цех",
    "световая": "Цех",
    "световые": "Цех",
    "световых": "Цех",
    "объемн": "Цех",
    "объёмн": "Цех",
    "буква": "Цех",
    "букв": "Цех",
    "короб": "Цех",
    "лайтбокс": "Цех",
    "лайт": "Цех",
    "фасадн": "Цех",
    "монтаж": "Цех",  # sign mounting
    "демонтаж": "Цех",
    "сборка": "Цех",
    "оклейк": "Цех",
    # Сольвент - large format, banners
    "баннер": "Сольвент",
    "банер": "Сольвент",
    "растяжк": "Сольвент",
    "широкоформатн": "Сольвент",
    "сольвент": "Сольвент",
    "интерьерн": "Сольвент",
    # Печатная - standard print
    "печатн": "Печатная",
    "принт": "Печатная",
    "полиграф": "Печатная",
    "листовк": "Печатная",
    "флаер": "Печатная",
    "буклет": "Печатная",
    "открытк": "Печатная",
    "наклейк": "Печатная",
    "самоклейк": "Печатная",
    "этикетк": "Печатная",
    # Дизайн - design services
    "дизайн": "Дизайн",
    "design": "Дизайн",
    "макет": "Дизайн",
    "верстк": "Дизайн",
    "иллюстрац": "Дизайн",
    "анимац": "Дизайн",
    "презентац": "Дизайн",
    "ролик": "Дизайн",
    "съемк": "Дизайн",
    # РИК - placement/advertising
    "рик": "РИК",
    "размещен": "РИК",
    "аренда": "РИК",
    "билборд": "РИК",
    "биллборд": "РИК",
    "щит": "РИК",
    "реклам": "РИК",  # broad - use carefully
    # Мерч - merchandise
    "мерч": "Мерч",
    "сувенир": "Мерч",
    "брендирован": "Мерч",
    "промо": "Мерч",
    # Безнал - financial modifier (not a direction per se)
    "безнал": "Безнал",
    "скидк": "Безнал",
    "надбавк": "Безнал",
}

# Bundle intent keywords
BUNDLE_KEYWORDS = {
    "набор", "комплект", "пакет", "состав", "под ключ", "включает",
    "вместе", "всё", "все", "полный", "комплексн", "под заказ", "что входит",
}

# Financial modifier signals (should not be recommended as primary products)
FINANCIAL_MODIFIER_SIGNALS = {"безнал", "скидка", "надбавка", "ндс", "предоплат"}


def tokenize_ru(text: str) -> list[str]:
    """
    Tokenize Russian text into lowercase tokens, min length 2.
    Mirrors tokenizeText() from RAG_ANALYTICS/lib/common.mjs.
    """
    if not text:
        return []
    text = text.lower()
    tokens = re.split(r"[^\w]+", text)
    return [t for t in tokens if len(t) >= 2]


@lru_cache(maxsize=1024)
def detect_direction(query: str) -> tuple[str | None, float]:
    """
    Detect business direction from query text.
    Returns (direction_name, confidence) where confidence is 0.0-1.0.
    Uses prefix matching to handle Russian morphology.
    """
    tokens = tokenize_ru(query)
    direction_votes: dict[str, int] = {}

    for token in tokens:
        for prefix, direction in DIRECTION_ALIASES.items():
            if token.startswith(prefix) or prefix.startswith(token[:4]) and len(token) >= 4:
                direction_votes[direction] = direction_votes.get(direction, 0) + 1

    if not direction_votes:
        return None, 0.0

    top_direction = max(direction_votes, key=direction_votes.get)
    total_votes = sum(direction_votes.values())
    confidence = direction_votes[top_direction] / max(total_votes, 1)

    # Normalize confidence (single strong signal = high confidence)
    if direction_votes[top_direction] >= 2:
        confidence = min(confidence * 1.3, 1.0)

    return top_direction, confidence


def detect_bundle_intent(query: str) -> bool:
    """Check if query is asking for a bundle/package."""
    query_lower = query.lower()
    tokens = set(tokenize_ru(query_lower))

    for keyword in BUNDLE_KEYWORDS:
        if keyword in query_lower or any(t.startswith(keyword[:5]) for t in tokens if len(keyword) >= 5):
            return True
    return False


def extract_budget(query: str) -> float | None:
    """Extract budget hint from query text (returns value in rubles)."""
    patterns = [
        r"(\d[\d\s]*)\s*тыс(?:\.|яч)?",   # X тыс = X*1000
        r"(\d[\d\s,.]*)(?:\s*)(?:руб|р\.?|₽)",  # X руб
        r"(\d[\d\s]*)\s*k\b",              # Xk (English)
        r"(\d[\d\s]*)\s*к\b",              # Xк (Russian k)
    ]
    query_lower = query.lower()

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, query_lower)
        if match:
            raw = match.group(1).replace(" ", "").replace(",", ".")
            try:
                value = float(raw)
                if i == 0:  # тыс
                    value *= 1000
                elif i in (2, 3):  # k/к
                    value *= 1000
                return value
            except ValueError:
                continue
    return None


def sanitize_placeholder(value: str | None) -> str:
    """Remove analytics pipeline placeholders (__MISSING_CLIENT__ etc.)."""
    if not value:
        return ""
    placeholders = {"__MISSING_CLIENT__", "__MISSING_DIRECTION__", "__MANUAL_OR_UNCATEGORIZED__", "__NOT_APPLICABLE__"}
    return "" if value in placeholders else value
