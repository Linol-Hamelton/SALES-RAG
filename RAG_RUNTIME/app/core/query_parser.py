"""
Query parser: extracts intent, direction, and budget from Russian/English queries.
Mirrors tokenizeText logic from RAG_ANALYTICS/lib/common.mjs.
"""
from dataclasses import dataclass, field
from app.utils.text import (
    tokenize_ru, detect_direction, detect_bundle_intent, extract_budget, DIRECTION_ALIASES
)


@dataclass
class ParsedQuery:
    raw: str
    tokens: list[str] = field(default_factory=list)
    direction: str | None = None
    direction_confidence: float = 0.0
    intent: str = "general"   # "product" | "bundle" | "policy" | "timeline" | "consulting" | "general"
    budget: float | None = None

    @property
    def has_direction(self) -> bool:
        return self.direction is not None

    @property
    def high_confidence_direction(self) -> bool:
        return self.direction is not None and self.direction_confidence >= 0.7


def parse_query(raw_query: str) -> ParsedQuery:
    """
    Parse a raw user query into structured intent signals.

    Args:
        raw_query: User's natural language query in Russian or English

    Returns:
        ParsedQuery with extracted signals
    """
    tokens = tokenize_ru(raw_query)
    direction, confidence = detect_direction(raw_query)
    is_bundle = detect_bundle_intent(raw_query)
    budget = extract_budget(raw_query)

    # Consulting/vague intent — triggers knowledge retrieval alongside products
    _q = raw_query.lower()
    is_consulting = any(kw in _q for kw in [
        # Consulting / recommendation
        "продвиж", "реклам", "как привлечь", "как продвигать",
        "что выбрать", "что лучше", "что посоветуете", "что порекомендуете",
        "посоветуй", "помогите", "помогите выбрать", "хочу вывеску",
        "нужна реклама", "нужны материалы", "для офиса", "для магазина",
        "для кафе", "для ресторана", "для салона", "для аптеки",
        "открываем", "открываю", "начинаем бизнес",
        "вопрос", "расскажите", "объясните",
        "идеально", "качественно", "премиум", "вип", "vip", "люкс", "лакшери",
        "всё должно быть", "хочу лучшее", "самое лучшее", "топ качество",
        # Briefs / production procedure
        "бриф", "заполнить", "заполни", "согласовать макет", "передать тз",
        "передать в производство", "что нужно для", "что указать",
        # Objection handling
        "возражен", "говорит дорого", "слишком дорого", "почему так дорого",
        "я подумаю", "подумаю", "у конкурентов", "не буду заказывать",
        "убедить клиента", "как ответить клиенту", "клиент отказывается",
        # Warranty / guarantees
        "гаранти", "вернуть", "поломка",
        "сломалась", "не работает", "расходники", "лампа перегорела",
        # Sales techniques
        "приём", "прием убежден", "техник продаж", "как продать",
        "как убедить", "работа с клиентом",
        # ROI / effectiveness
        "romi", "roi", "конверси", "эффективност", "окупаемост", "возврат инвестиций",
        "сколько приносит", "выгодно ли",
    ])

    # Timeline intent — production timelines
    is_timeline = any(kw in _q for kw in [
        "срок", "сколько дней", "сколько времени", "когда будет готово",
        "время изготовления", "время производства", "как долго",
        "как быстро", "сколько делается", "длительность",
        "готовность", "дней на изготовление",
    ])

    # Determine intent
    if is_timeline:
        intent = "timeline"
    elif is_bundle:
        intent = "bundle"
    elif any(kw in _q for kw in ["политика", "режим цен", "ценообразование", "правило"]):
        intent = "policy"
    elif is_consulting:
        intent = "consulting"
    elif direction is not None:
        intent = "product"
    else:
        intent = "general"

    return ParsedQuery(
        raw=raw_query,
        tokens=tokens,
        direction=direction,
        direction_confidence=confidence,
        intent=intent,
        budget=budget,
    )
