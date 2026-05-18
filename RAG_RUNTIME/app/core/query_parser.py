"""
Query parser: extracts intent, direction, and budget from Russian/English queries.
Mirrors tokenizeText logic from RAG_ANALYTICS/lib/common.mjs.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from app.utils.text import (
    tokenize_ru, detect_direction, detect_bundle_intent, extract_budget, DIRECTION_ALIASES
)

# P21.A: subcategory detection (buklet|listovka|signboard_box|...). Reuse the
# same inference module that build_index.py uses, so query subcategory matches
# doc subcategory taxonomy 1:1.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    from enrich_subcategory import detect_query_subcategory  # type: ignore
except ImportError:
    def detect_query_subcategory(query: str) -> str | None:  # type: ignore
        return None


@dataclass
class ParsedQuery:
    raw: str
    tokens: list[str] = field(default_factory=list)
    direction: str | None = None
    direction_confidence: float = 0.0
    intent: str = "general"   # "product" | "bundle" | "policy" | "timeline" | "consulting" | "general"
    budget: float | None = None
    needs_clarification: bool = False
    # P21.A: detected product subcategory (buklet, listovka, signboard_box, etc.)
    # None если query ambiguous OR не упоминает specific product. Используется
    # retriever.py для hard-filter в pricing-intents.
    detected_subcategory: str | None = None

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
        # Roadmap / process
        "дорожная карта", "этап", "этапы", "процесс", "шаг за шагом",
        "как организовать", "как запустить", "порядок работ",
        "чек-лист", "checklist", "план работ",
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

    # Determine if query needs clarification (too vague for pricing)
    _has_sign_type = any(kw in _q for kw in [
        "объёмн", "объемн", "световой короб", "неон", "баннер", "штендер",
        "табличк", "буквы", "букв", "короб", "лайтбокс", "крышн",
        "консол", "пилон", "панель-кронштейн", "логотип",
    ])
    needs_clarification = False
    if intent in ("general", "product") and direction is None and not is_bundle:
        needs_clarification = True
    elif any(kw in _q for kw in ["вывеск", "реклам"]) and not _has_sign_type and budget is None:
        needs_clarification = True
    elif direction is not None and confidence < 0.5:
        needs_clarification = True

    # P21.A: detect product subcategory из query (буклет/листовка/неон/...)
    # Используется retriever.py для hard-filter в pricing-intents.
    detected_subcategory = detect_query_subcategory(raw_query)

    return ParsedQuery(
        raw=raw_query,
        tokens=tokens,
        direction=direction,
        direction_confidence=confidence,
        intent=intent,
        budget=budget,
        needs_clarification=needs_clarification,
        detected_subcategory=detected_subcategory,
    )
