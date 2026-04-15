"""Query endpoints: /query and /query_structured."""
import re
import time
from collections import OrderedDict
from threading import Lock
from fastapi import APIRouter, Request, HTTPException, Depends
from app.schemas.query import QueryRequest, HumanQueryResponse, StructuredResponse, ChatMessage
from app.schemas.pricing import (
    PriceBand, EstimatedPrice, BundleItem, Reference, SourceDistinction,
    ParametricBreakdown, ParametricLineItem, DealItem,
    SourceSegment, SegmentedReferences,
)
from app.core.query_decomposer import decompose, QueryDecomposition
from app.core.query_parser import parse_query
from app.core import parametric_calculator as param_calc
from app.core.feedback_store import build_feedback_context
from app.core.deal_lookup import DealLookup
from app.core.smeta_engine import has_strong_keyword_override
from app.core.intent_classifier import IntentResult, get_classifier
from app.config import settings as app_settings
from app.auth import get_optional_user
from app.routers.chats import save_message, get_chat_history
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Query"])


def _get_intent_instruction(intent: str) -> str:
    """Load intent-specific instruction from prompts.yaml cache."""
    import yaml
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def _load():
        from pathlib import Path
        p = Path(__file__).parent.parent.parent / "configs" / "prompts.yaml"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f).get("intent_instructions", {})
        return {}

    return _load().get(intent, "")


MIN_RELEVANT_SCORE = 0.005  # below this = cross-encoder deems irrelevant, don't use for pricing

_ESTIMATE_KEYWORDS = [
    "осмет", "смету", "смета", "сметы",
    "список товаров", "что добавить в сделку", "товары для сделки",
    "сформируй сделку", "составь сделку", "позиции для сделки",
    "заполни сделку", "создай сделку", "что включить в сделку",
    # Pricing intent (П7): любые «сколько стоит / под ключ / оцени / прайс»
    "под ключ", "сколько стоит", "сколько будет", "сколько обойдётся", "сколько обойдется",
    "оцени", "оцените", "оценить", "оценка", "прайс", "цена на", "цену на",
    "стоимость", "посчитай", "посчитайте", "рассчитай", "рассчитайте", "расчёт", "расчет",
    # П7.1 #5: типография / тираж / форматные продукты
    "тираж", "напечат", "печать", "печатать",
    "визитк", "плакат", "баннер", "листовк", "наклейк", "буклет", "флаер", "афиш",
    "а3", "а4", "а5", "а2", "а1", "а0",
]

# П7.1 #4: маркеры under-key/брендбук-намерения. Если запрос содержит хоть один —
# дешёвый шаблон «Логотип» (отрисовка) НЕ должен побеждать. Форсим LLM-фоллбэк.
_UNDERKEY_INTENT_MARKERS = [
    "под ключ", "брендбук", "брендинг", "фирменн", "фирстил", "айдентик",
    "брифинг", "бриф ", "интервью", "концепци", "три идеи", "3 идеи",
    "несколько вариант", "вариант", "разработка с нуля", "с нуля",
    "креатив",
]

# Категории, которые слишком дёшевы для under-key запросов и должны отключаться.
_UNDERKEY_BLACKLIST_CATEGORIES = {"Логотип", "Логотипы"}

# П7.1-hotfix: «помойные» категории шаблонов — слишком маленькие/дешёвые,
# чтобы закрывать реальные under-key запросы. Блокируем в рантайме до П7.5-ребилда.
_JUNK_CATEGORIES = {"Вывески"}

# P9: категории, для которых у нас есть service_pricing_bridge-документы
# с корректными пакетами из roadmap (Стандарт/Базовый/Креативный и т.п.).
# SmetaEngine отдаёт по ним плоскую среднюю (~20K для логотипа), что ниже
# минимальной цены даже базового пакета. Блокируем SmetaEngine для этих
# категорий — пусть LLM получит bridge-документы через intent-aware retrieval
# и предложит клиенту 2-3 пакета с диапазонами.
_BRIDGE_CATEGORIES = {"Логотип", "Брендбук", "Фирменный стиль"}

# P10/A3: SmetaEngine category → service name в service_pricing_bridge.
# Используется `_force_inject_bridge`, чтобы гарантированно достать
# bridge-документ, когда SmetaEngine уже был заблокирован, но семантический
# retrieval может не дотянуть нужный чанк в top-k.
_CATEGORY_TO_BRIDGE_SERVICE = {
    "Логотип": "Дизайн логотипа",
    "Фирменный стиль": "Дизайн фирменного стиля",
    "Брендбук": "Дизайн брендбука",
}

# П7.1-hotfix: минимальная сумма шаблона, при которой его разрешено отдавать
# под under-key-запрос. Шаблон меньше — скорее всего дефектный.
_UNDERKEY_MIN_TEMPLATE_TOTAL = 15000

# П8.6-B: маркеры «требуется ручной расчёт менеджером». При попадании — confidence
# форсится в "manual" ПОСЛЕ всех вычислений (и LLM, и SmetaEngine override),
# независимо от их вердикта. Плюс в флаги добавляется "Требуется ручной расчёт".
# Отдельно от _ESTIMATE_KEYWORDS: кейс может быть estimate-запросом И требовать
# manual одновременно (напр. «Стоимость нестандартной вывески из нержавейки»).
_MANUAL_COMPLEXITY_MARKERS = [
    re.compile(r"на\s+высот[еы]\s+(бол[еь]е|свыше)?\s*\d+\s*м", re.IGNORECASE),
    re.compile(r"альпинист|промальп", re.IGNORECASE),
    re.compile(r"нестандартн|эксклюзивн", re.IGNORECASE),
    re.compile(r"уникальн\w*\s+\w*\s*(стенд|конструкци|проект|вывеск|издели|объект)", re.IGNORECASE),
    re.compile(r"индивидуальн\w*\s+\w*\s*(дизайн|кампани|проект|разработ|концепци|реклам)", re.IGNORECASE),
    re.compile(r"нержавей|бронз|латун", re.IGNORECASE),
    re.compile(r"\bремонт\b|восстановл|реставраци", re.IGNORECASE),
]


# П8.9: spec-heavy signage queries — запросы, где пользователь явно указал
# тип вывески (объёмные буквы / световая вывеска / контражур) вместе с
# геометрией (высота N см) и материалом (композит/алюминий/акрил/бортогиб).
# SmetaEngine по таким запросам либо через keyword override, либо через
# семантический backup матчит категорию «Световые вывески» и выдаёт auto-цену
# по template-медиане (~38k), тогда как реальная цена определяется именно
# спецификацией и должна считаться вручную. См. fb#14 (ожидание ~30k) и
# fb#55 (ожидание постатейного разложения по каталогу). Force manual.
_SPEC_HEAVY_SIGNAGE_TRIGGERS = [
    re.compile(r"объ[её]мн\w+\s+букв", re.IGNORECASE),
    re.compile(r"световы\w+\s+вывеск", re.IGNORECASE),
    re.compile(r"контражур", re.IGNORECASE),
    re.compile(r"вывеск\w*\s+.{0,40}?(подсветк|контражур)", re.IGNORECASE),
]
_SPEC_HEIGHT_PATTERN = re.compile(
    r"(высот\w*\s*\d+\s*см|\d+\s*см\s+(?:букв|высот|шрифт)|высот\w*\s+\d+\s*см)",
    re.IGNORECASE,
)
_SPEC_MATERIAL_PATTERN = re.compile(
    r"композит|акрил|алюмин|бортогиб|пвх|ламинат|подложк\w+\s+(композит|акрил|алюмин)",
    re.IGNORECASE,
)


def _is_spec_heavy_signage_query(query: str) -> bool:
    """П8.9: True если в запросе явно указаны (1) тип вывески, (2) высота,
    (3) материал. Такие запросы должны идти на ручной расчёт — template-оценка
    по «Световые вывески» систематически промахивается."""
    if not query:
        return False
    has_trigger = any(rx.search(query) for rx in _SPEC_HEAVY_SIGNAGE_TRIGGERS)
    if not has_trigger:
        return False
    has_height = _SPEC_HEIGHT_PATTERN.search(query) is not None
    has_material = _SPEC_MATERIAL_PATTERN.search(query) is not None
    return has_height and has_material


# П9.0-J: underspecified category queries — запрос называет широкую категорию
# но не уточняет критические параметры, от которых зависит цена.
# fb#1: «монтаж наружной рекламы» без типа изделия (баннер/вывеска/короб/плёнка)
# fb#2/7: «листовки А4 тираж 1000» без способа производства (офсет vs цифра)
_MONTAGE_TRIGGER = re.compile(r"монтаж\w*\s+\w*\s*(наружн|реклам|вывеск)", re.IGNORECASE)
_MONTAGE_SPECIFIERS = re.compile(
    r"баннер|короб|букв|плёнк|пленк|аппликац|штендер|неон|световой\s+короб|вывеск\w+\s+(из|с\s+)",
    re.IGNORECASE,
)
_LEAFLET_TRIGGER = re.compile(r"листовк|буклет|флаер|брошюр", re.IGNORECASE)
_LEAFLET_SPECIFIERS = re.compile(
    r"офсет|цифр\w+\s+печат|оператив|цифровая|срочн|быстр|ризограф",
    re.IGNORECASE,
)

_UNDERSPEC_MONTAGE_SUMMARY = (
    "Стоимость монтажа сильно зависит от типа изделия. "
    "Уточните, что именно нужно смонтировать: баннер на рейке, баннер на металлокаркасе, "
    "плёнку виниловую, аппликацию, световой короб, вывеску из объёмных букв? "
    "Цены и сроки у этих видов монтажа принципиально различаются."
)
_UNDERSPEC_LEAFLET_SUMMARY = (
    "Для расчёта стоимости листовок нужно уточнить способ производства: "
    "офсет (дешевле, но дольше — от 5 рабочих дней) или цифровая/оперативная печать "
    "(дороже, но быстрее — 1–2 дня). Цена при тираже 1 000 шт может отличаться в 2–3 раза "
    "в зависимости от направления."
)


# П9.1-B: height-based signage pricing — deterministic override когда
# SmetaEngine отдаёт одну category-median для всех высот. fb#66/68/70:
# ПАРАПЕТ 40см/60см/80см → одинаковая 38384₽. Рыночные ориентиры из промпта:
_HEIGHT_PRICE_BRACKETS = [
    (20, 30, 2500, 4000),
    (30, 50, 4500, 8000),
    (50, 80, 8000, 16000),
    (80, 120, 18000, 35000),
]
_MONTAGE_BRACKETS = [(0, 50, 20000, 30000), (50, 80, 30000, 45000), (80, 999, 45000, 60000)]
_KARKAS_BRACKETS = [(0, 50, 10000, 15000), (50, 80, 15000, 25000), (80, 999, 25000, 40000)]
_DESIGN_RANGE = (5000, 15000)
_SIGNAGE_HEIGHT_RE = re.compile(r"высот\w*\s*(\d+)\s*см", re.IGNORECASE)
_ALLCAPS_WORD_RE = re.compile(r"\b([А-ЯЁA-Z]{2,})\b")
_LETTER_COUNT_RE = re.compile(r"(\d+)\s*букв", re.IGNORECASE)
_SIGNAGE_TRIGGER_RE = re.compile(
    r"объ[её]мн\w+\s+букв|световы\w+\s+(объ[её]мн|вывеск)|вывеск\w+.*букв|букв\w+.*вывеск",
    re.IGNORECASE,
)


def _bracket_lookup(h: int, brackets: list[tuple]) -> tuple[float, float] | None:
    for lo, hi, pmin, pmax in brackets:
        if lo <= h < hi:
            return (pmin, pmax)
    return None


def _compute_height_based_price(query: str) -> dict | None:
    """П9.1-B: если запрос про вывеску с указанием высоты — считаем цену
    по рыночным ориентирам вместо template-медианы."""
    if not _SIGNAGE_TRIGGER_RE.search(query):
        return None
    m_h = _SIGNAGE_HEIGHT_RE.search(query)
    if not m_h:
        return None
    height = int(m_h.group(1))
    if height < 20 or height > 120:
        return None
    letter_bracket = _bracket_lookup(height, _HEIGHT_PRICE_BRACKETS)
    if not letter_bracket:
        return None
    lp_min, lp_max = letter_bracket
    m_lc = _LETTER_COUNT_RE.search(query)
    caps_words = _ALLCAPS_WORD_RE.findall(query)
    if m_lc:
        n_letters = int(m_lc.group(1))
    elif caps_words:
        longest = max(caps_words, key=len)
        n_letters = len(longest)
    else:
        n_letters = 0
    montage = _bracket_lookup(height, _MONTAGE_BRACKETS) or (20000, 30000)
    karkas = _bracket_lookup(height, _KARKAS_BRACKETS) or (10000, 15000)
    design_min, design_max = _DESIGN_RANGE
    if n_letters > 0:
        total_min = lp_min * n_letters + montage[0] + karkas[0] + design_min
        total_max = lp_max * n_letters + montage[1] + karkas[1] + design_max
        letters_label = f"{n_letters} букв × {lp_min:,}–{lp_max:,} руб/букву".replace(",", " ")
    else:
        total_min = lp_min * 5 + montage[0] + karkas[0] + design_min
        total_max = lp_max * 10 + montage[1] + karkas[1] + design_max
        letters_label = f"{lp_min:,}–{lp_max:,} руб/букву (кол-во букв не указано)".replace(",", " ")
    total_mid = round((total_min + total_max) / 2, -2)
    return {
        "height": height,
        "n_letters": n_letters,
        "lp_min": lp_min, "lp_max": lp_max,
        "total_min": round(total_min, -2),
        "total_max": round(total_max, -2),
        "total_mid": total_mid,
        "letters_label": letters_label,
        "montage": montage,
        "karkas": karkas,
    }


def _underspec_clarification(query: str) -> tuple[str, str] | None:
    """П9.0-J: returns (summary, flag) if query needs direction/type clarification."""
    if not query:
        return None
    q = query
    if _MONTAGE_TRIGGER.search(q) and not _MONTAGE_SPECIFIERS.search(q):
        return (_UNDERSPEC_MONTAGE_SUMMARY, "Не указан тип изделия для монтажа — уточните у клиента")
    if _LEAFLET_TRIGGER.search(q) and not _LEAFLET_SPECIFIERS.search(q):
        return (_UNDERSPEC_LEAFLET_SUMMARY, "Не указан способ производства (офсет/цифра) — уточните у клиента")
    return None


# П9.1-A: consultation-intent queries — ознакомительные вопросы типа
# «Вы делаете логотипы?», «У вас есть вывески?», «Вы работаете с ресторанами?»
# НЕ являются запросом цены. Эксперт fb#72: «Надо прежде презентовать
# компанию, продать услугу, а потом считать цену. Ты упираешься в цену
# даже если тебя не спрашивают.»
_CONSULTATION_PATTERNS = [
    re.compile(r"^\s*(а\s+)?(вы|Вы)\s+(делает|можете|умеете|работает|оказыва|выполня|принима|берёт|изготавлива|производит)\w*\s+", re.IGNORECASE),
    re.compile(r"^\s*(а\s+)?(у вас|У вас)\s+(есть|можно|бывает|имеется)", re.IGNORECASE),
    re.compile(r"^\s*(вы|а вы)\s+(делает|можете)\w*\s+\w+\s*\??\s*$", re.IGNORECASE),
]


def _is_consultation_query(query: str) -> bool:
    """П9.1-A: True если запрос — ознакомительный (не ценовой).
    'Вы делаете логотипы?' → presentation mode, no price."""
    if not query:
        return False
    if any(rx.search(query) for rx in _CONSULTATION_PATTERNS):
        has_price_kw = bool(re.search(
            r"стоимост|стоит|цен[аеуы]|расценк|прайс|смет|бюджет|сколько",
            query, re.IGNORECASE,
        ))
        return not has_price_kw
    return False


def _is_manual_complexity_query(query: str) -> bool:
    """П8.6-B: True если запрос содержит маркеры, при которых менеджер
    обязан делать расчёт вручную — ремонт, работы на высоте, нестандартные
    материалы (нержавейка/бронза), индивидуальный дизайн, уникальные стенды.
    SmetaEngine/LLM могут ошибаться с такими запросами, поэтому финальный
    confidence жёстко переводится в 'manual' и добавляется ручной флаг.
    """
    if not query:
        return False
    return any(rx.search(query) for rx in _MANUAL_COMPLEXITY_MARKERS)


# П8.7-D1: out-of-scope запросы — не про товары/услуги вообще. Модель
# склонна галлюцинировать цену («1000 руб» на «Какой у вас режим работы?»),
# поэтому для таких запросов принудительно обнуляем estimated_price/price_band,
# force confidence=manual и добавляем флаг о неподходящей тематике.
_OUT_OF_SCOPE_MARKERS = [
    re.compile(r"режим\s+работы|график\s+работы|часы\s+работы|время\s+работы", re.IGNORECASE),
    re.compile(r"когда\s+(откры|закры|работае)|во\s+сколько\s+(откры|закры)", re.IGNORECASE),
    re.compile(r"расписани[еяю]|выходны[еыхй]\s+(дн|ли)", re.IGNORECASE),
    re.compile(r"как\s+(добрать|проехать|найти)|где\s+(находит|расположен)", re.IGNORECASE),
    re.compile(r"адрес\s+офиса|адрес\s+компани|\bваш\s+адрес", re.IGNORECASE),
    re.compile(r"номер\s+телефон|контактн.*телефон|\bтелефон\s+офис", re.IGNORECASE),
]


def _is_out_of_scope_query(query: str) -> bool:
    """П8.7-D1: True если запрос не про наши товары/услуги (режим работы,
    адрес, контакты, график и т.д.). Такие запросы не должны получать
    estimated_price вообще — только информационный ответ."""
    if not query:
        return False
    return any(rx.search(query) for rx in _OUT_OF_SCOPE_MARKERS)


# П8.7-D2/D3: финансовые модификаторы — запросы про наценку/скидку/безнал/ндс,
# не являющиеся самостоятельным товаром. pricing_resolver детектирует только по
# product_name в retrieval, но ретривер может не вернуть нужный документ. Делаем
# query-level детектор, чтобы не зависеть от ретривера.
_FINANCIAL_MODIFIER_MARKERS = [
    # Модификатор с явным упоминанием процента или цены
    re.compile(r"безнал\w*", re.IGNORECASE),
    re.compile(r"\bналичн\w*|\bналичк\w*", re.IGNORECASE),
    re.compile(r"скидк\w*\s*\d+\s*%|скидк\w*.*стоимост|\d+\s*%\s*скидк", re.IGNORECASE),
    re.compile(r"надбавк\w*|наценк\w*", re.IGNORECASE),
    re.compile(r"\bндс\b|без\s*ндс|с\s*ндс", re.IGNORECASE),
    re.compile(r"предоплат\w*\s*\d+\s*%|рассрочк\w*", re.IGNORECASE),
]


def _is_financial_modifier_query(query: str) -> bool:
    """П8.7-D2: True если запрос — про финансовый модификатор (безнал,
    скидка, надбавка, НДС и т.п.), а не про товар. Такие запросы не должны
    получать конкретную цену — только флаг о модификаторе и ручной расчёт."""
    if not query:
        return False
    return any(rx.search(query) for rx in _FINANCIAL_MODIFIER_MARKERS)


# П8.8-A: пустой контекст смета-запроса. Клиент пишет «дай смету на эту услугу»
# или «напиши смету для сделки» БЕЗ указания товара/направления — ни контекста
# в запросе, ни продуктового существительного. SmetaEngine в таких случаях
# падал на ближайший центроид («Листовки» — самая большая категория) и возвращал
# 14 015 ₽ вместо клариф-ответа. См. fb#22, fb#24.
_EMPTY_CONTEXT_SMETA_MARKERS = [
    re.compile(r"(дай|составь|напиши|сделай|подготов|сформируй|нужн[аы]?)\s+смет\w*\s+(на|для)\s+эт", re.IGNORECASE),
    re.compile(r"осмет\w*\s+эт", re.IGNORECASE),
    re.compile(r"смет\w*\s+(для|на)\s+(сделк|услуг|позици|клиент|заявк|этого|этой|этому)", re.IGNORECASE),
    re.compile(r"^\s*(дай|напиши)\s+смет\w*\s*$", re.IGNORECASE),
]

# Продуктовые/направлениевые существительные — если хоть одно встретилось,
# запрос НЕ пустой (контекст указан прямо в фразе).
_PRODUCT_CONTEXT_NOUNS = (
    "логотип", "лого", "вывеск", "баннер", "листовк", "визитк", "наклейк",
    "буклет", "флаер", "брендбук", "брендинг", "штендер", "панель", "кронштейн",
    "табличк", "реклам", "печат", "монтаж", "дизайн", "объ[её]мн", "букв",
    "светов", "фасад", "витрин", "стикер", "этикет", "упаковк", "стенд",
    "постер", "афиш", "плакат", "рол[а-я]п", "шаурм", "кофейн", "ресторан",
    "магазин", "аптек", "офис", "стиль", "айдентик", "компани",
    "концепци", "3d", "3-d", "навигаци", "меню", "календар",
)


# П8.8-G: forbidden-promise патчи. Эксперт явно запрещает:
#   1. Предлагать бесплатные услуги («Мы не оказываем никаких бесплатных услуг»).
#   2. Давать визуализацию/расчёт по фото («По фото мы не можем оценить условия
#      монтажа», «Точный расчет возможен только после замеров и дизайна»).
# Пост-фильтр чистит текст summary/reasoning и инжектит канонические формулировки.
_FORBIDDEN_FREE_PATTERN = re.compile(r"\bбесплатн\w*", re.IGNORECASE)
_VISUALIZATION_QUERY_PATTERN = re.compile(
    r"как\s+(это\s+)?будет\s+на\s+фасад|визуализаци|покаж\w*\s+как|увидеть\s+как\s+(это|будет)|фото\s*ш?оп|3d\s*модел",
    re.IGNORECASE,
)
_CANONICAL_VIZ_PHRASE = (
    "Визуализация объекта выполняется только после выезда на замеры и "
    "платного аванса за дизайн-проект."
)
_CANONICAL_FREE_PHRASE = (
    "Мы не оказываем бесплатных услуг — каждая позиция тарифицируется."
)


def _is_visualization_request(query: str) -> bool:
    """П8.8-G: True если клиент просит «показать как будет на фасаде / визуализацию»."""
    if not query:
        return False
    return bool(_VISUALIZATION_QUERY_PATTERN.search(query))


_CANONICAL_VIZ_RESPONSE = (
    "Визуализация объекта на фасаде выполняется только после выезда "
    "замерщика и платного аванса за дизайн-проект. Без этих этапов мы не "
    "даём 3D-макеты и не оцениваем условия монтажа по фото. Согласуйте "
    "выезд на замеры — менеджер подберёт подходящий вариант и рассчитает "
    "стоимость дизайна и изготовления."
)


def _apply_forbidden_promise_filter(
    summary: str,
    reasoning: str,
    flags: list[str],
    query: str,
) -> tuple[str, str, list[str]]:
    """П8.8-G: пост-фильтр. Убирает упоминания «бесплат*» и для viz-запросов
    полностью заменяет summary каноном — иначе LLM успевает предложить
    3D-визуализацию за 3000 ₽ до выезда на замеры (см. fb#19).
    """
    new_flags = list(flags)
    txt_before = (summary or "") + " " + (reasoning or "")
    had_free_promise = bool(_FORBIDDEN_FREE_PATTERN.search(txt_before))
    is_viz_request = _is_visualization_request(query)

    def _strip_free(text: str) -> str:
        if not text:
            return text
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = [s for s in sentences if not _FORBIDDEN_FREE_PATTERN.search(s)]
        return " ".join(kept).strip() or text

    if had_free_promise and not is_viz_request:
        summary = _strip_free(summary)
        reasoning = _strip_free(reasoning)
        extra = _CANONICAL_FREE_PHRASE
        if extra not in summary:
            summary = (summary + " " + extra).strip()
        flag = "Запрос на бесплатную услугу заблокирован — все услуги платные"
        if flag not in new_flags:
            new_flags = [flag] + new_flags

    if is_viz_request:
        # P8.8-hotfix: полностью заменяем summary/reasoning — нельзя оставлять
        # LLM-генерацию, которая предлагает 3D-макет за 3000 ₽ отдельным
        # предложением от слова «визуализация».
        summary = _CANONICAL_VIZ_RESPONSE
        reasoning = (
            "Запрос про визуализацию объекта на фасаде. По правилам "
            "компании мы не выдаём бесплатных визуализаций и не оцениваем "
            "условия монтажа по фото — сначала замеры и аванс за дизайн."
        )
        flag = "Визуализация — только после замеров и аванса за дизайн"
        if flag not in new_flags:
            new_flags = [flag] + new_flags

    return summary, reasoning, new_flags


# П8.8-J: referential queries — клиент ссылается на «эту вывеску / вместо этой /
# для этого объекта» без приложенного фото или истории чата. Без контекста
# модель не может дать оценку, поэтому форсим clarification + «прикрепи фото».
_REFERENTIAL_MARKERS = [
    re.compile(r"вместо\s+(эт\w+|этой|этого|этих)", re.IGNORECASE),
    re.compile(r"\bэт[ау]\s+(услуг|позици|вывеск|букв|баннер|табличк|сделк)", re.IGNORECASE),
    re.compile(r"для\s+эт\w+\s+(объект|клиент|сделк|проект)", re.IGNORECASE),
    re.compile(r"как\s+в\s+прошл\w+\s+(раз|сделк|заказ)", re.IGNORECASE),
]


def _is_referential_query(query: str) -> bool:
    """П8.8-J: True если запрос ссылается на «эту услугу / вместо этой»
    и не содержит самостоятельного описания того объекта."""
    if not query:
        return False
    return any(rx.search(query) for rx in _REFERENTIAL_MARKERS)


def _is_empty_context_smeta_query(query: str) -> bool:
    """П8.8-A: True если запрос на смету без продуктового контекста.
    «Дай смету на эту услугу» → True (нужна кларификация).
    «Смета на листовки А4» → False (есть продукт).
    Защита от фолбэка SmetaEngine на ближайшую категорию (обычно «Листовки»).
    """
    if not query:
        return False
    if not any(rx.search(query) for rx in _EMPTY_CONTEXT_SMETA_MARKERS):
        return False
    q_lower = query.lower()
    for noun in _PRODUCT_CONTEXT_NOUNS:
        if re.search(noun, q_lower):
            return False
    return True


# П8.7-C: bundle-intent маркеры. Запросы, где клиент явно просит "под ключ /
# комплект / что входит / пакет". Для таких запросов стандартный ретривер
# часто возвращает товары/сделки выше bundle-доков, поэтому мы добавляем
# таргетированный doc_type=bundle фетч поверх обычного reranked.
_BUNDLE_INTENT_MARKERS = [
    re.compile(r"под\s+ключ", re.IGNORECASE),
    re.compile(r"что\s+(входит|включ|идёт|идет)", re.IGNORECASE),
    re.compile(r"из\s+чего\s+состо|какой\s+состав|состав\s+и\s+стоимост", re.IGNORECASE),
    re.compile(r"\bкомплект\w*|\bкомплектац", re.IGNORECASE),
    re.compile(r"\bпакет\s+(услуг|рекламн|для|на)|\bпакет\w*\s+для", re.IGNORECASE),
    re.compile(r"\bвесь\s+пакет|\bполный\s+пакет", re.IGNORECASE),
    re.compile(r"\bнабор\s+(рекламн|материал|услуг|для|монтаж)|\bполный\s+набор", re.IGNORECASE),
    re.compile(r"комплексн\w*\s+(печатн|рекламн|кампани|проект)", re.IGNORECASE),
    re.compile(r"брендирован\w*|брендинг", re.IGNORECASE),
    re.compile(r"всё\s+для\s+открыт|все\s+для\s+открыт|для\s+открыт\w*\s+(кафе|магазин|аптек|ресторан|офис|точк)", re.IGNORECASE),
]


def _is_bundle_intent_query(query: str) -> bool:
    """П8.7-C: True если запрос явно про комплект/пакет/под-ключ/состав.
    В этом случае применяем таргетированный bundle-фетч, чтобы гарантированно
    вернуть suggested_bundle в ответе."""
    if not query:
        return False
    return any(rx.search(query) for rx in _BUNDLE_INTENT_MARKERS)


# П7.1-hotfix: маркеры «не оценка, а помощь с текстом/описанием/переговорами».
# При попадании — SmetaEngine пропускаем и идём в LLM.
_DESCRIBE_INTENT_MARKERS = [
    "помоги составить описание", "составить описание", "корректное описание",
    "корректное текстовое описание", "текстовое описание", "помоги с описанием",
    "опиши смету", "опиши вывеску", "опиши", "напиши текст", "напиши описание",
    "как ответить", "что сказать", "что ему ответить", "что ответить",
    "как презентовать", "презентуй", "презентацию",
    "переговоры", "скрипт", "аргумент",
    "выстроить переговоры",
    # П8.8-F: расширение manager-script — клиент просит сформулировать ответ,
    # а не посчитать цену. См. fb#16, fb#33, fb#47.
    "подготовить текст", "подготовь текст", "подготовить ответ", "подготовь ответ",
    "сформулируй", "сформулировать",
    "первого ответа", "первый ответ", "ответа клиенту", "ответ клиенту",
    "клиент спрашивает", "клиент ответил", "клиент сомневается",
    "клиент пишет", "клиент написал", "клиент хочет",
    "три идеи", "3 идеи", "несколько идей",
]


def _is_deal_estimate_query(query: str) -> bool:
    """Detect if the user wants a deal estimate (list of products for Bitrix24).

    П8.5: strong L3 keyword override → форсим estimate-путь даже если в запросе
    нет классических ценовых ключей. «Нужен новый логотип», «Лого для кофейни»
    — это по определению заявки на оценку конкретного шаблонного продукта;
    без этого SmetaEngine блок пропускался, и LLM fallback возвращал guided
    вместо auto.
    """
    q_lower = query.lower()
    if any(kw in q_lower for kw in _ESTIMATE_KEYWORDS):
        return True
    if has_strong_keyword_override(query):
        return True
    return False


def _is_underkey_intent(query: str) -> bool:
    q_lower = query.lower()
    return any(m in q_lower for m in _UNDERKEY_INTENT_MARKERS)


def _is_describe_intent(query: str) -> bool:
    """True если пользователь просит помочь с текстом/описанием/переговорами,
    а не построить смету. SmetaEngine для таких запросов бесполезен."""
    q_lower = query.lower()
    return any(m in q_lower for m in _DESCRIBE_INTENT_MARKERS)


def _looks_like_user_provided_smeta(query: str) -> bool:
    """True если пользователь сам прислал готовую смету (таблица с позициями).
    Эвристика: ≥5 чисел + ≥2 единиц измерения (шт/м²/мп) + длина > 400.
    """
    if len(query) < 400:
        return False
    import re as _re
    nums = _re.findall(r"\b\d[\d\s]{2,}\b", query)
    units = sum(query.lower().count(u) for u in ("шт", "м²", "кв.м", "мп", "м2"))
    return len(nums) >= 5 and units >= 2


# П7.1 #1: in-memory LRU кэш SmetaResult по нормализованному запросу + decomp.
# Снимает 60-90с на повторных запросах от менеджеров (см. чаты 47–50).
_SMETA_CACHE_MAX = 256
_SMETA_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_SMETA_CACHE_LOCK = Lock()


def _smeta_cache_key(query: str, decomp_dict: dict | None) -> tuple:
    norm = " ".join(query.lower().split())
    d = decomp_dict or {}
    return (
        norm,
        int(d.get("letter_count", 0) or 0),
        int(d.get("height_cm", 0) or 0),
        float(d.get("linear_meters", 0) or 0),
    )


def _smeta_cache_get(key: tuple):
    with _SMETA_CACHE_LOCK:
        v = _SMETA_CACHE.get(key)
        if v is not None:
            _SMETA_CACHE.move_to_end(key)
        return v


def _smeta_cache_put(key: tuple, value) -> None:
    with _SMETA_CACHE_LOCK:
        _SMETA_CACHE[key] = value
        _SMETA_CACHE.move_to_end(key)
        while len(_SMETA_CACHE) > _SMETA_CACHE_MAX:
            _SMETA_CACHE.popitem(last=False)


def _emit_request_trace(
    *,
    query: str,
    intent: str,
    intent_confidence: float,
    intent_method: str,
    reranked: list[dict],
    smeta_pre=None,
    smeta_blocked_by: str = "",
    bridge_forced: bool = False,
    feedback_prefix: str = "",
    extra: dict | None = None,
) -> None:
    """P10/A7: единая телеметрия про каждый LLM-вызов.

    Нужна чтобы диагностировать дефекты retrieval/intent/smeta/feedback без
    повторной отладки кейса — каждая инъекция и каждый блокер виден в логах.
    """
    try:
        from collections import Counter
        doc_types = Counter(
            (d.get("payload") or {}).get("doc_type", "?")
            for d in reranked[:15]
        )
        smeta_info: dict = {}
        if smeta_pre is not None:
            smeta_info = {
                "category": getattr(smeta_pre, "category_name", "") or "",
                "total": int(round(getattr(smeta_pre, "total", 0) or 0)),
                "confidence": getattr(smeta_pre, "confidence", "") or "",
            }
        if smeta_blocked_by:
            smeta_info["blocked_by"] = smeta_blocked_by
        rules_count = 0
        lessons_count = 0
        if feedback_prefix:
            rules_count = 1 if "ПРАВИЛА ОТВЕТА" in feedback_prefix else 0
            lessons_count = feedback_prefix.count("сходство ")
        payload = {
            "query_head": query[:80],
            "intent": intent,
            "intent_confidence": round(float(intent_confidence), 3),
            "intent_method": intent_method,
            "retrieved_doc_types": dict(doc_types),
            "bridge_forced": bool(bridge_forced),
            "smeta_pre": smeta_info,
            "feedback_rules_block": rules_count,
            "feedback_lessons_matched": lessons_count,
        }
        if extra:
            payload.update(extra)
        logger.info("request_trace", **payload)
    except Exception as e:
        logger.warning("request_trace emit failed", error=str(e))


def _force_inject_bridge(retriever, reranked: list[dict], service_name: str) -> bool:
    """P10/A3: гарантированная доставка service_pricing_bridge-документа в контекст LLM.

    Вызывается сразу после того, как SmetaEngine заблокирован из-за
    `_BRIDGE_CATEGORIES`. Semantic retrieval может не подтянуть bridge-документ
    (особенно для intent=underspec/product_query с коротким запросом), поэтому
    делаем direct point-lookup по Qdrant filter (doc_type=service_pricing_bridge
    AND service=<name>) и кладём результат в index=0 `reranked` с синтетическим
    rrf_score выше текущего топа.

    Идемпотентна: если документ уже в reranked, просто поднимает его наверх.
    Возвращает True при успешной инжекции.
    """
    if retriever is None or not getattr(retriever, "is_ready", False) or not service_name:
        return False
    # Idempotent: document already present — just promote to top.
    for i, d in enumerate(reranked):
        payload = d.get("payload") or {}
        if payload.get("doc_type") == "service_pricing_bridge" \
                and payload.get("service") == service_name:
            if i > 0:
                reranked.insert(0, reranked.pop(i))
            return True
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        flt = Filter(must=[
            FieldCondition(key="doc_type", match=MatchValue(value="service_pricing_bridge")),
            FieldCondition(key="service", match=MatchValue(value=service_name)),
        ])
        scroll_res, _ = retriever._client.scroll(
            collection_name=retriever.settings.qdrant_collection,
            scroll_filter=flt,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not scroll_res:
            logger.warning("force_inject_bridge: no bridge doc found",
                           service=service_name)
            return False
        pt = scroll_res[0]
        top_score = (reranked[0].get("rrf_score") or reranked[0].get("final_score") or 1.0) \
            if reranked else 1.0
        synthetic_score = max(float(top_score) + 1.0, 1.0)
        reranked.insert(0, {
            "doc_id": pt.payload.get("doc_id", f"bridge_{service_name}"),
            "score": synthetic_score,
            "rrf_score": synthetic_score,
            "final_score": synthetic_score,
            "bm25_score": 0.0,
            "payload": pt.payload,
            "forced_bridge": True,
        })
        logger.info("bridge force-injected", service=service_name,
                    doc_id=pt.payload.get("doc_id"))
        return True
    except Exception as e:
        logger.warning("force_inject_bridge failed", service=service_name, error=str(e))
        return False


def _fetch_linked_offers(retriever, offer_ids: list[int], limit_per_type: int = 3) -> list[dict]:
    """P10.5-IV: достать offer_composition / offer_profile по списку offer_id.

    Вызывается после `_force_inject_bridge` для закрытия G2 (bridge.offer_ids
    видят в prompt, но без самого состава КП LLM не может сослаться на реальные
    позиции). Сначала ищем doc_type=offer_composition (там есть products[]),
    если мало — добавляем offer_profile как fallback.

    P10.6 B1+B4: fallback через offer_profile теперь работает — после ingest
    offer_profile.payload содержит `offer_id: int` (раньше там был только
    `deal_id`, и фильтр offer_id IN [...] возвращал пусто). offer_composition
    теперь тоже эмитится по умолчанию на все offers с ≥2 line-items (раньше
    был whitelist 22 KEY_OFFER_IDS).

    Returns: list of reranked-style dicts (same shape as reranked entries),
    до 2*limit_per_type элементов. Пустой список — если retriever не готов.
    """
    if retriever is None or not getattr(retriever, "is_ready", False) or not offer_ids:
        return []
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
    except Exception:
        return []

    ids_as_str = [str(i) for i in offer_ids]
    ids_as_int = [int(i) for i in offer_ids if str(i).isdigit()]

    out: list[dict] = []
    seen_offer_ids: set[str] = set()

    def _scan(doc_type: str, match_values) -> list:
        try:
            flt = Filter(must=[
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type)),
                FieldCondition(key="offer_id", match=MatchAny(any=match_values)),
            ])
            res, _ = retriever._client.scroll(
                collection_name=retriever.settings.qdrant_collection,
                scroll_filter=flt,
                limit=limit_per_type,
                with_payload=True, with_vectors=False,
            )
            return res or []
        except Exception:
            return []

    # offer_composition — богатый источник (products[], total_price)
    for match_set in (ids_as_int, ids_as_str):
        if not match_set:
            continue
        hits = _scan("offer_composition", match_set)
        for pt in hits:
            oid = str(pt.payload.get("offer_id", ""))
            if oid and oid not in seen_offer_ids:
                seen_offer_ids.add(oid)
                out.append({
                    "doc_id": pt.payload.get("doc_id", f"oc_{oid}"),
                    "score": 0.9, "rrf_score": 0.9, "final_score": 0.9,
                    "bm25_score": 0.0,
                    "payload": pt.payload,
                    "linked_from_bridge": True,
                })
        if out:
            break  # int/str — только одна схема хранения

    # offer_profile — fallback (если offer_composition не нашёлся)
    if len(out) < limit_per_type:
        remaining = limit_per_type - len(out)
        for match_set in (ids_as_int, ids_as_str):
            if not match_set:
                continue
            hits = _scan("offer_profile", match_set)
            # Cap по remaining
            for pt in hits[:remaining]:
                oid = str(pt.payload.get("offer_id", "")
                          or pt.payload.get("deal_id", ""))
                if oid and oid not in seen_offer_ids:
                    seen_offer_ids.add(oid)
                    out.append({
                        "doc_id": pt.payload.get("doc_id", f"op_{oid}"),
                        "score": 0.85, "rrf_score": 0.85, "final_score": 0.85,
                        "bm25_score": 0.0,
                        "payload": pt.payload,
                        "linked_from_bridge": True,
                    })
            if len(out) >= limit_per_type:
                break

    return out


def _inject_linked_offers_after_bridge(retriever, reranked: list[dict],
                                        max_offers: int = 3) -> int:
    """P10.5-IV: если reranked[0] — bridge, подмешивает в индекс 1..N связанные
    offer_composition/offer_profile по offer_ids из packages. Возвращает число
    добавленных docs (0 если bridge отсутствует наверху или нет связанных КП).
    """
    if not reranked:
        return 0
    top = reranked[0]
    payload = top.get("payload") or {}
    if payload.get("doc_type") != "service_pricing_bridge":
        return 0
    # Собираем offer_ids из всех packages
    offer_ids: list[int] = []
    for pkg in (payload.get("packages") or []):
        for oid in (pkg.get("offer_ids") or []):
            if oid not in offer_ids:
                offer_ids.append(oid)
    if not offer_ids:
        return 0
    linked = _fetch_linked_offers(retriever, offer_ids, limit_per_type=max_offers)
    if not linked:
        return 0
    # Уже в reranked? — пропускаем дубликаты
    existing_ids = {(d.get("payload") or {}).get("doc_id") for d in reranked}
    new_linked = [d for d in linked if d["doc_id"] not in existing_ids]
    # Вставляем сразу после bridge (index=1)
    for i, d in enumerate(new_linked):
        reranked.insert(1 + i, d)
    logger.info("linked offers injected after bridge",
                bridge_service=payload.get("service"),
                offer_ids=offer_ids[:5], injected=len(new_linked))
    return len(new_linked)


def _parse_deal_items(raw_json: dict) -> list[DealItem]:
    """Extract and validate deal_items from LLM JSON output."""
    raw_items = raw_json.get("deal_items")
    if not raw_items or not isinstance(raw_items, list):
        return []
    result = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            qty = float(item.get("quantity", 1) or 1)
            unit_price = float(item.get("unit_price", 0) or 0)
            total = float(item.get("total", 0) or 0)
            if total == 0 and unit_price > 0:
                total = round(qty * unit_price, 2)
            result.append(DealItem(
                product_name=str(item.get("product_name", "")),
                quantity=qty,
                unit=str(item.get("unit", "шт")),
                unit_price=unit_price,
                total=total,
                b24_section=str(item.get("b24_section", "")),
                notes=str(item.get("notes", "")),
            ))
        except (ValueError, TypeError):
            continue
    return result


def _build_size_context(decomp: "QueryDecomposition") -> str:
    """Build size-aware pricing context when query has letter/height info."""
    if decomp.letter_count == 0:
        return ""
    lines = [
        f"ПАРАМЕТРЫ ЗАПРОСА: надпись '{decomp.letter_text}' "
        f"({decomp.letter_count} букв)",
    ]
    if decomp.technology:
        lines.append(f"Технология: {decomp.technology}")
    if decomp.height_cm > 0:
        lines.append(f"Высота: {decomp.height_cm:.0f} см")
        # Use pricing_resolver's rate tables for consistency
        from app.core.pricing_resolver import PricingResolver
        rate_min, rate_max = PricingResolver._get_letter_rate(decomp.height_cm, decomp.technology)
        low = rate_min * decomp.letter_count
        high = rate_max * decomp.letter_count
        lines.append(
            f"Рыночный ориентир по размеру: "
            f"{low:,} – {high:,} руб (только буквы, без монтажа/каркаса)"
        )
    else:
        lines.append("Высота НЕ указана — уточни у клиента для точного расчёта")
    if decomp.linear_meters > 0:
        lines.append(f"Расчётные погонные метры: {decomp.linear_meters:.1f} мп")
    return "\n".join(lines)


def _format_pricing_breakdown(pr) -> str:
    """Format pre-calculated pricing breakdown for LLM context."""
    if pr is None or pr.total_under_key_min is None:
        return ""
    lines = [
        f"РАСЧЁТ СИСТЕМЫ: итого под ключ {pr.total_under_key_min:,.0f} – {pr.total_under_key_max:,.0f} руб"
    ]
    if pr.price_breakdown:
        parts = []
        for name, (lo, hi) in pr.price_breakdown.items():
            parts.append(f"{name}: {lo:,.0f}–{hi:,.0f}")
        lines.append(f"({' | '.join(parts)})")
    lines.append("Используй эти цифры как основу ответа. НЕ пересчитывай.")
    return "\n".join(lines)


def _build_references(docs: list[dict]) -> list[Reference]:
    """Build Reference objects from retrieved docs. Only show relevant ones."""
    refs = []
    for doc in docs[:8]:
        score = doc.get("final_score", doc.get("rrf_score", 0.0))
        if score < MIN_RELEVANT_SCORE:
            continue  # skip noise docs
        payload = doc.get("payload", {})
        searchable_text = payload.get("searchable_text", "")
        doc_type = payload.get("doc_type", "")
        article_id = None
        if doc_type == "product":
            article_id = str(payload.get("product_id") or payload.get("product_key") or "") or None
        elif doc_type in ("deal_profile", "offer_profile"):
            article_id = str(payload.get("deal_id") or "") or None
        refs.append(Reference(
            doc_id=payload.get("doc_id", ""),
            doc_type=doc_type,
            score=round(score, 4),
            snippet=searchable_text[:120],
            article_id=article_id,
            product_name=payload.get("product_name") or payload.get("title") or None,
            direction=payload.get("direction") or None,
        ))
    return refs


def _dominant_direction(docs: list[dict], top: int = 10) -> str | None:
    """Return direction appearing in >=60% of top docs, else None."""
    from collections import Counter
    dirs = [d.get("payload", {}).get("direction") for d in docs[:top] if d.get("payload", {}).get("direction")]
    if not dirs:
        return None
    counter = Counter(dirs)
    top_dir, freq = counter.most_common(1)[0]
    return top_dir if freq / len(dirs) >= 0.6 else None


def _seg_from_doc(doc: dict, kind: str, photo_enrich: dict | None = None) -> SourceSegment:
    p = doc.get("payload", {})
    score = round(doc.get("final_score", doc.get("rrf_score", 0.0)), 4)
    text = p.get("searchable_text", "")
    deal_id = str(p.get("deal_id") or "") or None

    title = (p.get("title") or p.get("sample_title") or p.get("deal_title")
             or p.get("sample_order_title") or "").strip() or (text[:60] if text else "")
    direction = p.get("direction") or None
    total = p.get("line_total") or p.get("median_deal_value") or p.get("deal_total")
    duration = p.get("deal_duration_days") or p.get("median_duration_days")
    image_urls = list(p.get("image_urls") or [])

    product_type = p.get("product_type") or None
    application = p.get("application") or None
    roi_hint = None
    dimensions = p.get("dimensions") or None

    # Enrich from photo_index for deal/offer segments when doc payload lacks visuals
    if photo_enrich:
        if not image_urls and photo_enrich.get("image_urls"):
            image_urls = list(photo_enrich["image_urls"])
        product_type = product_type or photo_enrich.get("product_type") or None
        application = application or photo_enrich.get("application") or None
        dimensions = dimensions or photo_enrich.get("dimensions") or None
        roi = photo_enrich.get("roi_romi_avg")
        if roi:
            roi_hint = f"ROMI ~{int(roi)}%"

    subtitle_parts = [x for x in [product_type, dimensions] if x]
    subtitle = " · ".join(subtitle_parts) if subtitle_parts else None

    return SourceSegment(
        kind=kind,
        deal_id=deal_id,
        title=title[:140],
        subtitle=subtitle,
        direction=direction,
        total=float(total) if isinstance(total, (int, float)) else None,
        duration_days=int(duration) if isinstance(duration, (int, float)) else None,
        snippet=text[:180],
        score=score,
        image_urls=image_urls[:4],
        product_type=product_type,
        application=application,
        roi_hint=roi_hint,
    )


def _build_segmented_references(docs: list[dict], photo_index=None) -> SegmentedReferences:
    """Partition retrieved docs into three thematic segments.

    - similar_orders: doc_type == deal_profile (top 7), enriched via photo_index.
    - similar_offers: doc_type == offer_profile, fallback to bundle[dataset=offers] (top 5).
    - product_links: doc_type == photo_analysis with image_urls (top 6), deduped by deal_id.
    """
    def _dir_hard_filter(doc, dom):
        if not dom:
            return True
        d = doc.get("payload", {}).get("direction")
        return (not d) or d == dom

    dom_dir = _dominant_direction(docs)
    relevant = [d for d in docs
                if d.get("final_score", d.get("rrf_score", 0.0)) >= MIN_RELEVANT_SCORE]

    seen_deals: set[str] = set()
    similar_orders: list[SourceSegment] = []
    similar_offers: list[SourceSegment] = []
    product_links: list[SourceSegment] = []
    seen_pt_dir: set[tuple] = set()

    # 1) similar_orders — deal_profile
    for doc in relevant:
        if len(similar_orders) >= 7:
            break
        p = doc.get("payload", {})
        if p.get("doc_type") != "deal_profile":
            continue
        if not _dir_hard_filter(doc, dom_dir):
            continue
        deal_id = str(p.get("deal_id") or "")
        if deal_id and deal_id in seen_deals:
            continue
        enrich = photo_index.get(deal_id) if photo_index and deal_id else None
        similar_orders.append(_seg_from_doc(doc, "order", enrich))
        if deal_id:
            seen_deals.add(deal_id)

    # 2) similar_offers — prefer offer_profile, fallback to bundle[offers]
    offer_profiles = [d for d in relevant
                      if d.get("payload", {}).get("doc_type") == "offer_profile"]
    if offer_profiles:
        for doc in offer_profiles:
            if len(similar_offers) >= 5:
                break
            if not _dir_hard_filter(doc, dom_dir):
                continue
            deal_id = str(doc.get("payload", {}).get("deal_id") or "")
            if deal_id and deal_id in seen_deals:
                continue
            enrich = photo_index.get(deal_id) if photo_index and deal_id else None
            similar_offers.append(_seg_from_doc(doc, "offer", enrich))
            if deal_id:
                seen_deals.add(deal_id)
    else:
        for doc in relevant:
            if len(similar_offers) >= 5:
                break
            p = doc.get("payload", {})
            if p.get("doc_type") != "bundle":
                continue
            if p.get("dataset_type") != "offers":
                continue
            if not _dir_hard_filter(doc, dom_dir):
                continue
            similar_offers.append(_seg_from_doc(doc, "offer"))

    # 3) product_links — photo_analysis with image_urls, dedup by (deal_id) and (product_type, direction)
    for doc in relevant:
        if len(product_links) >= 6:
            break
        p = doc.get("payload", {})
        if p.get("doc_type") != "photo_analysis":
            continue
        urls = p.get("image_urls") or []
        if not urls:
            continue
        if not _dir_hard_filter(doc, dom_dir):
            continue
        deal_id = str(p.get("deal_id") or "")
        if deal_id and deal_id in seen_deals:
            continue
        pt_key = (p.get("product_type") or "", p.get("direction") or "")
        if pt_key in seen_pt_dir and pt_key != ("", ""):
            continue
        product_links.append(_seg_from_doc(doc, "product_visual"))
        if deal_id:
            seen_deals.add(deal_id)
        seen_pt_dir.add(pt_key)

    return SegmentedReferences(
        similar_orders=similar_orders,
        similar_offers=similar_offers,
        product_links=product_links,
    )


def _filter_relevant_docs(docs: list[dict]) -> list[dict]:
    """Filter out docs with near-zero relevance scores to prevent noise pricing."""
    relevant = [d for d in docs if d.get("final_score", d.get("rrf_score", 0.0)) >= MIN_RELEVANT_SCORE]
    return relevant if relevant else docs[:3]  # always return at least 3 for LLM context


def _build_suggested_bundle(docs: list[dict]) -> list[BundleItem]:
    """Build BundleItem list from the first bundle doc in the candidate list.

    П8.7-C: scan ALL provided docs (not just top 5) — bundle docs may rank
    below product/deal_profile in mixed retrieval, but we still want to
    surface them for bundle-intent queries. For dedicated bundle fetches,
    the first doc IS the best bundle."""
    items = []
    for doc in docs:
        payload = doc.get("payload", {})
        if payload.get("doc_type") != "bundle":
            continue
        bundle_key = payload.get("bundle_key", "")
        product_keys = bundle_key.split("|") if bundle_key else []
        for key in product_keys[:6]:
            if key:
                items.append(BundleItem(
                    product_key=key.strip(),
                    product_name=f"Продукт {key.strip()}",
                    direction=payload.get("direction", ""),
                ))
        if items:
            break  # only top bundle doc
    return items


def _detect_source_distinction(docs: list[dict]) -> SourceDistinction:
    """Determine which data sources contributed to the response."""
    has_orders = False
    has_offers = False
    for doc in docs:
        payload = doc.get("payload", {})
        dataset = payload.get("dataset_type", "")
        if dataset == "orders" or payload.get("order_rows", 0) > 0:
            has_orders = True
        if dataset == "offers" or payload.get("offer_rows", 0) > 0:
            has_offers = True

    if has_orders and has_offers:
        dataset_type = "both"
    elif has_orders:
        dataset_type = "orders"
    elif has_offers:
        dataset_type = "offers"
    else:
        dataset_type = "catalog_only"

    return SourceDistinction(
        has_order_data=has_orders,
        has_offer_data=has_offers,
        dataset_type=dataset_type,
        data_freshness_note="Данные по заказам до 2026-03",
    )


def _build_parametric_breakdown(
    estimate: "param_calc.ParametricEstimate",
    decomp: QueryDecomposition,
) -> ParametricBreakdown:
    """Convert ParametricEstimate + QueryDecomposition to Pydantic schema."""
    items = [
        ParametricLineItem(
            component=li.component,
            label=li.label,
            product_name=li.product_name,
            unit_price=li.unit_price,
            quantity=li.quantity,
            unit=li.unit,
            total=li.total,
            confidence_tier=li.confidence_tier,
            quantity_basis=li.quantity_basis,
        )
        for li in estimate.line_items
    ]
    return ParametricBreakdown(
        line_items=items,
        total_estimate=estimate.total_estimate,
        total_min=estimate.total_min,
        total_max=estimate.total_max,
        missing_components=estimate.missing_components,
        letter_text=decomp.letter_text,
        letter_count=decomp.letter_count,
        height_cm=decomp.height_cm,
        linear_meters=decomp.linear_meters,
    )


async def _handle_complex_query(
    query: str,
    decomp: QueryDecomposition,
    retriever,
    reranker,
    generator,
    top_k: int,
) -> dict:
    """
    Pipeline for complex multi-component queries:
    1. Multi-retrieve: separate search per component (products, for parametric)
    2. Parametric calculate: internal production costs (reference only, NOT retail price)
    3. Standard retrieve: bundles + context with real deal prices
    4. Generate LLM summary using real bundle prices + market knowledge
    """
    # 1. Multi-retrieve per component (products only — for parametric breakdown)
    doc_pools = retriever.multi_retrieve(decomp.components)

    # 2. Parametric estimate — internal production norms, NOT retail prices
    estimate = param_calc.calculate(decomp.components, doc_pools)
    breakdown_text = param_calc.format_breakdown(estimate)

    # 3. Standard retrieval for bundles with real deal prices (primary price reference)
    standard_candidates = retriever.retrieve(query, top_k=top_k * 2)
    reranked = reranker.rerank(query, standard_candidates, top_n=top_k)

    # 4. Build context for LLM
    parametric_context = (
        f"Запрос: {query}\n\n"
        f"Параметры: буквы='{decomp.letter_text}' ({decomp.letter_count} букв), "
        f"высота={decomp.height_cm:.0f} см, линейные метры≈{decomp.linear_meters:.1f} мп\n\n"
        f"ВНИМАНИЕ: следующие данные — внутренние нормы производства (себестоимость операций), "
        f"НЕ розничные цены. Для розничной цены используй данные [Набор] из контекста.\n"
        f"{breakdown_text}"
    )

    return {
        "doc_pools": doc_pools,
        "estimate": estimate,
        "reranked": reranked,
        "parametric_context": parametric_context,
    }


@router.post("/query", response_model=HumanQueryResponse)
async def query_human(req: QueryRequest, request: Request,
                      user: dict | None = Depends(get_optional_user)) -> HumanQueryResponse:
    """
    Query the RAG system and get a human-readable Russian response.
    """
    t0 = time.monotonic()

    retriever = request.app.state.retriever
    reranker = request.app.state.reranker
    generator = request.app.state.generator
    pricing = request.app.state.pricing
    vision = getattr(request.app.state, "vision", None)

    if not retriever or not retriever.is_ready:
        raise HTTPException(503, "Retriever not ready. Index may not be built yet.")

    # Load chat history from DB when chat_id is provided
    if req.chat_id and user:
        db_history = get_chat_history(req.chat_id, user["id"], limit=12)
        if db_history and not req.history:
            req.history = [ChatMessage(role=m["role"], content=m["content"]) for m in db_history]

    try:
        # Vision analysis (if image provided)
        vision_context = ""
        if req.image_base64 and vision and vision.is_available:
            vision_context = await vision.analyze(req.image_base64, req.image_mime_type)
            if vision_context:
                logger.info("Vision analysis prepended to query context")

        extra_ctx = f"АНАЛИЗ ИЗОБРАЖЕНИЯ:\n{vision_context}\n\n" if vision_context else ""

        # --- Parse query for clarification needs ---
        parsed = parse_query(req.query)
        if parsed.needs_clarification:
            extra_ctx += "ВАЖНО: Запрос слишком общий — не указан тип изделия, размеры или направление. Задай 1-2 уточняющих вопроса клиенту ВМЕСТО угадывания цены. Дай примерные диапазоны по типам.\n\n"

        # --- Decompose query for size-aware pricing ---
        decomp = decompose(req.query)

        if decomp.is_complex and decomp.components:
            # Full parametric pipeline (same as query_structured)
            logger.info("Complex query in /query — using parametric pipeline",
                        letter_text=decomp.letter_text,
                        letter_count=decomp.letter_count,
                        height_cm=decomp.height_cm)

            result = await _handle_complex_query(
                req.query, decomp, retriever, reranker, generator, req.top_k
            )
            reranked = result["reranked"]
            extra_ctx += result["parametric_context"]
            pricing_resolution = pricing.resolve(reranked, decomp=decomp)

            # Inject pre-calculated breakdown
            breakdown_ctx = _format_pricing_breakdown(pricing_resolution)
            if breakdown_ctx:
                extra_ctx += "\n\n" + breakdown_ctx
        else:
            # Standard retrieval with optional size context
            candidates = retriever.retrieve(req.query, top_k=req.top_k * 2)
            if not candidates:
                return HumanQueryResponse(
                    summary="По вашему запросу не найдено подходящих товаров/услуг в базе данных.",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )

            reranked = reranker.rerank(req.query, candidates, top_n=req.top_k)
            pricing_resolution = pricing.resolve(reranked, decomp=decomp)

            # Inject size context for queries with letter info
            size_ctx = _build_size_context(decomp)
            if size_ctx:
                extra_ctx += size_ctx + "\n\n"

            # Inject pre-calculated breakdown so LLM doesn't do arithmetic
            breakdown_ctx = _format_pricing_breakdown(pricing_resolution)
            if breakdown_ctx:
                extra_ctx += breakdown_ctx + "\n\n"

        # Inject feedback-based learning context (RLHF).
        # P10/A6: rules (Tier 2) собираются всегда, lessons (Tier 1) —
        # только если retriever готов (нужен embedding запроса). Любой
        # silent-fail embedding теперь не обнуляет обязательные правила.
        feedback_store = getattr(request.app.state, "feedback_store", None)
        if feedback_store is not None:
            try:
                q_vec_fb = None
                if retriever.is_ready:
                    try:
                        q_vec_fb = retriever.embed_query(req.query)
                    except Exception as _e_emb:
                        logger.warning("Feedback embed_query failed", error=str(_e_emb))
                direction = getattr(decomp, "direction", "") or ""
                fb_ctx = build_feedback_context(feedback_store, q_vec_fb, direction)
                if fb_ctx:
                    extra_ctx = fb_ctx + "\n\n" + extra_ctx
                    logger.info("Feedback context injected", lessons_matched=fb_ctx.count("["))
            except Exception as e:
                logger.warning("Feedback context failed", error=str(e))

        # Generate
        summary = await generator.generate(
            req.query, reranked, pricing_resolution,
            history=req.history, extra_context=extra_ctx,
        )

        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info("Query completed", latency_ms=latency_ms, sources=len(reranked))

        # Persist messages to chat if authenticated
        if req.chat_id and user:
            try:
                save_message(req.chat_id, user["id"], "user", req.query, mode="simple")
                save_message(req.chat_id, user["id"], "assistant", summary,
                             mode="simple", latency_ms=latency_ms)
            except Exception as e:
                logger.warning("Failed to save chat message", error=str(e))

        return HumanQueryResponse(
            summary=summary,
            latency_ms=latency_ms,
            sources_count=len(reranked),
        )

    except Exception as e:
        logger.error("Query failed", error=str(e), query=req.query[:100])
        raise HTTPException(500, f"Query failed: {str(e)}")


@router.post("/query_structured", response_model=StructuredResponse, response_model_exclude_none=True)
async def query_structured(req: QueryRequest, request: Request,
                           user: dict | None = Depends(get_optional_user)) -> StructuredResponse:
    """
    Query the RAG system and get a structured JSON response with pricing, bundle, and evidence.
    """
    t0 = time.monotonic()

    retriever = request.app.state.retriever
    reranker = request.app.state.reranker
    generator = request.app.state.generator
    pricing = request.app.state.pricing
    vision = getattr(request.app.state, "vision", None)

    if not retriever or not retriever.is_ready:
        raise HTTPException(503, "Retriever not ready.")

    # Load chat history from DB when chat_id is provided
    if req.chat_id and user:
        db_history = get_chat_history(req.chat_id, user["id"], limit=12)
        if db_history and not req.history:
            req.history = [ChatMessage(role=m["role"], content=m["content"]) for m in db_history]

    try:
        # Vision analysis (if image provided)
        vision_context = ""
        if req.image_base64 and vision and vision.is_available:
            vision_context = await vision.analyze(req.image_base64, req.image_mime_type)
            if vision_context:
                logger.info("Vision analysis prepended to structured query context")

        # --- Parse query for clarification needs ---
        parsed = parse_query(req.query)
        clarification_prefix = ""
        if parsed.needs_clarification:
            clarification_prefix = "ВАЖНО: Запрос слишком общий — не указан тип изделия, размеры или направление. Задай 1-2 уточняющих вопроса клиенту ВМЕСТО угадывания цены. Дай примерные диапазоны по типам.\n\n"

        # --- Intent classification (when enabled) ---
        _has_history = bool(getattr(req, "history", None) or getattr(req, "chat_id", None))
        intent_result: IntentResult | None = None
        intent_instruction = ""
        _use_intent = app_settings.use_intent_classifier
        if _use_intent:
            _clf = get_classifier()
            if _clf is not None and _clf.is_ready:
                intent_result = _clf.classify(req.query, has_history=_has_history)
                intent_instruction = _get_intent_instruction(intent_result.intent)
                logger.info("Intent classified",
                            intent=intent_result.intent,
                            confidence=round(intent_result.confidence, 3),
                            method=intent_result.method,
                            query=req.query[:80])

        # --- Decompose query: detect complex multi-component requests ---
        decomp = decompose(req.query)
        parametric_breakdown_schema: ParametricBreakdown | None = None
        estimate = None
        is_estimate = _is_deal_estimate_query(req.query)

        # Feedback-based learning context (RLHF).
        # P10/A6: rules (Tier 2) всегда доступны из БД. Lessons (Tier 1) —
        # только когда retriever готов (нужен BGE-M3 embedding запроса).
        feedback_prefix = ""
        feedback_store = getattr(request.app.state, "feedback_store", None)
        if feedback_store is not None:
            try:
                q_vec_fb = None
                if retriever.is_ready:
                    try:
                        q_vec_fb = retriever.embed_query(req.query)
                    except Exception as _e_emb:
                        logger.warning("Feedback embed_query failed (structured)",
                                       error=str(_e_emb))
                direction = getattr(decomp, "direction", "") or ""
                feedback_prefix = build_feedback_context(feedback_store, q_vec_fb, direction)
                if feedback_prefix:
                    logger.info("Feedback context injected (structured)",
                                lessons_matched=feedback_prefix.count("["))
            except Exception as e:
                logger.warning("Feedback context failed", error=str(e))

        if decomp.is_complex and decomp.components:
            logger.info("Complex query detected — using parametric pipeline",
                        letter_text=decomp.letter_text,
                        letter_count=decomp.letter_count,
                        height_cm=decomp.height_cm,
                        linear_meters=decomp.linear_meters,
                        components=[c.type for c in decomp.components])

            result = await _handle_complex_query(
                req.query, decomp, retriever, reranker, generator, req.top_k
            )
            estimate = result["estimate"]
            reranked = result["reranked"]
            parametric_context = result["parametric_context"]

            if estimate.is_parametric:
                parametric_breakdown_schema = _build_parametric_breakdown(estimate, decomp)

            # Generate with parametric context (+ optional vision analysis + feedback)
            full_extra = parametric_context
            if clarification_prefix:
                full_extra = clarification_prefix + full_extra
            if feedback_prefix:
                full_extra = feedback_prefix + "\n\n" + full_extra
            if vision_context:
                full_extra = f"АНАЛИЗ ИЗОБРАЖЕНИЯ:\n{vision_context}\n\n---\n{full_extra}"
            pr_obj = type("PR", (), {
                "confidence": estimate.confidence,
                "estimated_value": estimate.total_estimate,
                "estimated_basis": "параметрический расчёт по компонентам",
                "price_band_min": estimate.total_min,
                "price_band_max": estimate.total_max,
                "flags": (
                    [f"Не найдено в базе: {', '.join(estimate.missing_components)}"]
                    if estimate.missing_components else []
                ),
                "risks": [],
                "is_financial_modifier": False,
            })()
            # П7.1-hotfix: describe-intent / user-provided-smeta → SmetaEngine не нужен
            _force_llm = _is_describe_intent(req.query) or _looks_like_user_provided_smeta(req.query)
            if _force_llm:
                logger.info("SmetaEngine skipped (describe/provided-smeta intent)")
            if is_estimate and not _force_llm:
                # П7.1 #2: pre-LLM SmetaEngine gating (complex/parametric path).
                _smeta_engine_pre_c = getattr(request.app.state, "smeta_engine", None)
                _smeta_pre_usable_c = False
                _smeta_pre_c = None
                if _smeta_engine_pre_c is not None and _smeta_engine_pre_c.is_ready:
                    try:
                        _decomp_pre_dict_c = {
                            "letter_count": getattr(decomp, "letter_count", 0) or 0,
                            "letter_text": getattr(decomp, "letter_text", "") or "",
                            "height_cm": getattr(decomp, "height_cm", 0) or 0,
                            "linear_meters": getattr(decomp, "linear_meters", 0) or 0,
                        }
                        _cache_key_c = _smeta_cache_key(req.query, _decomp_pre_dict_c)
                        _smeta_pre_c = _smeta_cache_get(_cache_key_c)
                        if _smeta_pre_c is None:
                            _q_vec_pre_c = retriever.embed_query(req.query)
                            _smeta_pre_c = _smeta_engine_pre_c.build_smeta(
                                req.query, _q_vec_pre_c, decomp=_decomp_pre_dict_c,
                            )
                            _smeta_cache_put(_cache_key_c, _smeta_pre_c)
                        else:
                            logger.info("SmetaEngine cache hit (complex)",
                                        category=getattr(_smeta_pre_c, "category_name", ""))
                        # Under-key blacklist
                        if _smeta_pre_c.is_usable and _is_underkey_intent(req.query) \
                                and _smeta_pre_c.category_name in _UNDERKEY_BLACKLIST_CATEGORIES:
                            logger.info("SmetaEngine result blocked by under-key intent (complex)",
                                        category=_smeta_pre_c.category_name)
                            _smeta_pre_c = None
                        # П7.1-hotfix: junk categories + min total gate
                        if _smeta_pre_c is not None and _smeta_pre_c.is_usable \
                                and _smeta_pre_c.category_name in _JUNK_CATEGORIES:
                            logger.info("SmetaEngine junk category blocked (complex)",
                                        category=_smeta_pre_c.category_name)
                            _smeta_pre_c = None
                        # P9: bridge-категории → через LLM с пакетами
                        if _smeta_pre_c is not None and _smeta_pre_c.is_usable \
                                and _smeta_pre_c.category_name in _BRIDGE_CATEGORIES:
                            logger.info("SmetaEngine bridge-category blocked (complex)",
                                        category=_smeta_pre_c.category_name)
                            _smeta_pre_c = None
                        if _smeta_pre_c is not None and _smeta_pre_c.is_usable \
                                and _is_underkey_intent(req.query) \
                                and (_smeta_pre_c.total or 0) < _UNDERKEY_MIN_TEMPLATE_TOTAL:
                            logger.info("SmetaEngine under min total for under-key (complex)",
                                        total=_smeta_pre_c.total)
                            _smeta_pre_c = None
                        if _smeta_pre_c is not None and _smeta_pre_c.is_usable:
                            _smeta_pre_usable_c = True
                            request.state._smeta_precomputed = _smeta_pre_c
                            logger.info("SmetaEngine hit pre-LLM (complex), skipping generate_deal_estimate",
                                        category=_smeta_pre_c.category_name,
                                        sim=round(_smeta_pre_c.match_similarity, 3))
                    except Exception as _e_pre_c:
                        logger.warning("Pre-LLM SmetaEngine failed (complex)", error=str(_e_pre_c))

                if _smeta_pre_usable_c:
                    raw_json = {"summary": "", "reasoning": "", "flags": [], "risks": [],
                                "deal_items": [], "estimated_price": None, "price_band": None,
                                "confidence": None}
                else:
                    # Soft-context: top-1 шаблон → LLM как ориентир
                    try:
                        if _smeta_pre_c is not None and getattr(_smeta_pre_c, "category_name", ""):
                            _soft_total_c = int(round(getattr(_smeta_pre_c, "total", 0) or 0))
                            _soft_lines_c = [
                                "СПРАВОЧНЫЙ ШАБЛОН (для ориентира, не итоговая цена):",
                                f"- категория: «{_smeta_pre_c.category_name}» "
                                f"({getattr(_smeta_pre_c, 'deals_in_category', 0)} сделок, "
                                f"совпадение {getattr(_smeta_pre_c, 'match_similarity', 0):.0%})",
                                f"- ориентировочный итог: {_soft_total_c:,} ₽".replace(",", " "),
                            ]
                            for _it_c in (getattr(_smeta_pre_c, "deal_items", []) or [])[:6]:
                                _soft_lines_c.append(
                                    f"  · {_it_c.product_name} — {_it_c.quantity} {_it_c.unit} × "
                                    f"{int(round(_it_c.unit_price)):,} ₽".replace(",", " ")
                                )
                            full_extra += "\n".join(_soft_lines_c) + "\n\n"
                    except Exception as _e_soft_c:
                        logger.warning("Soft-context inject failed (complex)", error=str(_e_soft_c))

                    # Inject real product names for deal_items.
                    real_names_set = {
                        doc["payload"].get("product_name")
                        for doc in reranked if doc["payload"].get("product_name")
                    }
                    try:
                        catalog_docs = retriever.retrieve_for_component(req.query, top_k=12)
                        for d in catalog_docs:
                            name = d["payload"].get("product_name")
                            if name:
                                real_names_set.add(name)
                    except Exception as e:
                        logger.warning("Catalog retrieval for deal_items failed", error=str(e))
                    real_names = sorted(n for n in real_names_set if n)
                    if real_names:
                        full_extra += "\nНАЗВАНИЯ ТОВАРОВ (используй ТОЛЬКО эти названия в deal_items):\n"
                        full_extra += "\n".join(f"- {n}" for n in real_names[:50]) + "\n\n"
                    raw_json = await generator.generate_deal_estimate(
                        req.query, reranked, pr_obj,
                        extra_context=full_extra, history=req.history,
                        intent_instruction=intent_instruction,
                    )
            else:
                raw_json = await generator.generate_structured(
                    req.query, reranked, pr_obj,
                    extra_context=full_extra, history=req.history,
                    intent_instruction=intent_instruction,
                )

        else:
            # --- Standard pipeline for simple queries ---
            # Use intent-aware retrieval when classifier is active
            if _use_intent and intent_result is not None and intent_result.confidence >= 0.75:
                candidates = retriever.retrieve_by_intent(
                    req.query, intent_result.intent,
                    hints=intent_result.hints, top_k=req.top_k * 2,
                )
            else:
                candidates = retriever.retrieve(req.query, top_k=req.top_k * 2)
            if not candidates:
                return StructuredResponse(
                    summary="По вашему запросу не найдено подходящих товаров/услуг.",
                    confidence="manual",
                    flags=["Нет данных в базе для данного запроса."],
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )

            reranked = reranker.rerank(req.query, candidates, top_n=req.top_k)
            pr = pricing.resolve(reranked, decomp=decomp)
            # Filter noise: only pass relevant docs to generator for pricing context
            reranked_relevant = _filter_relevant_docs(reranked)
            vision_extra = ""
            if clarification_prefix:
                vision_extra += clarification_prefix
            if feedback_prefix:
                vision_extra += feedback_prefix + "\n\n"
            if vision_context:
                vision_extra += f"АНАЛИЗ ИЗОБРАЖЕНИЯ:\n{vision_context}\n\n"
            # Add size context for structured queries with letter info
            size_ctx = _build_size_context(decomp)
            if size_ctx:
                vision_extra += size_ctx + "\n\n"
            # Inject pre-calculated breakdown
            breakdown_ctx = _format_pricing_breakdown(pr)
            if breakdown_ctx:
                vision_extra += breakdown_ctx + "\n\n"
            _force_llm = _is_describe_intent(req.query) or _looks_like_user_provided_smeta(req.query)
            if _force_llm:
                logger.info("SmetaEngine skipped (describe/provided-smeta intent, std)")
            # P10/A7: trace flags
            _bridge_forced = False
            _smeta_blocked_by = ""
            if is_estimate and not _force_llm:
                # П7.1 #2: try SmetaEngine FIRST. If usable, skip the 60-90s LLM
                # generate_deal_estimate call entirely — summary/items/prices are
                # all derived deterministically from the template downstream.
                _smeta_engine_pre = getattr(request.app.state, "smeta_engine", None)
                _smeta_pre_usable = False
                if _smeta_engine_pre is not None and _smeta_engine_pre.is_ready:
                    try:
                        _decomp_pre_dict = None
                        _decomp_pre = locals().get("decomp")
                        if _decomp_pre is not None:
                            _decomp_pre_dict = {
                                "letter_count": getattr(_decomp_pre, "letter_count", 0) or 0,
                                "letter_text": getattr(_decomp_pre, "letter_text", "") or "",
                                "height_cm": getattr(_decomp_pre, "height_cm", 0) or 0,
                                "linear_meters": getattr(_decomp_pre, "linear_meters", 0) or 0,
                            }
                        _cache_key = _smeta_cache_key(req.query, _decomp_pre_dict)
                        _smeta_pre = _smeta_cache_get(_cache_key)
                        if _smeta_pre is None:
                            _q_vec_pre = retriever.embed_query(req.query)
                            _smeta_pre = _smeta_engine_pre.build_smeta(
                                req.query, _q_vec_pre, decomp=_decomp_pre_dict,
                            )
                            _smeta_cache_put(_cache_key, _smeta_pre)
                        else:
                            logger.info("SmetaEngine cache hit",
                                        category=getattr(_smeta_pre, "category_name", ""))
                        # П7.1 #4: under-key blacklist — дешёвая «Логотип»-категория
                        # не должна закрывать запрос на брендбук/под ключ.
                        if _smeta_pre.is_usable and _is_underkey_intent(req.query) \
                                and _smeta_pre.category_name in _UNDERKEY_BLACKLIST_CATEGORIES:
                            logger.info("SmetaEngine result blocked by under-key intent",
                                        category=_smeta_pre.category_name)
                            _smeta_pre = None
                        # П7.1-hotfix: junk categories + min total gate
                        if _smeta_pre is not None and _smeta_pre.is_usable \
                                and _smeta_pre.category_name in _JUNK_CATEGORIES:
                            logger.info("SmetaEngine junk category blocked (std)",
                                        category=_smeta_pre.category_name)
                            _smeta_pre = None
                        # P9: bridge-категории → через LLM с пакетами
                        if _smeta_pre is not None and _smeta_pre.is_usable \
                                and _smeta_pre.category_name in _BRIDGE_CATEGORIES:
                            logger.info("SmetaEngine bridge-category blocked (std)",
                                        category=_smeta_pre.category_name)
                            _smeta_blocked_by = f"bridge:{_smeta_pre.category_name}"
                            _bridge_service = _CATEGORY_TO_BRIDGE_SERVICE.get(
                                _smeta_pre.category_name
                            )
                            if _bridge_service:
                                if _force_inject_bridge(retriever, reranked, _bridge_service):
                                    _bridge_forced = True
                                    # P10.5-IV: подмешиваем реальные КП (G2)
                                    _inject_linked_offers_after_bridge(retriever, reranked)
                                reranked_relevant = _filter_relevant_docs(reranked)
                            _smeta_pre = None
                        if _smeta_pre is not None and _smeta_pre.is_usable \
                                and _is_underkey_intent(req.query) \
                                and (_smeta_pre.total or 0) < _UNDERKEY_MIN_TEMPLATE_TOTAL:
                            logger.info("SmetaEngine under min total for under-key (std)",
                                        total=_smeta_pre.total)
                            _smeta_pre = None
                        if _smeta_pre is not None and _smeta_pre.is_usable:
                            _smeta_pre_usable = True
                            # Stash so the downstream block can reuse without re-embedding.
                            request.state._smeta_precomputed = _smeta_pre
                            logger.info("SmetaEngine hit pre-LLM, skipping generate_deal_estimate",
                                        category=_smeta_pre.category_name,
                                        sim=round(_smeta_pre.match_similarity, 3))
                    except Exception as _e_pre:
                        logger.warning("Pre-LLM SmetaEngine failed", error=str(_e_pre))

                if _smeta_pre_usable:
                    # Stub raw_json — downstream block overrides summary/reasoning/items.
                    raw_json = {"summary": "", "reasoning": "", "flags": [], "risks": [],
                                "deal_items": [], "estimated_price": None, "price_band": None,
                                "confidence": None}
                else:
                    # П7.1 #3: soft-context. Если top-1 шаблон не usable (или заблокирован
                    # under-key blacklist), всё равно отдаём его LLM как «справочный шаблон»,
                    # чтобы модель не галлюцинировала с нуля.
                    try:
                        _soft = locals().get("_smeta_pre")
                        if _soft is not None and getattr(_soft, "category_name", ""):
                            _soft_total = int(round(getattr(_soft, "total", 0) or 0))
                            _soft_lines = [
                                "СПРАВОЧНЫЙ ШАБЛОН (для ориентира, не итоговая цена):",
                                f"- категория: «{_soft.category_name}» "
                                f"({getattr(_soft, 'deals_in_category', 0)} сделок, "
                                f"совпадение {getattr(_soft, 'match_similarity', 0):.0%})",
                                f"- ориентировочный итог по шаблону: {_soft_total:,} ₽".replace(",", " "),
                            ]
                            for _it in (getattr(_soft, "deal_items", []) or [])[:6]:
                                _soft_lines.append(
                                    f"  · {_it.product_name} — {_it.quantity} {_it.unit} × "
                                    f"{int(round(_it.unit_price)):,} ₽".replace(",", " ")
                                )
                            vision_extra += "\n".join(_soft_lines) + "\n\n"
                    except Exception as _e_soft:
                        logger.warning("Soft-context inject failed", error=str(_e_soft))
                    # Inject real product names — force catalog retrieval when
                    # general reranked misses product docs.
                    real_names_set = {
                        doc["payload"].get("product_name")
                        for doc in reranked if doc["payload"].get("product_name")
                    }
                    try:
                        catalog_docs = retriever.retrieve_for_component(req.query, top_k=12)
                        for d in catalog_docs:
                            name = d["payload"].get("product_name")
                            if name:
                                real_names_set.add(name)
                    except Exception as e:
                        logger.warning("Catalog retrieval for deal_items failed", error=str(e))
                    real_names = sorted(n for n in real_names_set if n)
                    if real_names:
                        vision_extra += "НАЗВАНИЯ ТОВАРОВ (используй ТОЛЬКО эти названия в deal_items):\n"
                        vision_extra += "\n".join(f"- {n}" for n in real_names[:50]) + "\n\n"
                    _emit_request_trace(
                        query=req.query,
                        intent=(intent_result.intent if intent_result else "general"),
                        intent_confidence=(intent_result.confidence if intent_result else 0.0),
                        intent_method=(intent_result.method if intent_result else "default"),
                        reranked=reranked,
                        smeta_pre=locals().get("_smeta_pre"),
                        smeta_blocked_by=_smeta_blocked_by,
                        bridge_forced=_bridge_forced,
                        feedback_prefix=feedback_prefix,
                        extra={"path": "std.deal_estimate",
                               "smeta_pre_usable": _smeta_pre_usable},
                    )
                    raw_json = await generator.generate_deal_estimate(
                        req.query, reranked_relevant, pr,
                        extra_context=vision_extra, history=req.history,
                        intent_instruction=intent_instruction,
                    )
            else:
                _emit_request_trace(
                    query=req.query,
                    intent=(intent_result.intent if intent_result else "general"),
                    intent_confidence=(intent_result.confidence if intent_result else 0.0),
                    intent_method=(intent_result.method if intent_result else "default"),
                    reranked=reranked,
                    smeta_pre=None,
                    smeta_blocked_by=_smeta_blocked_by,
                    bridge_forced=_bridge_forced,
                    feedback_prefix=feedback_prefix,
                    extra={"path": "std.structured"},
                )
                raw_json = await generator.generate_structured(
                    req.query, reranked_relevant, pr,
                    extra_context=vision_extra, history=req.history,
                    intent_instruction=intent_instruction,
                )

        # --- Build unified response ---
        top_payload = reranked[0]["payload"] if reranked else {}
        summary = raw_json.get("summary", "") or top_payload.get("searchable_text", req.query)[:120]
        reasoning = raw_json.get("reasoning", "")
        llm_flags = raw_json.get("flags", []) if isinstance(raw_json.get("flags"), list) else []
        llm_risks = raw_json.get("risks", []) if isinstance(raw_json.get("risks"), list) else []

        if decomp.is_complex and estimate is not None:
            # Complex path: compute under-key totals from pricing resolver
            complex_pr = pricing.resolve(reranked, decomp=decomp)

            # Prefer pre-calculated under-key totals when available
            if complex_pr.total_under_key_min is not None:
                total_mid = round((complex_pr.total_under_key_min + complex_pr.total_under_key_max) / 2)
                estimated_price = EstimatedPrice(
                    value=total_mid,
                    currency="RUB",
                    basis=complex_pr.estimated_basis + " (итого под ключ)" if complex_pr.estimated_basis else "рыночный ориентир (итого под ключ)",
                )
                price_band = PriceBand(
                    min=complex_pr.total_under_key_min,
                    max=complex_pr.total_under_key_max,
                    currency="RUB",
                )
            else:
                # Fallback to LLM's synthesized price
                llm_price = raw_json.get("estimated_price")
                llm_band = raw_json.get("price_band")

                if llm_price and isinstance(llm_price, dict) and llm_price.get("value"):
                    estimated_price = EstimatedPrice(
                        value=float(llm_price["value"]),
                        currency="RUB",
                        basis=llm_price.get("basis", "оценка по аналогичным сделкам и рынку"),
                    )
                else:
                    estimated_price = None

                if llm_band and isinstance(llm_band, dict) and llm_band.get("min") and llm_band.get("max"):
                    price_band = PriceBand(
                        min=float(llm_band["min"]),
                        max=float(llm_band["max"]),
                        currency="RUB",
                    )
                else:
                    price_band = PriceBand()

            llm_confidence = raw_json.get("confidence")
            confidence_out = llm_confidence if llm_confidence in ("auto", "guided", "manual") else (complex_pr.confidence or "manual")
            pr_flags = [f"Не найдено в базе: {', '.join(estimate.missing_components)}"] if estimate.missing_components else []
            pr_risks = []
        else:
            # Standard path: prefer under-key totals > LLM price > resolver estimate
            llm_price = raw_json.get("estimated_price")
            llm_band = raw_json.get("price_band")
            llm_confidence = raw_json.get("confidence")

            # Under-key totals available — use as authoritative price band
            if pr.total_under_key_min is not None and not pr.is_financial_modifier:
                total_mid = round((pr.total_under_key_min + pr.total_under_key_max) / 2)
                estimated_price = EstimatedPrice(
                    value=total_mid,
                    currency="RUB",
                    basis=pr.estimated_basis + " (итого под ключ)",
                )
                price_band = PriceBand(
                    min=pr.total_under_key_min,
                    max=pr.total_under_key_max,
                    currency="RUB",
                )
            # Use LLM price only when resolver can't determine a confident value
            elif llm_price and isinstance(llm_price, dict) and llm_price.get("value") and \
               pr.confidence == "manual" and not pr.is_financial_modifier:
                estimated_price = EstimatedPrice(
                    value=float(llm_price["value"]),
                    currency="RUB",
                    basis=llm_price.get("basis", "оценка на основе контекста"),
                )
                # Price band from LLM
                if llm_band and isinstance(llm_band, dict) and \
                   llm_band.get("min") and llm_band.get("max"):
                    price_band = PriceBand(
                        min=float(llm_band["min"]),
                        max=float(llm_band["max"]),
                        currency="RUB",
                    )
                else:
                    price_band = PriceBand(
                        min=pr.price_band_min,
                        max=pr.price_band_max,
                        currency="RUB",
                    )
            elif not pr.is_financial_modifier and pr.estimated_value is not None:
                estimated_price = EstimatedPrice(
                    value=pr.estimated_value,
                    currency="RUB",
                    basis=pr.estimated_basis,
                )
                price_band = PriceBand(
                    min=pr.price_band_min,
                    max=pr.price_band_max,
                    currency="RUB",
                )
            else:
                estimated_price = None
                # Price band: prefer LLM's when resolver has no data
                if llm_band and isinstance(llm_band, dict) and \
                   llm_band.get("min") and llm_band.get("max"):
                    price_band = PriceBand(
                        min=float(llm_band["min"]),
                        max=float(llm_band["max"]),
                        currency="RUB",
                    )
                else:
                    price_band = PriceBand(
                        min=pr.price_band_min,
                        max=pr.price_band_max,
                        currency="RUB",
                    )

            confidence_out = llm_confidence if llm_confidence in ("auto", "guided", "manual") else pr.confidence
            pr_flags = pr.flags
            pr_risks = pr.risks

        all_flags = list(dict.fromkeys(pr_flags + llm_flags))
        all_risks = list(dict.fromkeys(pr_risks + llm_risks))

        deal_items = []
        smeta_result = None
        # П7.1-hotfix: respect pre-LLM decision to skip SmetaEngine entirely.
        # П8.8-A: empty-context smeta queries also block SmetaEngine (no product noun).
        if _is_empty_context_smeta_query(req.query):
            _smeta_blocked_reason = "empty-context smeta query"
        elif _is_describe_intent(req.query) or _looks_like_user_provided_smeta(req.query):
            _smeta_blocked_reason = "describe/provided-smeta intent"
        else:
            _smeta_blocked_reason = None
        if is_estimate and _smeta_blocked_reason:
            logger.info("SmetaEngine downstream override blocked",
                        reason=_smeta_blocked_reason)
        if is_estimate and not _smeta_blocked_reason:
            # PRIMARY (П7): SmetaEngine — deterministic template with price statistics.
            smeta_engine = getattr(request.app.state, "smeta_engine", None)
            # П7.1 #2: reuse precomputed smeta_result from pre-LLM stage if present.
            _precomputed = getattr(request.state, "_smeta_precomputed", None)
            if _precomputed is not None:
                smeta_result = _precomputed
            elif smeta_engine is not None and smeta_engine.is_ready:
                try:
                    q_vec_smeta = retriever.embed_query(req.query)
                    _decomp_for_smeta = None
                    try:
                        _decomp_local = locals().get("decomp")
                        if _decomp_local is not None:
                            _decomp_for_smeta = {
                                "letter_count": getattr(_decomp_local, "letter_count", 0) or 0,
                                "letter_text": getattr(_decomp_local, "letter_text", "") or "",
                                "height_cm": getattr(_decomp_local, "height_cm", 0) or 0,
                                "linear_meters": getattr(_decomp_local, "linear_meters", 0) or 0,
                            }
                    except Exception:
                        _decomp_for_smeta = None
                    smeta_result = smeta_engine.build_smeta(
                        req.query, q_vec_smeta, decomp=_decomp_for_smeta,
                    )
                except Exception as e:
                    logger.warning("SmetaEngine failed", error=str(e))

            # П7.1-hotfix: downstream filter — junk categories + min total under-key.
            if smeta_result is not None and smeta_result.is_usable:
                if smeta_result.category_name in _JUNK_CATEGORIES:
                    logger.info("SmetaEngine junk category blocked (downstream)",
                                category=smeta_result.category_name)
                    smeta_result = None
                elif smeta_result.category_name in _BRIDGE_CATEGORIES:
                    logger.info("SmetaEngine bridge-category blocked (downstream)",
                                category=smeta_result.category_name)
                    _bridge_service_ds = _CATEGORY_TO_BRIDGE_SERVICE.get(
                        smeta_result.category_name
                    )
                    if _bridge_service_ds:
                        if _force_inject_bridge(retriever, reranked, _bridge_service_ds):
                            # P10.5-IV: подмешиваем реальные КП (G2)
                            _inject_linked_offers_after_bridge(retriever, reranked)
                    smeta_result = None
                elif _is_underkey_intent(req.query) \
                        and smeta_result.category_name in _UNDERKEY_BLACKLIST_CATEGORIES:
                    logger.info("SmetaEngine under-key blacklist (downstream)")
                    smeta_result = None
                elif _is_underkey_intent(req.query) \
                        and (smeta_result.total or 0) < _UNDERKEY_MIN_TEMPLATE_TOTAL:
                    logger.info("SmetaEngine under min total (downstream)",
                                total=smeta_result.total)
                    smeta_result = None

            if smeta_result is not None:
                try:
                    if smeta_result.is_usable:
                        deal_items = smeta_result.deal_items
                        logger.info("Deal estimate from SmetaEngine",
                                    category=smeta_result.category_name,
                                    quality=smeta_result.match_quality,
                                    sim=round(smeta_result.match_similarity, 3),
                                    items=len(deal_items),
                                    total=smeta_result.total)
                        # Authoritative price override from template statistics
                        estimated_price = EstimatedPrice(
                            value=smeta_result.total,
                            currency="RUB",
                            basis=f"шаблон категории «{smeta_result.category_name}» ({smeta_result.deals_in_category} сделок)",
                        )
                        price_band = PriceBand(
                            min=smeta_result.price_band_min,
                            max=smeta_result.price_band_max,
                            currency="RUB",
                        )
                        # Map smeta confidence to StructuredResponse schema
                        _smeta_conf_map = {"high": "auto", "medium": "guided", "low": "manual"}
                        confidence_out = _smeta_conf_map.get(smeta_result.confidence, "manual")
                        all_flags = [smeta_result.match_reason] + smeta_result.flags + all_flags

                        # Fix E: rewrite summary from authoritative smeta data so
                        # the textual answer cannot disagree with estimated_price.
                        try:
                            _fmt_total = f"{int(round(smeta_result.total)):,}".replace(",", " ")
                            _fmt_min = f"{int(round(smeta_result.price_band_min)):,}".replace(",", " ")
                            _fmt_max = f"{int(round(smeta_result.price_band_max)):,}".replace(",", " ")
                            _conf_ru = {"high": "высокая", "medium": "средняя", "low": "низкая"}.get(
                                smeta_result.confidence, "средняя"
                            )
                            _lines = [
                                f"Оценка по шаблону категории «{smeta_result.category_name}» "
                                f"(база: {smeta_result.deals_in_category} аналогичных сделок).",
                                f"Итого: {_fmt_total} ₽ (диапазон {_fmt_min}–{_fmt_max} ₽), "
                                f"уверенность {_conf_ru}.",
                            ]
                            if deal_items:
                                _lines.append(f"Состав сметы: {len(deal_items)} позиций.")
                            if smeta_result.flags:
                                _lines.append("⚠ " + "; ".join(smeta_result.flags[:2]))
                            summary = " ".join(_lines)
                            reasoning = (
                                f"Шаблон выбран по семантической близости запроса к категории "
                                f"«{smeta_result.category_name}» (cosine {smeta_result.match_similarity:.2f}, "
                                f"качество совпадения: {smeta_result.match_quality}). "
                                f"Цены — статистическое среднее (mean+median+weighted+trimmed) "
                                f"по {smeta_result.deals_in_category} сделкам категории."
                            )
                        except Exception as _e_sum:
                            logger.warning("Smeta summary rewrite failed", error=str(_e_sum))
                    else:
                        logger.info("SmetaEngine no match",
                                    reason=smeta_result.match_reason)
                except Exception as e:
                    logger.warning("Smeta override failed", error=str(e))

            # FALLBACK 1: existing DealLookup (legacy real-deal matching)
            if not deal_items:
                deal_lookup: DealLookup | None = getattr(request.app.state, "deal_lookup", None)
                matched_title = ""
                if deal_lookup and deal_lookup._loaded:
                    deal_items, matched_title = deal_lookup.find_best_deal_items(reranked)
                    if deal_items:
                        logger.info("Deal estimate from DealLookup fallback",
                                    items=len(deal_items), source_deal=matched_title[:60])
                        if matched_title:
                            all_flags = [f"Состав по аналогу: «{matched_title[:80]}»"] + all_flags

            # FALLBACK 2: LLM-generated items if no template and no real deal matched
            if not deal_items:
                deal_items = _parse_deal_items(raw_json)
                if deal_items:
                    all_flags = ["Состав сформирован моделью — нет точного аналога в базе"] + all_flags
                logger.info("Deal estimate from LLM fallback", item_count=len(deal_items))

            # --- PRICE RECONCILIATION (SINGLE SOURCE OF TRUTH) ---
            # When deal_items exist, their sum becomes authoritative: estimated_price,
            # price_band and summary must agree. This prevents the 4-number disagreement
            # (summary / deal_items / estimated_price / parametric_breakdown).
            if deal_items:
                items_total = sum((di.total or 0) for di in deal_items)
                if items_total > 0:
                    current_val = estimated_price.value if estimated_price else 0
                    drift = abs(items_total - current_val) / max(items_total, 1)
                    if current_val == 0 or drift > 0.05:
                        logger.info("Price reconciliation applied",
                                    old=current_val, new=items_total,
                                    items=len(deal_items), drift=round(drift, 2))
                        estimated_price = EstimatedPrice(
                            value=round(items_total),
                            currency="RUB",
                            basis="сумма позиций сметы (deal_items)",
                        )
                        price_band = PriceBand(
                            min=round(items_total * 0.85),
                            max=round(items_total * 1.15),
                            currency="RUB",
                        )
                        # Parametric breakdown was computed independently — stale now
                        if parametric_breakdown_schema is not None:
                            all_flags = ["Параметрический расчёт заменён сметой"] + all_flags
                            parametric_breakdown_schema = None

        # --- Intent-driven post-LLM handling (replaces hardcoded gate summaries) ---
        # When intent classifier is active, no-price intents nullify pricing
        # but KEEP the LLM's summary (which was guided by intent_instruction).
        # Old gates below still run as safety nets for edge cases.
        _intent_handled = False
        if _use_intent and intent_result is not None and intent_result.confidence >= 0.75:
            _ir = intent_result
            if _ir.is_no_price:
                logger.info("Intent-driven no-price override",
                            intent=_ir.intent, method=_ir.method,
                            query=req.query[:80])
                estimated_price = None
                price_band = PriceBand()
                deal_items = []
                parametric_breakdown_schema = None
                confidence_out = "manual"
                _intent_handled = True
                # Add a flag but do NOT replace summary — LLM already got the instruction
                _intent_flag_map = {
                    "consultation": "Ознакомительный запрос — презентация, не расчёт цены",
                    "describe": "Переговорный сценарий — ручной расчёт или уточнение у клиента",
                    "out_of_scope": "Запрос вне рабочей тематики (цены не применимы)",
                    "financial_modifier": "Финансовый модификатор — не является самостоятельным товаром (ручной расчёт)",
                    "visualization": "Визуализация — только после замеров и аванса за дизайн",
                    "referential": "Ссылка на «эту/прошлую» услугу — прикрепите фото или укажите параметры",
                    "empty_context_smeta": "Не указан товар/направление — уточните, какую услугу оценить",
                    "underspec": "Запрос слишком общий — требуется уточнение параметров",
                }
                _iflag = _intent_flag_map.get(_ir.intent, "")
                if _iflag and _iflag not in all_flags:
                    all_flags = [_iflag] + all_flags

        # П8.8-F: manager-script / describe-intent queries must never emit
        # an estimated price. Модель может проскочить в LLM fallback и выдать
        # число (см. fb#47 → 200 000 ₽). Жёстко обнуляем ПОСЛЕ основного флоу.
        if not _intent_handled and _is_describe_intent(req.query):
            logger.info("Manager-script intent — forcing no-price response",
                        query=req.query[:80])
            estimated_price = None
            price_band = PriceBand()
            deal_items = []
            parametric_breakdown_schema = None
            confidence_out = "manual"
            _script_flag = "Переговорный сценарий — ручной расчёт или уточнение у клиента"
            if _script_flag not in all_flags:
                all_flags = [_script_flag] + all_flags

        # П8.8-J: referential queries without chat history — сlient refers
        # to "эту услугу / вместо этой вывески" but no previous message exists.
        # Force manual + clarification instructing to attach photo or specs.
        if not _intent_handled and _is_referential_query(req.query) and not _has_history:
            logger.info("Referential query without history — forcing clarification",
                        query=req.query[:80])
            estimated_price = None
            price_band = PriceBand()
            deal_items = []
            parametric_breakdown_schema = None
            confidence_out = "manual"
            _ref_flag = "Ссылка на «эту/прошлую» услугу — прикрепите фото или укажите параметры"
            if _ref_flag not in all_flags:
                all_flags = [_ref_flag] + all_flags

        # П8.8-A: empty-context smeta queries — force clarification response
        # (модель не должна угадывать товар по умолчанию).
        if not _intent_handled and _is_empty_context_smeta_query(req.query):
            logger.info("Empty-context smeta query — forcing clarification",
                        query=req.query[:80])
            estimated_price = None
            price_band = PriceBand()
            deal_items = []
            parametric_breakdown_schema = None
            confidence_out = "manual"
            # P8.8-hotfix: reset summary/reasoning so LLM/smeta hallucinations
            # (напр. «БББ-2115, направление РИК, 115003 руб» или «Визуальный
            # анализ фотографий…») не попадают в ответ клиенту.
            summary = (
                "Не указан товар или направление. Пожалуйста, уточните, "
                "какую конкретно услугу нужно оценить — логотип, вывеска "
                "(объёмные буквы, световой короб, штендер), листовки, "
                "баннер, монтаж и т.п. — и пришлите ключевые параметры: "
                "размеры, тираж, материалы, место размещения."
            )
            reasoning = (
                "Запрос не содержит продуктового контекста — оценка без "
                "конкретики неизбежно будет неверной, поэтому возвращаем "
                "запрос на уточнение вместо предположения категории."
            )
            _clarify_flag = "Не указан товар/направление — уточните, какую услугу оценить"
            if _clarify_flag not in all_flags:
                all_flags = [_clarify_flag] + all_flags

        # П8.8-B: smeta explosion guard. Если шаблон имеет >25 позиций или
        # price_band раздут в >20 раз — это брендбук/под-ключ с категорией,
        # где сумма доминирует выбросами. См. fb#12 (Брендбук 2.57M ₽, 39 позиций,
        # band 0–6.4M). Обнуляем цену, форсим manual, флажок.
        _smeta_res = locals().get("smeta_result")
        if _smeta_res is not None and getattr(_smeta_res, "is_usable", False):
            _n_items = len(getattr(_smeta_res, "deal_items", []) or [])
            _bmin = getattr(_smeta_res, "price_band_min", 0) or 0
            _bmax = getattr(_smeta_res, "price_band_max", 0) or 0
            _total = getattr(_smeta_res, "total", 0) or 0
            _band_ratio = (_bmax / _bmin) if _bmin > 0 else float("inf") if _bmax > 0 else 0
            if _n_items > 25 or _band_ratio > 20 or (_bmin == 0 and _total > 500_000):
                logger.warning("SmetaEngine explosion blocked",
                               category=getattr(_smeta_res, "category_name", ""),
                               items=_n_items, total=_total,
                               band_min=_bmin, band_max=_bmax,
                               band_ratio=round(_band_ratio, 1) if _band_ratio != float("inf") else "inf")
                estimated_price = None
                price_band = PriceBand()
                deal_items = []
                parametric_breakdown_schema = None
                confidence_out = "manual"
                # P8.8-hotfix: reset summary — смета уже переписала его
                # в «Оценка по шаблону… 2 572 901 ₽». Заменяем на честный
                # текст про слишком большой разброс категории.
                _cat_name = getattr(_smeta_res, "category_name", "")
                summary = (
                    f"Категория «{_cat_name}» объединяет {_n_items} разнородных "
                    "позиций с очень широким разбросом цен, поэтому "
                    "автоматическая оценка будет некорректной. "
                    "Уточните, что именно нужно: базовый вариант, полный "
                    "комплект, сроки и материалы — менеджер соберёт точный "
                    "состав и пришлёт смету."
                )
                reasoning = (
                    f"В шаблоне {_n_items} позиций, диапазон цен "
                    f"{_bmin:.0f}–{_bmax:.0f} ₽ (разброс x{_band_ratio:.0f}). "
                    "Суммирование даёт бессмысленный итог — требуется ручной "
                    "подбор позиций менеджером."
                )
                _expl_flag = (
                    f"Слишком большой разброс в шаблоне ({_n_items} позиций) "
                    f"— требуется ручной расчёт менеджером"
                )
                if _expl_flag not in all_flags:
                    all_flags = [_expl_flag] + all_flags

        # П8.7-D1: out-of-scope queries (рабочие часы, адрес, контакты).
        # Force manual confidence + nullify price + inject clear scope flag.
        if _is_out_of_scope_query(req.query):
            logger.info("Out-of-scope query — forcing no-price response",
                        query=req.query[:80])
            estimated_price = None
            price_band = PriceBand()
            deal_items = []
            parametric_breakdown_schema = None
            confidence_out = "manual"
            _oos_flag = "Запрос вне рабочей тематики (цены не применимы)"
            if _oos_flag not in all_flags:
                all_flags = [_oos_flag] + all_flags

        # П8.7-D2/D3: financial modifier queries (безнал, скидка, НДС).
        # Не являются самостоятельным товаром — обнуляем цену, force manual,
        # инжектим флаг со словом "модификатор".
        if _is_financial_modifier_query(req.query):
            logger.info("Financial modifier query — forcing no-price response",
                        query=req.query[:80])
            estimated_price = None
            price_band = PriceBand()
            deal_items = []
            parametric_breakdown_schema = None
            confidence_out = "manual"
            _fm_flag = "Финансовый модификатор — не является самостоятельным товаром (ручной расчёт)"
            if _fm_flag not in all_flags:
                all_flags = [_fm_flag] + all_flags

        # П8.6-B: complexity markers force confidence=manual regardless of
        # LLM/SmetaEngine verdict. Also injects a flag containing the words
        # "ручной" and "расчёт" so downstream flag-check validation passes.
        if _is_manual_complexity_query(req.query):
            if confidence_out != "manual":
                logger.info("Manual confidence forced by complexity marker",
                            original=confidence_out, query=req.query[:80])
                confidence_out = "manual"
            _manual_flag = "Требуется ручной расчёт менеджером (нестандартные условия)"
            if _manual_flag not in all_flags:
                all_flags = [_manual_flag] + all_flags

        # П8.9: spec-heavy signage queries — height + material spec → manual.
        # См. fb#14 (41см алюминий/бортогиб) и fb#55 (45см композит/контражур):
        # template-оценка по «Световые вывески» даёт ~38k, но реальная цена
        # определяется спецификацией. Обнуляем price/band, force manual.
        if not _intent_handled and _is_spec_heavy_signage_query(req.query):
            if confidence_out != "manual":
                logger.info("Spec-heavy signage — forcing manual",
                            original=confidence_out, query=req.query[:80])
            confidence_out = "manual"
            estimated_price = None
            price_band = PriceBand()
            deal_items = []
            _spec_flag = "Требуется ручной расчёт по спецификации (высота + материал)"
            if _spec_flag not in all_flags:
                all_flags = [_spec_flag] + all_flags
            summary = (
                "Цена объёмной вывески с подсветкой определяется именно спецификацией: "
                "высотой букв, материалом (композит / акрил / алюминий / бортогиб), "
                "типом подсветки (лицевая / контражур) и площадью подложки. "
                "Template-оценка по категории здесь системно промахивается, поэтому "
                "менеджер соберёт смету постатейно из прайс-листа — дизайн, "
                "производство буквы, блок питания, светодиоды, подложка, монтаж."
            )
            reasoning = (
                "П8.9 spec-heavy signage gate: в запросе есть спецификация "
                "(высота + материал), которая перекрывает template-медиану категории. "
                "Estimated_price обнулён; расчёт — вручную по позициям каталога."
            )

        # П9.0-J: underspecified category — монтаж без типа, листовки без способа.
        # Overrides SmetaEngine template price with clarification question.
        _underspec = _underspec_clarification(req.query)
        if not _intent_handled and _underspec is not None:
            _us_summary, _us_flag = _underspec
            logger.info("Underspecified category — forcing clarification",
                        query=req.query[:80])
            confidence_out = "manual"
            estimated_price = None
            price_band = PriceBand()
            deal_items = []
            summary = _us_summary
            reasoning = "П9.0-J underspecified category gate: запрос слишком общий для автоматической оценки."
            if _us_flag not in all_flags:
                all_flags = [_us_flag] + all_flags

        # П9.1-A: consultation queries — «Вы делаете логотипы?» = не ценовой
        # запрос. Презентуем компанию, не даём цену. См. fb#72.
        if not _intent_handled and _is_consultation_query(req.query):
            logger.info("Consultation query — presentation mode, no price",
                        query=req.query[:80])
            confidence_out = "manual"
            estimated_price = None
            price_band = PriceBand()
            deal_items = []
            summary = (
                "Да, мы это делаем! Лабус — рекламно-производственная компания "
                "в Махачкале с собственным цехом и 15+ лет опыта. Полный цикл: "
                "дизайн, производство, монтаж, гарантия 12 месяцев. "
                "Чтобы прикинуть бюджет — уточните, что именно нужно: "
                "тип изделия, размеры, материал, нужен ли монтаж."
            )
            reasoning = "П9.1-A consultation gate: ознакомительный вопрос, не запрос цены."
            _consult_flag = "Ознакомительный запрос — презентация, не расчёт цены"
            if _consult_flag not in all_flags:
                all_flags = [_consult_flag] + all_flags

        # П9.1-B: height-based signage pricing — детерминистический расчёт по
        # рыночным ориентирам. Заменяет template-median когда указана высота букв.
        # fb#66 (40см), fb#68 (80см), fb#70 (60см) — все давали 38384₽.
        if not _intent_handled and not _is_spec_heavy_signage_query(req.query):
            _hbp = _compute_height_based_price(req.query)
            if _hbp is not None:
                logger.info("Height-based signage pricing override",
                            height=_hbp["height"], n_letters=_hbp["n_letters"],
                            total_mid=_hbp["total_mid"], query=req.query[:80])
                estimated_price = _hbp["total_mid"]
                price_band = PriceBand(
                    min=_hbp["total_min"],
                    max=_hbp["total_max"],
                    currency="RUB",
                )
                confidence_out = "guided"
                deal_items = []
                _h = _hbp["height"]
                _nl = _hbp["n_letters"]
                _nl_str = f" ({_nl} букв)" if _nl > 0 else ""
                summary = (
                    f"Объёмные буквы высотой {_h} см{_nl_str} под ключ: "
                    f"ориентировочно {_hbp['total_min']:,.0f}–{_hbp['total_max']:,.0f} руб. "
                    f"Буквы: {_hbp['letters_label']}. "
                    f"Монтаж: {_hbp['montage'][0]:,.0f}–{_hbp['montage'][1]:,.0f} руб. "
                    f"Каркас: {_hbp['karkas'][0]:,.0f}–{_hbp['karkas'][1]:,.0f} руб. "
                    f"Дизайн: {_DESIGN_RANGE[0]:,.0f}–{_DESIGN_RANGE[1]:,.0f} руб. "
                    "Точная цена зависит от материала, типа подсветки и сложности монтажа — "
                    "пришлите фото фасада для расчёта."
                ).replace(",", " ")
                reasoning = (
                    f"П9.1-B height-based pricing: высота {_h} см, "
                    f"{_nl} букв. Расчёт по рыночным ориентирам (цена/букву × кол-во + "
                    "монтаж + каркас + дизайн)."
                )
                _hbp_flag = f"Расчёт по рыночным ориентирам (высота {_h} см)"
                if _hbp_flag not in all_flags:
                    all_flags = [_hbp_flag] + all_flags

        # П8.7-C: для bundle-intent запросов делаем таргетированный фетч
        # doc_type=bundle и используем его как основу для suggested_bundle.
        # Стандартный reranked остаётся приоритетным (если bundle-док уже в топе),
        # а dedicated fetch служит fallback'ом когда bundle проигрывает
        # product/deal_profile докам по semantic score.
        bundle_pool = reranked
        if _is_bundle_intent_query(req.query):
            try:
                dedicated_bundles = retriever.retrieve_bundles(req.query, top_k=5)
                if dedicated_bundles:
                    bundle_pool = list(reranked) + dedicated_bundles
                    logger.info("Bundle intent — dedicated bundle fetch",
                                query=req.query[:80], hits=len(dedicated_bundles))
            except Exception as e:
                logger.warning("Bundle dedicated fetch failed", error=str(e))

        # П8.8-G: forbidden-promise post-filter (after all pricing decisions).
        # Strips «бесплат*» mentions and injects canonical visualization phrasing
        # when клиент просит показать как будет на фасаде. См. fb#19.
        try:
            summary, reasoning, all_flags = _apply_forbidden_promise_filter(
                summary, reasoning, all_flags, req.query,
            )
            # If visualization request — also force manual (no cheap 3D offer).
            if _is_visualization_request(req.query) and confidence_out != "manual":
                logger.info("Visualization request — forcing manual confidence",
                            query=req.query[:80])
                confidence_out = "manual"
                estimated_price = None
                price_band = PriceBand()
                deal_items = []
        except Exception as _e_fp:
            logger.warning("Forbidden-promise filter failed", error=str(_e_fp))

        response = StructuredResponse(
            summary=summary,
            suggested_bundle=_build_suggested_bundle(bundle_pool),
            estimated_price=estimated_price,
            price_band=price_band,
            confidence=confidence_out,
            reasoning=reasoning,
            flags=all_flags,
            risks=all_risks,
            references=_build_references(reranked),
            segmented_references=_build_segmented_references(
                reranked,
                photo_index=getattr(request.app.state, "photo_index", None),
            ),
            source_distinction=_detect_source_distinction(reranked),
            parametric_breakdown=parametric_breakdown_schema,
            deal_items=deal_items,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

        logger.info("Structured query completed",
                    latency_ms=response.latency_ms,
                    confidence=response.confidence,
                    sources=len(reranked))

        # Persist messages to chat if authenticated
        if req.chat_id and user:
            try:
                save_message(req.chat_id, user["id"], "user", req.query, mode="structured")
                structured_data = response.model_dump(exclude={"references"})
                save_message(req.chat_id, user["id"], "assistant", response.summary,
                             mode="structured", structured_data=structured_data,
                             latency_ms=response.latency_ms)
            except Exception as e:
                logger.warning("Failed to save structured chat message", error=str(e))

        return response

    except Exception as e:
        logger.error("Structured query failed", error=str(e), query=req.query[:100])
        raise HTTPException(500, f"Structured query failed: {str(e)}")
