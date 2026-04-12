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
from app.auth import get_optional_user
from app.routers.chats import save_message, get_chat_history
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Query"])


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

        # Inject feedback-based learning context (RLHF)
        feedback_store = getattr(request.app.state, "feedback_store", None)
        if feedback_store and retriever.is_ready:
            try:
                q_vec = retriever.embed_query(req.query)
                direction = getattr(decomp, "direction", "") or ""
                fb_ctx = build_feedback_context(feedback_store, q_vec, direction)
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

        # --- Decompose query: detect complex multi-component requests ---
        decomp = decompose(req.query)
        parametric_breakdown_schema: ParametricBreakdown | None = None
        estimate = None
        is_estimate = _is_deal_estimate_query(req.query)

        # Feedback-based learning context (RLHF)
        feedback_prefix = ""
        feedback_store = getattr(request.app.state, "feedback_store", None)
        if feedback_store and retriever.is_ready:
            try:
                q_vec = retriever.embed_query(req.query)
                direction = getattr(decomp, "direction", "") or ""
                feedback_prefix = build_feedback_context(feedback_store, q_vec, direction)
                if feedback_prefix:
                    logger.info("Feedback context injected (structured)", lessons_matched=feedback_prefix.count("["))
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
                    )
            else:
                raw_json = await generator.generate_structured(
                    req.query, reranked, pr_obj,
                    extra_context=full_extra, history=req.history,
                )

        else:
            # --- Standard pipeline for simple queries ---
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
                    raw_json = await generator.generate_deal_estimate(
                        req.query, reranked_relevant, pr,
                        extra_context=vision_extra, history=req.history,
                    )
            else:
                raw_json = await generator.generate_structured(
                    req.query, reranked_relevant, pr,
                    extra_context=vision_extra, history=req.history,
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

        # П8.8-F: manager-script / describe-intent queries must never emit
        # an estimated price. Модель может проскочить в LLM fallback и выдать
        # число (см. fb#47 → 200 000 ₽). Жёстко обнуляем ПОСЛЕ основного флоу.
        if _is_describe_intent(req.query):
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
        _has_history = bool(getattr(req, "history", None) or getattr(req, "chat_id", None))
        if _is_referential_query(req.query) and not _has_history:
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
        if _is_empty_context_smeta_query(req.query):
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
        if _is_spec_heavy_signage_query(req.query):
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
        if _underspec is not None:
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
