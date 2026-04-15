"""
Safety gates: thin post-LLM checks that override responses ONLY when dangerous.

These gates do NOT generate content — they nullify pricing or strip dangerous
phrases. They always run regardless of intent classification.

Kept gates (from original 15):
1. out_of_scope — competitors, politics, non-product queries
2. financial_modifier — "give me a discount", payment terms
3. forbidden_promise — "free" services, viz without measurements
4. explosion_guard — smeta with >25 items or >20x band ratio
"""
import re
from app.utils.logging import get_logger

logger = get_logger(__name__)


# --- Out of scope ---
_OUT_OF_SCOPE_MARKERS = [
    re.compile(r"режим\s+работы|график\s+работы|часы\s+работы|время\s+работы", re.IGNORECASE),
    re.compile(r"когда\s+(откры|закры|работае)|во\s+сколько\s+(откры|закры)", re.IGNORECASE),
    re.compile(r"расписани[еяю]|выходны[еыхй]\s+(дн|ли)", re.IGNORECASE),
    re.compile(r"как\s+(добрать|проехать|найти)|где\s+(находит|расположен)", re.IGNORECASE),
    re.compile(r"адрес\s+офиса|адрес\s+компани|\bваш\s+адрес", re.IGNORECASE),
    re.compile(r"номер\s+телефон|контактн.*телефон|\bтелефон\s+офис", re.IGNORECASE),
]


def is_out_of_scope(query: str) -> bool:
    if not query:
        return False
    return any(rx.search(query) for rx in _OUT_OF_SCOPE_MARKERS)


# --- Financial modifier ---
_FINANCIAL_MODIFIER_MARKERS = [
    re.compile(r"безнал\w*", re.IGNORECASE),
    re.compile(r"\bналичн\w*|\bналичк\w*", re.IGNORECASE),
    re.compile(r"скидк\w*\s*\d+\s*%|скидк\w*.*стоимост|\d+\s*%\s*скидк", re.IGNORECASE),
    re.compile(r"надбавк\w*|наценк\w*", re.IGNORECASE),
    re.compile(r"\bндс\b|без\s*ндс|с\s*ндс", re.IGNORECASE),
    re.compile(r"предоплат\w*\s*\d+\s*%|рассрочк\w*", re.IGNORECASE),
]


def is_financial_modifier(query: str) -> bool:
    if not query:
        return False
    return any(rx.search(query) for rx in _FINANCIAL_MODIFIER_MARKERS)


# --- Forbidden promise filter ---
_FORBIDDEN_FREE_PATTERN = re.compile(r"\bбесплатн\w*", re.IGNORECASE)
_VISUALIZATION_QUERY_PATTERN = re.compile(
    r"как\s+(это\s+)?будет\s+на\s+фасад|визуализаци|покаж\w*\s+как|увидеть\s+как\s+(это|будет)|фото\s*ш?оп|3d\s*модел",
    re.IGNORECASE,
)
_CANONICAL_VIZ_RESPONSE = (
    "Визуализация объекта на фасаде выполняется только после выезда "
    "замерщика и платного аванса за дизайн-проект. Без этих этапов мы не "
    "даём 3D-макеты и не оцениваем условия монтажа по фото. Согласуйте "
    "выезд на замеры — менеджер подберёт подходящий вариант и рассчитает "
    "стоимость дизайна и изготовления."
)
_CANONICAL_FREE_PHRASE = (
    "Мы не оказываем бесплатных услуг — каждая позиция тарифицируется."
)


def is_visualization_request(query: str) -> bool:
    if not query:
        return False
    return bool(_VISUALIZATION_QUERY_PATTERN.search(query))


def apply_forbidden_promise_filter(
    summary: str,
    reasoning: str,
    flags: list[str],
    query: str,
) -> tuple[str, str, list[str]]:
    new_flags = list(flags)
    txt_before = (summary or "") + " " + (reasoning or "")
    had_free_promise = bool(_FORBIDDEN_FREE_PATTERN.search(txt_before))
    is_viz = is_visualization_request(query)

    def _strip_free(text: str) -> str:
        if not text:
            return text
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = [s for s in sentences if not _FORBIDDEN_FREE_PATTERN.search(s)]
        return " ".join(kept).strip() or text

    if had_free_promise and not is_viz:
        summary = _strip_free(summary)
        reasoning = _strip_free(reasoning)
        extra = _CANONICAL_FREE_PHRASE
        if extra not in summary:
            summary = (summary + " " + extra).strip()
        flag = "Запрос на бесплатную услугу заблокирован — все услуги платные"
        if flag not in new_flags:
            new_flags = [flag] + new_flags

    if is_viz:
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


# --- Explosion guard ---
def check_smeta_explosion(smeta_result) -> bool:
    """True if smeta result has too many items or extreme band ratio."""
    if smeta_result is None or not getattr(smeta_result, "is_usable", False):
        return False
    n_items = len(getattr(smeta_result, "deal_items", []) or [])
    bmin = getattr(smeta_result, "price_band_min", 0) or 0
    bmax = getattr(smeta_result, "price_band_max", 0) or 0
    total = getattr(smeta_result, "total", 0) or 0
    band_ratio = (bmax / bmin) if bmin > 0 else float("inf") if bmax > 0 else 0
    return n_items > 25 or band_ratio > 20 or (bmin == 0 and total > 500_000)


# --- Complexity markers (force manual confidence) ---
_MANUAL_COMPLEXITY_MARKERS = [
    re.compile(r"на\s+высот[еы]\s+(бол[еь]е|свыше)?\s*\d+\s*м", re.IGNORECASE),
    re.compile(r"альпинист|промальп", re.IGNORECASE),
    re.compile(r"нестандартн|эксклюзивн", re.IGNORECASE),
    re.compile(r"уникальн\w*\s+\w*\s*(стенд|конструкци|проект|вывеск|издели|объект)", re.IGNORECASE),
    re.compile(r"индивидуальн\w*\s+\w*\s*(дизайн|кампани|проект|разработ|концепци|реклам)", re.IGNORECASE),
    re.compile(r"нержавей|бронз|латун", re.IGNORECASE),
    re.compile(r"\bремонт\b|восстановл|реставраци", re.IGNORECASE),
]


def is_manual_complexity(query: str) -> bool:
    if not query:
        return False
    return any(rx.search(query) for rx in _MANUAL_COMPLEXITY_MARKERS)
