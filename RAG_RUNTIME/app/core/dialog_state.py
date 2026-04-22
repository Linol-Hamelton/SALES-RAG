"""Dialog-state extractor.

Analyses chat history to extract signals that the single-turn pipeline
(SmetaEngine, intent classifier, generator) cannot see on its own:

  - `is_first_touch`   — is this the very first assistant reply?
  - `rejected_products` — services the customer explicitly refused ("не нужен логотип")
  - `confirmed_product` — the product the customer is actively asking about
                          (locked from the earliest user turn that named a concrete product)
  - `already_priced`    — prices already quoted in prior assistant turns;
                          cheap guard against repeating numbers
  - `size_params`       — тираж/высота/количество already mentioned (any turn)
  - `has_explicit_price_ask` — the LAST user turn actually asks for a price

Used by `app/routers/query.py` to gate the SmetaEngine summary override and
to mark the SmetaEngine category as "off-limits" when the customer rejected
that service.

Design principle: **cheap, pure-Python, regex-based**. No embeddings, no
extra LLM calls. Must be deterministic and testable. Keep it strict — false
positives here would silently drop the price answer when the customer does
want it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# Product vocabulary
# ---------------------------------------------------------------------------
# Canonical product keys → regex stems matching common morphology.
# Keys are stable identifiers used by callers; SmetaEngine category names
# are mapped to these via `smeta_category_to_product()` below.
PRODUCT_PATTERNS: dict[str, re.Pattern] = {
    "logo":       re.compile(r"\b(логотип\w*|лого|айдентик\w*)\b", re.IGNORECASE),
    "brandbook":  re.compile(r"\b(брендбук\w*|фирменн\w+\s+стил\w*)", re.IGNORECASE),
    "signboard":  re.compile(r"\b(вывеск\w*|буквы?\s+объ[её]мн\w*|объ[её]мн\w+\s+букв\w*|световой?\s+короб\w*)", re.IGNORECASE),
    "business_card": re.compile(r"\bвизитк\w*", re.IGNORECASE),
    "flyer":      re.compile(r"\bлистовк\w*", re.IGNORECASE),
    "banner":     re.compile(r"\bбаннер\w*", re.IGNORECASE),
    "booklet":    re.compile(r"\bбуклет\w*", re.IGNORECASE),
    "sticker":    re.compile(r"\b(наклейк\w*|стикер\w*)", re.IGNORECASE),
    "menu":       re.compile(r"\bменю\b", re.IGNORECASE),
    "sign_table": re.compile(r"\bтабличк\w*", re.IGNORECASE),
    "stand":      re.compile(r"\b(штендер\w*|стенд\w*)", re.IGNORECASE),
    "bracket":    re.compile(r"\b(панель-?кронштейн\w*|кронштейн\w*)", re.IGNORECASE),
    "packaging":  re.compile(r"\bупаковк\w*", re.IGNORECASE),
    "neon":       re.compile(r"\bнеонов\w+\s+вывеск\w*", re.IGNORECASE),
    "design":     re.compile(r"\b(дизайн)\b", re.IGNORECASE),
}

# Map from SmetaEngine/bridge category names to our product keys.
# Keep loose (substring) so rare category renames don't break us.
_CATEGORY_TO_PRODUCT: list[tuple[re.Pattern, str]] = [
    (re.compile(r"логотип",           re.IGNORECASE), "logo"),
    (re.compile(r"брендбук",          re.IGNORECASE), "brandbook"),
    (re.compile(r"фирменн.+стил",     re.IGNORECASE), "brandbook"),
    (re.compile(r"объ[её]мн.+букв|световые?\s+вывеск", re.IGNORECASE), "signboard"),
    (re.compile(r"вывеск",            re.IGNORECASE), "signboard"),
    (re.compile(r"визитк",            re.IGNORECASE), "business_card"),
    (re.compile(r"листовк",           re.IGNORECASE), "flyer"),
    (re.compile(r"баннер",            re.IGNORECASE), "banner"),
    (re.compile(r"буклет",            re.IGNORECASE), "booklet"),
    (re.compile(r"наклейк|стикер",    re.IGNORECASE), "sticker"),
    (re.compile(r"меню",              re.IGNORECASE), "menu"),
    (re.compile(r"табличк",           re.IGNORECASE), "sign_table"),
    (re.compile(r"штендер|стенд",     re.IGNORECASE), "stand"),
    (re.compile(r"кронштейн",         re.IGNORECASE), "bracket"),
    (re.compile(r"упаковк",           re.IGNORECASE), "packaging"),
    (re.compile(r"неон",              re.IGNORECASE), "neon"),
]


def smeta_category_to_product(category_name: str) -> str | None:
    """Map a SmetaEngine category name to our canonical product key.

    Returns None if no mapping is found (callers should then treat the result
    as non-maskable — i.e. apply no rejection/context rules).
    """
    if not category_name:
        return None
    for rx, key in _CATEGORY_TO_PRODUCT:
        if rx.search(category_name):
            return key
    return None


# ---------------------------------------------------------------------------
# Negation / rejection detectors
# ---------------------------------------------------------------------------
# Matches "не нужен логотип", "не надо вывеску", "без логотипа", "логотип
# не нужен", "не хочу делать логотип". We require the negation and the
# product token within a small window to avoid false positives like
# "логотип не такой как хотелось".
#
# Note on the stem `нуж\w*`: Russian masculine short-adjective «нужен» has a
# fleeting «е» (нуж-ен vs fem. нужн-а, neut. нужн-о, pl. нужн-ы). The stem
# `нуж` covers every form including «нужен»; `нужн\w*` alone silently misses
# the most common form («логотип не нужен»). Same reasoning for `нуждат\w*`.
_NEG_PREFIX = r"(?:не\s+(?:нуж\w*|надо|хоч\w+|планир\w+|требует\w+|треб\w+)|без|нет)"

def _neg_rx(product_rx: str) -> re.Pattern:
    """Build a 'X is rejected' regex — two windows (negation→product, product→negation).

    The window is wide enough (30 chars) to span patterns like
    «нет логотипа/брендбука и он мне не нужен», where the product token and
    the explicit rejection phrase are separated by a parenthetical or a
    conjunction clause but still belong to the same sentence.
    """
    ahead  = rf"{_NEG_PREFIX}\b[^.!?\n]{{0,30}}?{product_rx}"
    behind = rf"{product_rx}[^.!?\n]{{0,30}}?\s+{_NEG_PREFIX}\b"
    return re.compile(f"(?:{ahead}|{behind})", re.IGNORECASE)


# Built once at import; map product_key → compiled negation regex.
_NEG_PATTERNS: dict[str, re.Pattern] = {
    key: _neg_rx(rx.pattern) for key, rx in PRODUCT_PATTERNS.items()
}


# ---------------------------------------------------------------------------
# Parametric signal detectors
# ---------------------------------------------------------------------------
_SIZE_PATTERNS = {
    # "тираж 1000", "тираж 1 000 шт"
    "tiraж":       re.compile(r"\bтираж\s*[:\-]?\s*(\d[\d\s]*)", re.IGNORECASE),
    # "1000 шт", "500 штук", "100 экземпляров"
    "quantity":    re.compile(r"\b(\d[\d\s]{0,5}\d|\d)\s*(?:шт|штук|экземпл|ед)\b", re.IGNORECASE),
    # "высотой 40 см", "40см", "высота 50"
    "height_cm":   re.compile(r"\b(?:высот\w*\s+)?(\d{1,3})\s*см\b", re.IGNORECASE),
    # "3x4", "3×4", "3 на 4" (banners/signs)
    "dim_2d":      re.compile(r"\b(\d+)\s*[×хx]\s*(\d+)\b", re.IGNORECASE),
    # "на фасаде", "в помещении" — installation context
}

_PRICE_ASK_RX = re.compile(
    r"стоимост|стоит|сколько\s+(?:стоит|будет|обойд)|цен[аеуыёо]|расценк|"
    r"прайс|бюджет|смет\w*|почём|по\s+чём|за\s+сколько",
    re.IGNORECASE,
)

# Consultation intent (rule #10 from feedback): «Вы делаете X?», «Какие услуги?»,
# «Можете рассказать», «нужна консультация», «как выбрать», «что лучше».
# When matched, downstream pricing engines should defer in favour of presentation+discovery.
_CONSULTATION_INTENT_RX = re.compile(
    r"\bвы\s+делаете\b|\bвы\s+занимаетесь\b|\bкак\w*\s+услуг\w*\b|"
    r"\bконсультаци\w+\b|\bпорекомендуй\w*\b|\bпосовет\w+\b|"
    r"\bкак\s+выбра\w+\b|\bчто\s+лучше\b|\bрасскаж\w+\s+(?:о|про)\b|"
    r"\bнужн\w+\s+совет\b|\bпомог\w+\s+(?:разобрат|выбра)\b",
    re.IGNORECASE,
)

# Outdoor signage construction-type signals (rule #14): when user mentions
# montage/ад but doesn't name a concrete construction type, gate pricing.
_OUTDOOR_MOUNT_RX = re.compile(
    r"\b(?:монтаж|демонтаж|установк\w+)\b.*\b(?:рекламн\w*|вывеск\w*|конструкци\w+|наружн\w+)\b|"
    r"\b(?:наружн\w+\s+реклам\w+|нар\.\s*реклам\w+)\b",
    re.IGNORECASE,
)
_SIGN_TYPE_NAMED_RX = re.compile(
    r"\b(объ[её]мн\w+\s+букв|световой?\s+короб|неонов\w+\s+вывеск|фасадн\w+\s+вывеск|"
    r"баннер\w*|брандмауэр\w*|лайтбокс\w*|штендер\w*|стел[аы]\w*|плёнк\w+|пиллар\w*)\b",
    re.IGNORECASE,
)

# Mount city/address signal (rule #16): pricing must not include delivery/mount
# until address known. Detect any RU-locality mention — even "город X" — that
# proves the user has provided a location.
_LOCATION_RX = re.compile(
    r"\b(?:в\s+г\.|город\w*\s+|г\.\s*[А-ЯЁ])|"
    r"\b(?:Махачкал|Каспийск|Дербент|Хасавюрт|Буйнакск|Кизляр|Избербаш|"
    r"Москв|Питер|Краснодар|Ростов|Ставропол|Грозн|Нальчик)",
    re.IGNORECASE,
)
_MOUNT_REQUEST_RX = re.compile(
    r"\b(?:монтаж|демонтаж|выезд\w*|доставк\w+|установк\w+\s+на)\b",
    re.IGNORECASE,
)

# Brandmauer format-correction signal (chat #94/282): user mentions брендмауэр
# but assistant is suggesting volumetric letters. Brandmauer = banner on
# building wall, NOT volumetric letters.
_BRANDMAUER_RX = re.compile(r"\bбр[еа]ндмау[эе]р\w*\b", re.IGNORECASE)

_QUOTED_PRICE_RX = re.compile(
    # `₽` is not a word char in Unicode, so a trailing `\b` after it
    # NEVER matches (₽→any = two non-word chars = no boundary). Drop it.
    # Word-form tokens (руб/р.) still need their own word-boundary handling.
    r"(\d[\d\s]{2,})\s*(?:₽|руб(?:л\w+)?\b|р\.)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# DialogState dataclass
# ---------------------------------------------------------------------------
@dataclass
class DialogState:
    is_first_touch: bool = True
    """True when there are zero prior assistant turns."""

    rejected_products: set[str] = field(default_factory=set)
    """Product keys the customer explicitly refused (e.g. {'logo'})."""

    confirmed_product: str | None = None
    """Product the customer seems to be actively asking about (first named wins)."""

    mentioned_products: set[str] = field(default_factory=set)
    """All products mentioned anywhere in the conversation."""

    already_priced: list[tuple[str | None, float]] = field(default_factory=list)
    """(product_key, price_rub) pairs for every price quoted in prior assistant turns."""

    size_params: dict[str, str] = field(default_factory=dict)
    """Extracted тираж/высота/количество from any turn (values kept as strings)."""

    has_explicit_price_ask: bool = False
    """True when the LAST user turn asks for a price. Used by discovery gates."""

    is_consultation_intent: bool = False
    """True when the LAST user turn matches consultation patterns (rule #10).
    «Вы делаете X?», «Какие услуги?», «нужна консультация», «как выбрать».
    Pricing engines should defer in favour of presentation+discovery."""

    needs_signage_height: bool = False
    """Rule #15 gate: confirmed product is signboard/letters but no height_cm
    in size_params. Gate pricing — first ask «какая высота букв?»."""

    needs_signage_type: bool = False
    """Rule #14 gate: query asks about outdoor advertising mount/dismount
    without naming a concrete construction type."""

    needs_mount_location: bool = False
    """Rule #16 gate: query asks about mount/delivery but no city/address
    is mentioned anywhere in dialog. Don't add delivery to deal_items."""

    brandmauer_format_correction: bool = False
    """chat #94/282: user mentions брендмауэр. Assistant should treat it as
    banner/baner-on-wall, NOT volumetric letters. Used to inject a corrective
    nudge into the system context."""

    turn_count: int = 0
    """Number of user+assistant messages in history (excluding current)."""

    last_user_text: str = ""
    """Raw last user message (for logging/debug)."""

    def is_rejected(self, product_key: str | None) -> bool:
        if not product_key:
            return False
        return product_key in self.rejected_products

    def product_context_switch(self, candidate_product: str | None) -> bool:
        """True when `candidate_product` differs from what the user is actively
        asking about (confirmed_product). We suppress SmetaEngine on switch —
        it usually means the engine matched the wrong category due to a stray
        keyword in the latest message.
        """
        if not candidate_product or not self.confirmed_product:
            return False
        return candidate_product != self.confirmed_product

    def discovery_question(self) -> str | None:
        """Returns the most pressing missing parameter as a customer-facing
        question, or None if no parametric gate is open. Caller should treat
        this as an instruction to skip pricing and ask the question instead.

        Priority: signage type > height > mount location. Only one question
        per turn — too many at once kills conversation flow.
        """
        if self.needs_signage_type:
            return (
                "Уточните, пожалуйста, тип конструкции: объёмные буквы, "
                "световой короб, неоновая вывеска, баннер или фасадная "
                "вывеска? Цена сильно зависит от типа."
            )
        if self.needs_signage_height:
            return (
                "Какая высота букв планируется? Цена за букву 20 см и 80 см "
                "отличается в 3-5 раз — без размера точная цена невозможна."
            )
        if self.needs_mount_location:
            return (
                "В каком городе/адресе нужен монтаж? От этого зависят "
                "позиции выезда, доставки и стоимость монтажных работ."
            )
        return None

    @property
    def needs_discovery_turn(self) -> bool:
        """True when the bot should run a discovery/qualifier turn instead of
        quoting a template price. This is narrower than ``is_first_touch``:
        if the very first user message already carries specifics
        (explicit price ask, тираж/размер/quantity), the client wants a
        number — not a sales-funnel intro. Blocking SmetaEngine on those
        would regress auto-pricing against single-turn eval cases.
        """
        if not self.is_first_touch:
            return False
        if self.has_explicit_price_ask:
            return False
        if self.size_params:
            return False
        return True


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
def _find_products(text: str) -> set[str]:
    out: set[str] = set()
    for key, rx in PRODUCT_PATTERNS.items():
        if rx.search(text):
            out.add(key)
    return out


def _find_rejections(text: str) -> set[str]:
    out: set[str] = set()
    for key, rx in _NEG_PATTERNS.items():
        if rx.search(text):
            out.add(key)
    return out


def _find_sizes(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, rx in _SIZE_PATTERNS.items():
        m = rx.search(text)
        if m:
            out[key] = m.group(0).strip()
    return out


def _extract_quoted_prices(text: str) -> list[float]:
    out: list[float] = []
    for m in _QUOTED_PRICE_RX.finditer(text):
        raw = m.group(1).replace(" ", "").replace("\u00a0", "")
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def _msg_role_content(msg) -> tuple[str, str] | None:
    """Accept both ChatMessage dataclasses/pydantic and dict-like entries."""
    try:
        role = getattr(msg, "role", None) or msg["role"]
        content = getattr(msg, "content", None) or msg["content"]
    except (KeyError, TypeError):
        return None
    return str(role or ""), str(content or "")


def extract(history, current_query: str = "") -> DialogState:
    """Build a DialogState from chat history + current user query.

    Args:
        history: iterable of ChatMessage-like or dict-like {'role','content'}.
                 Expected ordering: oldest first. Safe to pass None.
        current_query: the message the user just sent; contributes to
                       rejections/sizes/product signals but NOT to turn_count
                       or to `is_first_touch` (which reflects prior state only).
    """
    state = DialogState()
    if not history:
        history = []

    # Normalise to list[(role, content)]
    turns: list[tuple[str, str]] = []
    for msg in history:
        rc = _msg_role_content(msg)
        if rc is None:
            continue
        role, content = rc
        if not content.strip():
            continue
        turns.append((role, content))

    state.turn_count = len(turns)
    state.is_first_touch = not any(r == "assistant" for r, _ in turns)

    # Walk turns: lock confirmed_product on the earliest USER turn that names one
    confirmed_locked = False
    for role, content in turns:
        prods = _find_products(content)
        state.mentioned_products |= prods
        if role == "user" and not confirmed_locked and prods:
            # Prefer the first product mentioned positionally within the text
            first = _earliest_product(content, prods)
            if first:
                state.confirmed_product = first
                confirmed_locked = True

        # rejections from user turns only (assistant doesn't reject on customer's behalf)
        if role == "user":
            state.rejected_products |= _find_rejections(content)

        # size params from any turn
        state.size_params.update(_find_sizes(content))

        # prices already quoted by assistant
        if role == "assistant":
            for price in _extract_quoted_prices(content):
                # associate price with the product mentioned in same assistant turn (best-effort)
                prod = _earliest_product(content, prods) if prods else None
                state.already_priced.append((prod, price))

    # Apply current_query AFTER history walk so it can (a) override confirmed
    # product if the user just switched topic, (b) add rejections, (c) set
    # has_explicit_price_ask.
    q = current_query or ""
    if q:
        q_prods = _find_products(q)
        state.mentioned_products |= q_prods
        state.rejected_products |= _find_rejections(q)
        state.size_params.update(_find_sizes(q))
        state.has_explicit_price_ask = bool(_PRICE_ASK_RX.search(q))
        state.last_user_text = q[:400]
        # Topic switch: if current query names a product AND that product is
        # not in mentioned_products from earlier, treat it as the new
        # confirmed product IFF it's not rejected.
        if q_prods:
            first = _earliest_product(q, q_prods)
            if first and first not in state.rejected_products:
                # Only override if we didn't already lock a product OR if the
                # user is explicitly naming a new one AND not the same as before.
                # Keep original confirmed_product unless current query is clearly about
                # a different product (e.g. earlier visitки, now "мне еще нужна вывеска").
                if not state.confirmed_product:
                    state.confirmed_product = first
                elif first != state.confirmed_product and _is_topic_switch(q):
                    state.confirmed_product = first

    # ---- New parametric gates (rules #14, #15, #16, #10) ----
    # Aggregate text from all user turns + current query for detection scope.
    user_text_all = " ".join(c for r, c in turns if r == "user") + " " + (q or "")

    # Rule #10: consultation intent — match against the LAST user message only,
    # since intent shifts turn-by-turn.
    state.is_consultation_intent = bool(_CONSULTATION_INTENT_RX.search(q or ""))

    # Rule #15: signage with letters/volumetric needs height. Trigger when
    # confirmed_product is signboard/neon AND no height_cm in size_params.
    if state.confirmed_product in {"signboard", "neon"} and "height_cm" not in state.size_params:
        # Only gate when user actually asks for price — first-touch discovery
        # already covers the "no specifics yet" case.
        if state.has_explicit_price_ask or state.size_params.get("quantity"):
            state.needs_signage_height = True

    # Rule #14: outdoor mount/dismount without named construction type.
    if _OUTDOOR_MOUNT_RX.search(user_text_all) and not _SIGN_TYPE_NAMED_RX.search(user_text_all):
        state.needs_signage_type = True

    # Rule #16: mount request without city/address.
    if _MOUNT_REQUEST_RX.search(user_text_all) and not _LOCATION_RX.search(user_text_all):
        state.needs_mount_location = True

    # chat #94/282: брендмауэр format-correction nudge.
    if _BRANDMAUER_RX.search(user_text_all):
        state.brandmauer_format_correction = True

    return state


def _earliest_product(text: str, candidates: set[str]) -> str | None:
    """From a set of matched products, return the one whose first hit has
    the lowest start offset in `text`.
    """
    best_key: str | None = None
    best_pos = 10**9
    for key in candidates:
        rx = PRODUCT_PATTERNS.get(key)
        if rx is None:
            continue
        m = rx.search(text)
        if m and m.start() < best_pos:
            best_pos = m.start()
            best_key = key
    return best_key


# Topic-switch cues: "теперь", "еще нужна", "также", "плюс к этому", "а еще"
_TOPIC_SWITCH_RX = re.compile(
    r"\b(теперь|ещё\s+нужн|еще\s+нужн|также\s+нужн|плюс|а\s+ещё|а\s+еще|кроме\s+того|добавим)",
    re.IGNORECASE,
)


def _is_topic_switch(text: str) -> bool:
    return bool(_TOPIC_SWITCH_RX.search(text))


# ---------------------------------------------------------------------------
# System-prompt injection helper
# ---------------------------------------------------------------------------
PRODUCT_LABELS_RU: dict[str, str] = {
    "logo":          "логотип",
    "brandbook":     "брендбук",
    "signboard":     "вывеска",
    "business_card": "визитки",
    "flyer":         "листовки",
    "banner":        "баннер",
    "booklet":       "буклет",
    "sticker":       "наклейки",
    "menu":          "меню",
    "sign_table":    "табличка",
    "stand":         "штендер",
    "bracket":       "панель-кронштейн",
    "packaging":     "упаковка",
    "neon":          "неоновая вывеска",
    "design":        "дизайн",
}


def _ru_product(key: str) -> str:
    return PRODUCT_LABELS_RU.get(key, key)


def build_system_context_block(
    state: DialogState,
    smeta_category: str | None = None,
) -> str:
    """Render a prompt fragment that describes the current dialog constraints.

    This string is meant to be prepended to the `extra_context` passed into
    `generator.generate_*`. The LLM sees it as hard rules: first-touch
    discovery, rejected products to avoid, the active product to stay on,
    prices already quoted, etc.

    When `smeta_category` is provided, we also flag if the SmetaEngine
    template landed on a rejected/off-topic category — useful for the LLM to
    understand WHY the template result (if present in context) should be
    ignored.

    Returns "" when the state is effectively empty (nothing to tell the
    model). Callers should concatenate unconditionally — empty string is a
    no-op.
    """
    lines: list[str] = []

    if state.needs_discovery_turn:
        lines.append(
            "РЕЖИМ: первое касание клиента. НЕ называй конкретную цену до "
            "того как выяснишь потребность. Задай 1-2 уточняющих вопроса "
            "(сфера бизнеса, тираж/размер, есть ли макет/ТЗ, сроки). "
            "Короткая презентация направления и компании уместна. "
            "Оценочные диапазоны «от X ₽» допустимы, но итоговую цену — "
            "только после параметров."
        )

    if state.is_consultation_intent and not state.has_explicit_price_ask:
        lines.append(
            "РЕЖИМ КОНСУЛЬТАЦИИ: клиент просит совет / консультацию / спрашивает "
            "«вы делаете X?». Это НЕ запрос сметы. Презентуй услугу, расскажи "
            "что входит и какой результат, приведи 1 пример из практики Лабус. "
            "НЕ считай итоговую цену, не выдавай deal_items, пока клиент сам "
            "явно не спросит «сколько стоит». Закончи 1 уточняющим вопросом "
            "о сфере бизнеса/задаче клиента."
        )

    # Parametric gates from active feedback rules — render as TOP-priority
    # blockers since the LLM has been ignoring them in prod.
    pending_question = state.discovery_question()
    if pending_question:
        lines.append(
            f"⛔ ОБЯЗАТЕЛЬНЫЙ ВОПРОС ПЕРЕД РАСЧЁТОМ: «{pending_question}» "
            f"Ответ должен быть ТОЛЬКО этим вопросом плюс короткий контекст почему "
            f"он важен (1-2 предложения). НЕ выдавай оценочную цену, НЕ заполняй "
            f"deal_items, НЕ упоминай готовые пакеты — пока клиент не ответит."
        )

    if state.brandmauer_format_correction:
        lines.append(
            "⚠ БРЕНДМАУЭР — это печать (сольвент/перфо-баннер) на торце здания. "
            "НИКОГДА не предлагай объёмные буквы или световые конструкции для "
            "брендмауэра — это техническая ошибка. Цена — за квадратный метр "
            "печати + материал + монтаж на высоте."
        )

    if state.rejected_products:
        ru = ", ".join(_ru_product(p) for p in sorted(state.rejected_products))
        lines.append(
            f"ОТКАЗАНО КЛИЕНТОМ: {ru}. Эти услуги клиент уже явно отклонил "
            f"в предыдущих сообщениях. НЕ предлагай их повторно, даже если "
            f"каталог/шаблон матчит. Не поднимай тему без прямого запроса клиента."
        )

    if state.confirmed_product and state.confirmed_product not in state.rejected_products:
        ru = _ru_product(state.confirmed_product)
        lines.append(
            f"АКТИВНЫЙ ПРОДУКТ: {ru}. Ответы — в контексте этого продукта. "
            f"Переключайся на другой продукт только если клиент явно сказал "
            f"«ещё нужна ...», «также», «плюс», «а ещё». Одиночное упоминание "
            f"другого термина в вопросе НЕ означает смену темы."
        )

    if state.size_params:
        pretty = ", ".join(f"{k}={v}" for k, v in state.size_params.items())
        lines.append(f"УЖЕ ИЗВЕСТНО ИЗ ДИАЛОГА: {pretty}. Не переспрашивай.")

    if state.already_priced:
        uniq = sorted({int(round(p)) for _, p in state.already_priced if p})
        if uniq:
            pretty = ", ".join(f"{p:,}".replace(",", " ") + " ₽" for p in uniq[:6])
            lines.append(
                f"УЖЕ НАЗЫВАЛИ ЦЕНЫ: {pretty}. Не повторяй те же цифры, если "
                f"параметры не изменились. Если цена изменилась — объясни почему."
            )

    if smeta_category:
        candidate = smeta_category_to_product(smeta_category)
        if candidate and state.is_rejected(candidate):
            lines.append(
                f"⚠ Шаблон SmetaEngine попал в отклонённую категорию "
                f"«{smeta_category}» — ИГНОРИРУЙ цифры из этого шаблона."
            )
        elif candidate and state.product_context_switch(candidate):
            lines.append(
                f"⚠ Шаблон SmetaEngine («{smeta_category}») не совпадает с "
                f"активным продуктом — ИГНОРИРУЙ цифры из этого шаблона."
            )

    if not lines:
        return ""
    return "[ДИАЛОГ-СОСТОЯНИЕ]\n" + "\n".join(f"• {ln}" for ln in lines) + "\n"
