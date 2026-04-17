"""Regression test for prod chat #93 (2026-04-16, Rabadan, dentistry visitki).

The chat failed in 6 places. This test freezes the dialog_state signals
that the SmetaEngine gate and prompt injector rely on at each of those
moments. A regression in dialog_state.py or in the chat #93 understanding
will fail here with a specific label.

Run:  python RAG_RUNTIME/scripts/test_chat93_regression.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAG_ROOT = HERE.parent
sys.path.insert(0, str(RAG_ROOT))

from app.core.dialog_state import (  # noqa: E402
    build_system_context_block,
    extract,
    smeta_category_to_product,
)


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


FAILS: list[str] = []


def _assert(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  OK    {label}")
    else:
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))
        FAILS.append(label)


# Canonical history slices aligned to actual prod messages.
MSG262 = []  # empty: first user turn
MSG264 = [
    _msg("user", "Нужны визитки"),
    _msg("assistant", "Оценка по шаблону категории «Визитки»… 4 456 ₽. На какой тираж?"),
]
MSG268 = MSG264 + [
    _msg("user", "я хочу заказать их"),
    _msg("assistant", "Уточните тираж и нужен ли дизайн."),
    _msg("user", "1000 визиток и нужен дизайн"),
    _msg("assistant", "1000 визиток с дизайном ~5030–6030 ₽. Пришлите логотип или брендбук."),
]
MSG270 = MSG268 + [
    _msg("user", "у меня нет логотипа/брендбука и он мне не нужен. напиши просто стоматологи05"),
    _msg("assistant", "Разработка логотипа 30 000–50 000 ₽."),
]
MSG272 = MSG270 + [
    _msg("user", "ле что за цена? мне не нужен логотип"),
    _msg("assistant", "Текстовый дизайн ~5030 ₽."),
]
MSG274 = MSG272 + [
    _msg("user", "напиши просто хельветикой стоматология05 - вот мой логотип"),
    _msg("assistant", "Разработка логотипа 27 500–75 000 ₽."),
]


def test_msg262_first_touch_blocks_pricing() -> None:
    """msg262 «Нужны визитки» — should trigger first-touch discovery mode.

    Manager feedback: «Не переходи сразу к оценке. Сначала проведи
    уточняющий вводный разговор и проведи презентацию компании».
    """
    print("\n[msg262 — first touch]")
    st = extract(history=MSG262, current_query="Нужны визитки")
    _assert(st.is_first_touch is True, "first_touch signals discovery")
    _assert(st.confirmed_product == "business_card", "confirmed = business_card")
    _assert("logo" not in st.rejected_products, "logo not rejected yet")

    block = build_system_context_block(st)
    _assert("первое касание" in block.lower(), "ctx block enables discovery mode")
    _assert("не называй конкретную цену" in block.lower(),
            "ctx block suppresses immediate pricing")


def test_msg268_logo_rejection_seen_from_current_turn() -> None:
    """msg268 «нет логотипа/брендбука и он мне не нужен» — logo and brandbook
    should be rejected the moment this message arrives.

    Manager feedback: «Нельзя так сразу переходить на продажу логотипа.»
    """
    print("\n[msg268 — reject logo + brandbook]")
    st = extract(
        history=MSG268,
        current_query="у меня нет логотипа/брендбука и он мне не нужен. напиши просто стоматологи05",
    )
    _assert("logo" in st.rejected_products, "logo rejected",
            f"rejected={st.rejected_products}")
    _assert("brandbook" in st.rejected_products, "brandbook rejected")
    _assert(st.confirmed_product == "business_card",
            "confirmed still business_card (no switch marker)")

    # Simulate SmetaEngine returning «Логотип» — gate must see it as rejected.
    cat = "Логотип"
    prod = smeta_category_to_product(cat)
    _assert(prod == "logo", "category → product mapping")
    _assert(st.is_rejected(prod) is True,
            "gate: rejection must block logo-category override")

    block = build_system_context_block(st, smeta_category=cat)
    _assert("отказано клиентом" in block.lower(),
            "ctx block warns LLM about rejection")
    _assert("игнорируй" in block.lower(),
            "ctx block tells LLM to ignore the smeta template")


def test_msg270_logo_rejection_reverse_phrase() -> None:
    """msg270 «ле что за цена? мне не нужен логотип» — reverse-order phrase
    («мне не нужен логотип»). Masc. short adj. «нужен» (fleeting vowel) must
    be matched by the stem `нуж\\w*`, not the over-specific `нужн\\w*`.

    This was the exact regex bug fixed when building the gate.
    """
    print("\n[msg270 — reverse rejection 'мне не нужен логотип']")
    st = extract(
        history=MSG270,
        current_query="ле что за цена? мне не нужен логотип",
    )
    _assert("logo" in st.rejected_products,
            "reverse-phrase rejection detected",
            f"rejected={st.rejected_products}")


def test_msg274_context_switch_blocks_signboard_override() -> None:
    """msg274 «вот логотип» — prod critically offered a ВЫВЕСКА 25-80k here.

    Manager feedback: «Клиент сам написал шрифтом и отправил чтобы ему
    сделали визитку, но вместо этого мы начинаем предлагать ему вывеску —
    это критическая ошибка!»

    Gate must:
      (a) keep confirmed_product == 'business_card'
      (b) rejected_products still contains 'logo' (from earlier turns)
      (c) SmetaEngine category «Световые вывески» → 'signboard'
      (d) product_context_switch('signboard') is True — confirmed is
          'business_card' and current turn lacks «ещё/также/плюс» markers
    """
    print("\n[msg274 — context switch 'вот логотип']")
    st = extract(history=MSG274, current_query="вот логотип")
    _assert(st.confirmed_product == "business_card",
            "confirmed stays business_card", f"got {st.confirmed_product}")
    _assert("logo" in st.rejected_products,
            "logo still rejected across turns")

    # Simulate the prod bug: SmetaEngine returned «Световые вывески».
    cat = "Световые вывески"
    prod = smeta_category_to_product(cat)
    _assert(prod == "signboard", "category mapped to signboard")
    _assert(st.product_context_switch(prod) is True,
            "gate: context-switch must block this override")

    block = build_system_context_block(st, smeta_category=cat)
    _assert("активный продукт" in block.lower(),
            "ctx block reminds LLM of active product")
    _assert("визитки" in block.lower(),
            "ctx block names business_card as active")
    _assert("не совпадает с активным продуктом" in block.lower(),
            "ctx block flags the smeta category as off-topic")


def test_msg276_direct_print_request_not_first_touch() -> None:
    """Late turn «все есть макет визитки в векторе - сколько стоит печать 1000шт».

    At this point the dialog already has pricing history and an explicit
    product. This is NOT first touch — the gate must let SmetaEngine run
    (but rejection of logo still holds).
    """
    print("\n[msg276 — direct-print request, not first touch]")
    late_history = MSG274 + [
        _msg("user", "вот логотип"),
        _msg("assistant", "Для вывески по вашему макету 25 000–80 000 ₽."),
    ]
    st = extract(
        history=late_history,
        current_query="все есть макет визитки в векторе - сколько стоит распечатать 1000шт",
    )
    _assert(st.is_first_touch is False, "not first touch by this point")
    _assert(st.has_explicit_price_ask is True, "explicit price ask detected")
    _assert(st.confirmed_product == "business_card", "still business_card")
    _assert("logo" in st.rejected_products, "logo rejection persists")

    # SmetaEngine returning «Визитки» here IS appropriate.
    cat = "Визитки"
    prod = smeta_category_to_product(cat)
    _assert(prod == "business_card", "category matches confirmed product")
    _assert(st.is_rejected(prod) is False, "business_card not rejected")
    _assert(st.product_context_switch(prod) is False,
            "no context switch — category matches active product")


def run_all() -> int:
    tests = [
        test_msg262_first_touch_blocks_pricing,
        test_msg268_logo_rejection_seen_from_current_turn,
        test_msg270_logo_rejection_reverse_phrase,
        test_msg274_context_switch_blocks_signboard_override,
        test_msg276_direct_print_request_not_first_touch,
    ]
    for t in tests:
        t()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} / {sum(1 for _ in tests)} test cases")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print(f"ALL OK ({len(tests)} tests, chat #93 regression suite)")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(run_all())
