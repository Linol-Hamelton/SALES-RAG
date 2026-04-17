"""Tests for app.core.dialog_state.

Run directly:  python RAG_RUNTIME/scripts/test_dialog_state.py
Exit code 0 on success, non-zero on any failure.

Cases are grounded in the prod chat #93 failure (stoma/визитки dialog,
16-17 Apr 2026), plus additional coverage of product-vocabulary edge cases.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure we import from the repo's app/ package (not some installed wheel).
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


def test_first_touch_on_empty_history() -> None:
    print("[first-touch detection]")
    st = extract(history=[], current_query="Вы делаете логотипы?")
    _assert(st.is_first_touch is True, "empty history → is_first_touch")
    _assert(st.turn_count == 0, "turn_count == 0")
    _assert(st.confirmed_product == "logo", "confirmed_product = logo",
            f"got {st.confirmed_product}")
    _assert(st.has_explicit_price_ask is False, "no price ask in plain qualifier")


def test_not_first_touch_after_assistant_turn() -> None:
    print("\n[non-first-touch after assistant turn]")
    history = [
        _msg("user", "Вы делаете логотипы?"),
        _msg("assistant", "Да, делаем. Расскажите про ваш бизнес?"),
    ]
    st = extract(history=history, current_query="кофейня")
    _assert(st.is_first_touch is False, "is_first_touch False")
    _assert(st.turn_count == 2, "turn_count 2")


def test_rejection_logo_classic_form() -> None:
    print("\n[rejection: 'не нужен логотип']")
    history = [
        _msg("user", "Нужны визитки"),
        _msg("assistant", "Для визиток нужен тираж. Хотите дизайн логотипа?"),
    ]
    st = extract(history=history, current_query="у меня нет логотипа/брендбука и он мне не нужен")
    _assert("logo" in st.rejected_products, "logo rejected",
            f"rejected={st.rejected_products}")
    _assert("brandbook" in st.rejected_products, "brandbook rejected")


def test_rejection_logo_reverse_form() -> None:
    print("\n[rejection: 'логотип не нужен']")
    st = extract(history=[], current_query="логотип не нужен")
    _assert("logo" in st.rejected_products, "logo rejected reverse form")


def test_rejection_signboard() -> None:
    print("\n[rejection: 'мне не нужна вывеска']")
    st = extract(history=[], current_query="мне не нужна вывеска, только визитки")
    _assert("signboard" in st.rejected_products, "signboard rejected")
    _assert("logo" not in st.rejected_products, "logo NOT rejected (false-positive guard)")


def test_confirmed_product_lock_first_user_turn() -> None:
    print("\n[confirmed_product lock: earliest user turn wins]")
    history = [
        _msg("user", "Нужны визитки"),
        _msg("assistant", "Конечно. Какой тираж?"),
        _msg("user", "1000 визиток и нужен дизайн"),
    ]
    st = extract(history=history, current_query="можете предложить?")
    _assert(st.confirmed_product == "business_card", "confirmed = business_card",
            f"got {st.confirmed_product}")


def test_topic_switch_takes_effect() -> None:
    print("\n[topic switch updates confirmed_product]")
    history = [
        _msg("user", "1000 визиток с дизайном"),
        _msg("assistant", "Стоимость ~5000₽."),
    ]
    st = extract(history=history, current_query="мне еще нужна вывеска объемные буквы")
    _assert(st.confirmed_product == "signboard", "confirmed switched to signboard",
            f"got {st.confirmed_product}")


def test_no_topic_switch_without_marker() -> None:
    print("\n[no topic switch without 'ещё/также/плюс' marker]")
    history = [
        _msg("user", "Нужны визитки"),
        _msg("assistant", "Какой тираж?"),
    ]
    # bare mention of 'логотип' while dialogue is about visitki — DO NOT switch
    st = extract(history=history, current_query="и вообще как работает логотип")
    _assert(st.confirmed_product == "business_card",
            "confirmed stays on business_card (no switch marker)",
            f"got {st.confirmed_product}")


def test_size_params_extraction() -> None:
    print("\n[size params: тираж / высота / quantity]")
    history = [
        _msg("user", "Листовки А4 тираж 1000 шт"),
    ]
    st = extract(history=history, current_query="")
    _assert("tiraж" in st.size_params, "tiraж captured")
    _assert("quantity" in st.size_params, "quantity captured")
    st2 = extract(history=[], current_query="буквы высотой 80 см")
    _assert("height_cm" in st2.size_params, "height captured")


def test_explicit_price_ask() -> None:
    print("\n[explicit price ask detection]")
    s1 = extract(history=[], current_query="Сколько стоит брендбук?")
    _assert(s1.has_explicit_price_ask is True, "explicit price ask True")
    s2 = extract(history=[], current_query="Вы делаете логотипы?")
    _assert(s2.has_explicit_price_ask is False, "no price ask False")
    s3 = extract(history=[], current_query="расценки на листовки А4")
    _assert(s3.has_explicit_price_ask is True, "'расценки' → price ask")


def test_already_priced_collected() -> None:
    print("\n[already-priced collected from assistant turns]")
    history = [
        _msg("user", "Нужны визитки, 1000 шт"),
        _msg("assistant", "Для 1000 визиток с дизайном ~5030–6030 ₽. Нужен логотип?"),
    ]
    st = extract(history=history, current_query="ну да")
    _assert(len(st.already_priced) >= 1, "at least one price captured",
            f"got {st.already_priced}")


def test_category_to_product_mapping() -> None:
    print("\n[category name → product mapping]")
    _assert(smeta_category_to_product("Логотип") == "logo", "Логотип → logo")
    _assert(smeta_category_to_product("Объемные буквы") == "signboard", "Объемные буквы → signboard")
    _assert(smeta_category_to_product("Световые вывески") == "signboard", "Световые вывески → signboard")
    _assert(smeta_category_to_product("Визитки") == "business_card", "Визитки → business_card")
    _assert(smeta_category_to_product("") is None, "empty → None")
    _assert(smeta_category_to_product("Какая-то неизвестная") is None, "unknown → None")


def test_chat93_end_to_end_key_turn() -> None:
    """Replay the critical msg270 turn from prod chat #93.

    At msg270, the customer had already said «не нужен логотип» twice,
    explicitly written «хельветикой стоматология05 — вот мой логотип»,
    and is now sending «вот логотип» as an attachment reply.

    The system wrongly proposed a вывеска quote. Our dialog_state MUST flag:
      - logo is rejected  → any logo-upsell must be suppressed
      - business_card is the active product (set in turn 1)
      - current query should NOT trigger a topic switch (no «ещё/также/плюс»)
    """
    print("\n[end-to-end: prod chat #93 msg270 replay]")
    history = [
        _msg("user", "Нужны визитки"),
        _msg("assistant", "Оценка по шаблону категории «Визитки»… 4 456 ₽."),
        _msg("user", "я хочу заказать их"),
        _msg("assistant", "Для визиток нужен тираж. Хотите дизайн?"),
        _msg("user", "1000 визиток и нужен дизайн"),
        _msg("assistant", "1000 визиток с дизайном ~5030–6030 ₽."),
        _msg("user", "у меня нет логотипа/брендбука и он мне не нужен. напиши просто стоматологи05"),
        _msg("assistant", "Разработка логотипа 30 000–50 000 ₽."),
        _msg("user", "ле что за цена? мне не нужен логотип"),
        _msg("assistant", "Текстовый дизайн ~5030 ₽."),
        _msg("user", "напиши просто хельветикой стоматология05 - вот мой логотип"),
        _msg("assistant", "Разработка логотипа 27 500–75 000 ₽."),
    ]
    st = extract(history=history, current_query="вот логотип")
    _assert(st.is_first_touch is False, "not first touch")
    _assert(st.confirmed_product == "business_card",
            "confirmed stays business_card despite later mentions of логотип",
            f"got {st.confirmed_product}")
    _assert("logo" in st.rejected_products, "logo rejected",
            f"rejected={st.rejected_products}")
    # Topic switch requires «ещё/также/плюс»; bare «вот логотип» does NOT switch.
    # (Whether signboard is offered or not is a separate decision, but the
    #  key signal here is that logo is rejected and business_card is active.)


def test_build_system_context_block_first_touch() -> None:
    print("\n[system context: first-touch block]")
    st = extract(history=[], current_query="Вы делаете логотипы?")
    block = build_system_context_block(st)
    _assert("первое касание" in block.lower(), "first-touch phrase present")
    _assert("не называй конкретную цену" in block.lower(), "no-price rule present")
    _assert("логотип" in block.lower(), "active product mentioned",
            f"block={block!r}")


def test_build_system_context_block_rejection_and_context() -> None:
    print("\n[system context: rejection + active product]")
    history = [
        _msg("user", "Нужны визитки"),
        _msg("assistant", "Для визиток нужен тираж."),
    ]
    st = extract(history=history, current_query="логотип мне не нужен")
    block = build_system_context_block(st, smeta_category="Логотип")
    _assert("отказано клиентом" in block.lower(), "rejection clause present")
    _assert("логотип" in block.lower(), "rejected product named")
    _assert("активный продукт" in block.lower(), "active product clause present")
    _assert("визитки" in block.lower(), "business_card named as active")
    _assert("игнорируй" in block.lower(), "smeta override warning present")


def test_build_system_context_block_empty_state() -> None:
    print("\n[system context: empty state → empty string]")
    # Assistant already answered (not first touch), no rejections, no products,
    # no priced quotes, no sizes.
    history = [
        _msg("user", "здравствуйте"),
        _msg("assistant", "здравствуйте, чем помочь?"),
    ]
    st = extract(history=history, current_query="")
    block = build_system_context_block(st)
    _assert(block == "", "empty state renders empty string",
            f"got {block!r}")


def test_build_system_context_block_already_priced() -> None:
    print("\n[system context: already-priced included]")
    history = [
        _msg("user", "Нужны визитки, 1000 шт"),
        _msg("assistant", "1000 визиток с дизайном ~5030–6030 ₽."),
    ]
    st = extract(history=history, current_query="ну да")
    block = build_system_context_block(st)
    _assert("уже называли цены" in block.lower(),
            "priced clause present", f"block={block!r}")


def test_product_context_switch_and_is_rejected_helpers() -> None:
    print("\n[helpers: product_context_switch + is_rejected]")
    history = [_msg("user", "Нужны визитки")]
    st = extract(history=history, current_query="")
    _assert(st.product_context_switch("logo") is True,
            "candidate logo vs confirmed business_card → switch")
    _assert(st.product_context_switch("business_card") is False,
            "same product → no switch")
    _assert(st.product_context_switch(None) is False, "None candidate → no switch")
    _assert(st.is_rejected(None) is False, "None rejected → False")


def run_all() -> int:
    tests = [
        test_first_touch_on_empty_history,
        test_not_first_touch_after_assistant_turn,
        test_rejection_logo_classic_form,
        test_rejection_logo_reverse_form,
        test_rejection_signboard,
        test_confirmed_product_lock_first_user_turn,
        test_topic_switch_takes_effect,
        test_no_topic_switch_without_marker,
        test_size_params_extraction,
        test_explicit_price_ask,
        test_already_priced_collected,
        test_category_to_product_mapping,
        test_chat93_end_to_end_key_turn,
        test_build_system_context_block_first_touch,
        test_build_system_context_block_rejection_and_context,
        test_build_system_context_block_empty_state,
        test_build_system_context_block_already_priced,
        test_product_context_switch_and_is_rejected_helpers,
    ]
    for t in tests:
        t()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} / {sum(1 for _ in tests)}")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print(f"ALL OK ({sum(1 for _ in tests)} tests)")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(run_all())
