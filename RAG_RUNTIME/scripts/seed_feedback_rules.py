"""
Seed curated feedback_rules into the RAG database.

Rules are Tier 2 (most authoritative) — injected as "ПРАВИЛА ОТВЕТА (обязательные):"
into the LLM prompt before auto-matched lessons.

Usage:
    # Test against local dump:
    python scripts/seed_feedback_rules.py --db _dbdump/2026-04-13/labus_rag.db

    # Apply to production DB (after docker cp):
    python scripts/seed_feedback_rules.py --db /app/data/labus_rag.db
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path


RULES = [
    # --- Priority 10: Hard business rules ---
    {
        "rule_text": (
            "Мы B2B-компания: НЕ даём скидок и НЕ обсуждаем дисконт с клиентом в ответе. "
            "Прогрессивный дисконт по объёму закупки встроен в прайс-лист, но НЕ упоминай "
            "его в диалоге — это обесценивает услугу. Если клиент прямо просит скидку — "
            "переключись на аргументы качества, кейсы, ROI, НЕ предлагай снижение цены."
        ),
        "direction": "",
        "priority": 10,
        "source_ids": json.dumps(["fb#99"]),
    },
    {
        "rule_text": (
            "Разработка логотипа — ПАКЕТНОЕ ценообразование: "
            "«Стандарт» 27 500–48 500 ₽ (1-3 варианта, ручная отрисовка), "
            "«Базовый» 34 000–58 500 ₽ (3-5 концепций, логобук 12 слайдов), "
            "«Креативная группа» 44 000–75 000 ₽ (до 5 концепций, индивидуальный логобук, паттерн). "
            "Пакеты — это БАЗОВЫЕ ТОЧКИ для переговоров, клиенты почти всегда кастомизируют. "
            "Не навязывай пакет силой: если клиент возражает — аргументируй ценностью (кейсы, "
            "бренды, качество отдела дизайна, ROMI 180%), а не повторением цены."
        ),
        "direction": "Дизайн",
        "priority": 10,
        "source_ids": json.dumps(["roadmap:Дизайн логотипа.md", "fb#85", "fb#88", "fb#107"]),
    },

    # --- Priority 9: Intent-level rules ---
    {
        "rule_text": (
            "Если клиент спрашивает «Вы делаете X?» или «Какие услуги?» — "
            "это КОНСУЛЬТАЦИЯ, а не запрос сметы. На раннем этапе диалога: "
            "(1) презентуй услугу и компанию (15+ лет опыта, собственный цех, крупные кейсы), "
            "(2) уточни сферу бизнеса клиента и причину запроса, "
            "(3) приведи пример успешной работы в его нише (если есть), "
            "(4) ТОЛЬКО ПОСЛЕ ЭТОГО презентуй пакеты как базовые точки для переговоров. "
            "НЕ начинай с цены."
        ),
        "direction": "",
        "priority": 9,
        "source_ids": json.dumps(["fb#79", "fb#85", "fb#88"]),
    },
    {
        "rule_text": (
            "Работа с возражением «слишком дорого» / «у конкурента дешевле»: "
            "НЕ снижай цену, НЕ оправдывайся. Аргументы: "
            "(1) логотип напрямую влияет на продажи и узнаваемость бренда, "
            "(2) дешёвая работа (10К за логотип) обычно неполноценная — потребует переделки, "
            "(3) наш отдел дизайна — специалисты высокого уровня с портфолио крупных брендов, "
            "(4) ROMI наших логотипов — 180%+ при правильной интеграции. "
            "Задай открытый вопрос о бизнес-задачах клиента."
        ),
        "direction": "Дизайн",
        "priority": 9,
        "source_ids": json.dumps(["fb#105", "fb#107"]),
    },

    # --- Priority 8: Data grounding rules ---
    {
        "rule_text": (
            "Пункты сметы (deal_items) ДОЛЖНЫ состоять из названий товаров из каталога goods.csv. "
            "Не выдумывай абстрактные позиции вроде «Дизайн» — используй конкретные артикулы: "
            "«Заполнение брифа», «Креативно-графическая идея», «Ручная отрисовка» и т.д."
        ),
        "direction": "",
        "priority": 8,
        "source_ids": "[]",
    },

    # --- Priority 7: Service-specific pricing ---
    {
        "rule_text": (
            "Фирменный стиль = 50 000–100 000 ₽. Брендбук = 100 000–300 000 ₽. "
            "Это РАЗНЫЕ услуги разного масштаба. Не путай их и не завышай стоимость "
            "брендбука до миллионов."
        ),
        "direction": "Дизайн",
        "priority": 7,
        "source_ids": json.dumps([
            "roadmap:Дизайн фирменного стиля.md",
            "roadmap:Дизайн брендбука.md",
        ]),
    },

    # --- Priority 6: Clarification rules ---
    {
        "rule_text": (
            "Для монтажа/демонтажа наружной рекламы без указания типа конструкции — "
            "ОБЯЗАТЕЛЬНО уточни: фасадная вывеска, световой короб, объёмные буквы, "
            "баннер или другой тип. Цена сильно зависит от типа конструкции."
        ),
        "direction": "Цех",
        "priority": 6,
        "source_ids": "[]",
    },
    {
        "rule_text": (
            "Для объёмных/световых букв — ОБЯЗАТЕЛЬНО уточни высоту букв. "
            "Цена за букву 20 см и 80 см отличается в 3-5 раз. "
            "Не выдавай единую цену без знания размера."
        ),
        "direction": "Цех",
        "priority": 6,
        "source_ids": "[]",
    },
    {
        "rule_text": (
            "Для монтажа, демонтажа, выезда, доставки — НЕ добавляй в смету "
            "позиции с выездом/доставкой/логистикой, пока клиент НЕ указал город "
            "и конкретный адрес. Не считай по дефолтному городу (Махачкала/Каспийск). "
            "Сначала уточни: «В каком городе/адресе нужен монтаж?» и только потом "
            "формируй позиции выезда. Позиции вида «Выезд 20 км» без подтверждённой "
            "локации — это галлюцинация системы."
        ),
        "direction": "Цех",
        "priority": 6,
        "source_ids": json.dumps(["fb#77", "fb#83"]),
    },
]


def seed_rules(db_path: str, dry_run: bool = False):
    """Insert curated rules into feedback_rules table."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check existing rules
    existing = conn.execute(
        "SELECT id, rule_text, priority, direction, is_active FROM feedback_rules"
    ).fetchall()

    print(f"Existing rules in DB: {len(existing)}")
    for r in existing:
        status = "active" if r["is_active"] else "inactive"
        print(f"  [{status}] p={r['priority']} dir='{r['direction']}': {r['rule_text'][:80]}...")

    if existing:
        print(f"\nWARNING: {len(existing)} rules already exist.")
        if not dry_run:
            resp = input("Deactivate existing and insert new? [y/N]: ").strip().lower()
            if resp != "y":
                print("Aborted.")
                conn.close()
                return

            # Deactivate old rules
            conn.execute("UPDATE feedback_rules SET is_active = 0")
            print(f"Deactivated {len(existing)} old rules.")

    # Insert new rules
    inserted = 0
    for rule in RULES:
        if dry_run:
            print(f"  [DRY] p={rule['priority']} dir='{rule['direction']}': {rule['rule_text'][:80]}...")
        else:
            conn.execute(
                """INSERT INTO feedback_rules (rule_text, direction, priority, source_ids, is_active)
                   VALUES (?, ?, ?, ?, 1)""",
                (rule["rule_text"], rule["direction"], rule["priority"], rule["source_ids"]),
            )
        inserted += 1

    if not dry_run:
        conn.commit()

    # Verify
    if not dry_run:
        active = conn.execute(
            "SELECT COUNT(*) as cnt FROM feedback_rules WHERE is_active = 1"
        ).fetchone()["cnt"]
        print(f"\nInserted {inserted} rules. Active rules in DB: {active}")

        # Show what will be injected for each direction
        for direction in ["", "Дизайн", "Цех"]:
            label = direction or "(общие)"
            if direction:
                rows = conn.execute("""
                    SELECT rule_text, priority FROM feedback_rules
                    WHERE is_active = 1 AND (direction = '' OR direction = ?)
                    ORDER BY priority DESC
                """, (direction,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT rule_text, priority FROM feedback_rules
                    WHERE is_active = 1
                    ORDER BY priority DESC
                """).fetchall()
            print(f"\nDirection '{label}' → {len(rows)} rules:")
            for r in rows:
                print(f"  p={r['priority']}: {r['rule_text'][:90]}...")
    else:
        print(f"\n[DRY RUN] Would insert {inserted} rules.")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed feedback_rules into RAG DB")
    parser.add_argument("--db", required=True, help="Path to labus_rag.db")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: DB not found: {args.db}")
        sys.exit(1)

    seed_rules(args.db, dry_run=args.dry_run)
