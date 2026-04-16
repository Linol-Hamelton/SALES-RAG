#!/usr/bin/env python3
"""
Knowledge base ingestion: FAQs, Markdown guides, DOCX manuals, PDF strategy docs.

Writes two new JSONL files:
  data/faq_docs.jsonl        — Q&A pairs from labus_faqs_final.json
  data/knowledge_docs.jsonl  — Chunked sections from MD, DOCX, PDF

Usage:
    python scripts/ingest_knowledge.py [--verbose]
"""
import json
import re
import sys
import click
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAG_DATA = PROJECT_ROOT.parent / "RAG_DATA"
OUTPUT_DIR = PROJECT_ROOT / "data"

GENERATED_AT = datetime.now(timezone.utc).isoformat()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_answer(text: str) -> str:
    """Normalize FAQ answer text."""
    text = text.replace("\\n", "\n").replace("\\'", "'")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("Связаться для заказа.", "").strip()
    return text


def chunk_markdown(text: str, source_file: str, min_chunk: int = 80) -> list[dict]:
    """Split markdown by headings or bold-text section markers into chunks.

    Handles:
      # H1, ## H2, ### H3, #### H4 — standard markdown
      **Bold line** — used in Labus MD files as section titles
      Numbered items (1\. text) — treated as sub-sections
    Falls back to paragraph chunking if no headings detected.
    """
    chunks = []
    current_section = ""
    current_content: list[str] = []
    parent_section = ""

    def flush():
        nonlocal current_section, current_content
        if current_section and current_content:
            body = "\n".join(current_content).strip()
            if len(body) >= min_chunk:
                chunks.append({"section": current_section, "parent": parent_section, "body": body})
        current_content = []

    for line in text.split("\n"):
        # Standard markdown headings
        m = re.match(r"^(#{1,4})\s+(.+)", line)
        if m:
            flush()
            title = m.group(2).strip().strip("*").strip()
            if len(m.group(1)) == 1:
                parent_section = title
            else:
                current_section = title
            continue

        # **Bold text** as heading (entire line is bold)
        m_bold = re.match(r"^\*\*([^*]{5,})\*\*\s*(?:\\\\)?$", line)
        if m_bold:
            flush()
            current_section = m_bold.group(1).strip()
            continue

        current_content.append(line)

    flush()

    # Fallback: paragraph-based chunking if nothing was found
    if not chunks:
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) >= min_chunk]
        current = []
        current_len = 0
        section_idx = 0
        for para in paras:
            if current_len + len(para) > 700 and current:
                chunks.append({
                    "section": f"Раздел {section_idx + 1}",
                    "parent": source_file,
                    "body": "\n\n".join(current),
                })
                section_idx += 1
                current = [para]
                current_len = len(para)
            else:
                current.append(para)
                current_len += len(para)
        if current:
            chunks.append({
                "section": f"Раздел {section_idx + 1}",
                "parent": source_file,
                "body": "\n\n".join(current),
            })

    return chunks


def read_docx(path: Path) -> str:
    """Extract plain text from DOCX.
    Uses double newline between paragraphs so the chunker can split them properly.
    """
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"  [WARN] Could not read {path.name}: {e}")
        return ""


def read_txt(path: Path) -> str:
    """Read plain text file (UTF-8 with BOM fallback to cp1251)."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1251")


def read_pdf(path: Path) -> str:
    """Extract plain text from PDF."""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        print(f"  [WARN] Could not read {path.name}: {e}")
        return ""


# ---------------------------------------------------------------------------
# FAQ ingestion
# ---------------------------------------------------------------------------

def ingest_faqs(verbose: bool) -> list[dict]:
    """Ingest labus_faqs_final.json → faq doc per Q&A pair."""
    path = RAG_DATA / "labus_faqs_final.json"
    if not path.exists():
        print(f"[WARN] FAQ JSON not found: {path}")
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for i, item in enumerate(data):
        question = item.get("question", "").strip()
        answer = clean_answer(item.get("answer", ""))
        title = item.get("title", "")
        url = item.get("url", "")

        if not question or not answer or len(answer) < 20:
            continue

        searchable = f"Вопрос: {question}\n\nОтвет: {answer}"

        doc = {
            "doc_id": f"faq_{i:04d}",
            "doc_type": "faq",
            "searchable_text": searchable,
            "payload": {
                "doc_id": f"faq_{i:04d}",
                "doc_type": "faq",
                "question": question,
                "answer": answer,
                "category": title,
                "url": url,
                "searchable_text": searchable,
            },
            "provenance": {"source": str(path), "generated_at": GENERATED_AT},
        }
        docs.append(doc)

    if verbose:
        print(f"  FAQ docs: {len(docs)}")
    return docs


# ---------------------------------------------------------------------------
# Knowledge base ingestion (MD, DOCX, PDF)
# ---------------------------------------------------------------------------

# P12.3.C: macro flag classifies sources whose primary function is a
# manager-script (бриф, возражения, скрипт переговоров), not a data/fact
# reference. These get force-retrieved when the intent-detector fires.
KNOWLEDGE_SOURCES = [
    # (filename, source_label, doc_kind, is_macro, macro_type)
    ("Консультирование по продвижению бизнеса и рекламе для менеджеров рекламной компании.md",
     "consulting_guide", "Консультирование по продвижению бизнеса", True, "consulting"),
    ("Часто задаваемые вопросы.md",
     "business_faq", "Часто задаваемые вопросы", False, ""),
    ("Продажа дизайна и сопровождение сделки.md",
     "design_sales_guide", "Продажа дизайна", True, "sales_script"),
    ("Использование приемов убеждения.md",
     "persuasion_guide", "Приёмы убеждения", True, "persuasion"),
    ("Брифы.md",
     "briefs_guide", "Брифы по продуктам", True, "brief"),
    ("warranty-card (1).docx",
     "warranty", "Гарантийная карта", False, ""),
    # NOTE: ROADMAPS/* файлы индексируются отдельно через ingest_roadmaps.py
    # (doc_type=roadmap, cross-link linked_product_ids/linked_smeta_category_ids).
    # Дублирование сюда удалено в P10.6 B2.
]


def ingest_knowledge(verbose: bool) -> list[dict]:
    """Chunk all knowledge sources into docs."""
    docs = []
    doc_counter = 0

    for filename, source_label, display_label, is_macro, macro_type in KNOWLEDGE_SOURCES:
        path = RAG_DATA / filename
        if not path.exists():
            print(f"  [SKIP] Not found: {filename}")
            continue

        # Read content
        suffix = path.suffix.lower()
        if suffix == ".md":
            text = path.read_text(encoding="utf-8")
        elif suffix == ".docx":
            text = read_docx(path)
        elif suffix == ".pdf":
            text = read_pdf(path)
        elif suffix == ".txt":
            text = read_txt(path)
        else:
            print(f"  [SKIP] Unsupported format: {filename}")
            continue

        if not text.strip():
            print(f"  [SKIP] Empty content: {filename}")
            continue

        # Chunk — markdown uses heading-aware chunker; all others use paragraph chunker
        MAX_CHUNK = 1000  # chars per chunk (larger = better semantic coherence)
        if suffix == ".md":
            chunks = chunk_markdown(text, filename)
        else:
            chunks = []
            # For .txt: lines may have single \n — split per-line, then group
            # For .docx/.pdf: paragraphs are already \n\n-separated after read_docx fix
            if suffix == ".txt":
                raw_units = [l.strip() for l in text.split("\n") if len(l.strip()) > 10]
            else:
                raw_units = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 40]

            current: list[str] = []
            current_len = 0
            section_idx = 0
            OVERLAP_UNITS = 2  # overlap last N units from previous chunk
            prev_tail: list[str] = []
            for unit in raw_units:
                if current_len + len(unit) > MAX_CHUNK and current:
                    body = "\n".join(current) if suffix == ".txt" else "\n\n".join(current)
                    chunks.append({
                        "section": f"{display_label} — часть {section_idx + 1}",
                        "parent": display_label,
                        "body": body,
                    })
                    section_idx += 1
                    # Keep last N units as overlap
                    prev_tail = current[-OVERLAP_UNITS:] if len(current) >= OVERLAP_UNITS else current[:]
                    current = prev_tail + [unit]
                    current_len = sum(len(u) for u in current)
                else:
                    current.append(unit)
                    current_len += len(unit)
            if current:
                body = "\n".join(current) if suffix == ".txt" else "\n\n".join(current)
                chunks.append({
                    "section": f"{display_label} — часть {section_idx + 1}" if section_idx > 0 else display_label,
                    "parent": display_label,
                    "body": body,
                })

        for chunk in chunks:
            section = chunk["section"]
            body = chunk["body"]
            parent = chunk.get("parent", display_label)

            # Build searchable text
            header = f"[{display_label}]" if parent == section else f"[{display_label} → {section}]"
            searchable = f"{header}\n\n{body}"

            doc_id = f"knowledge_{source_label}_{doc_counter:04d}"
            doc = {
                "doc_id": doc_id,
                "doc_type": "knowledge",
                "searchable_text": searchable,
                "payload": {
                    "doc_id": doc_id,
                    "doc_type": "knowledge",
                    "source": source_label,
                    "source_label": display_label,
                    "section": section,
                    "parent_section": parent,
                    "content": body,
                    "searchable_text": searchable,
                    # P12.3.C: manager-script routing metadata
                    "is_macro": is_macro,
                    "macro_type": macro_type,
                },
                "provenance": {"source": filename, "generated_at": GENERATED_AT},
            }
            docs.append(doc)
            doc_counter += 1

        if verbose:
            print(f"  {filename[:60]:60s}: {len(chunks)} chunks")

    return docs


# ---------------------------------------------------------------------------
# Anchor knowledge docs — hardcoded critical facts that must always be findable
# ---------------------------------------------------------------------------

ANCHOR_DOCS = [
    {
        "id": "warranty_terms",
        "source": "warranty_anchor",
        "source_label": "Гарантийная карта Labus.pro",
        "section": "Гарантия на рекламные конструкции",
        "is_macro": False,
        "macro_type": "",
        "body": (
            "Гарантия на рекламные вывески Labus.pro: 12 месяцев с момента приёмки.\n\n"
            "Что покрывается гарантией: световые короба, объёмные буквы, сборные элементы наружной рекламы — 12 месяцев.\n\n"
            "Что НЕ покрывается гарантией (расходные материалы): лампы, дросселя, фотореле, стабилизаторы и выпрямители напряжения.\n\n"
            "Гарантия недействительна при: нарушении правил эксплуатации или питающего напряжения; "
            "самостоятельной переделке или регулировке конструкции; ремонте/чистке не специалистами Labus.pro; "
            "демонтаже не специалистами Labus.pro; попадании инородных предметов или агрессивных веществ; "
            "воздействии стихийных бедствий (град, ураган, молния, наводнение, землетрясение).\n\n"
            "Для получения гарантийного обслуживания: предъявить дефектное изделие вместе с оригиналом гарантийного документа. "
            "Если конструкция не может быть предъявлена — оформляется платная заявка на выезд диагностической бригады."
        ),
    },
    {
        "id": "brief_merch",
        "source": "briefs_anchor",
        "source_label": "Брифы по продуктам — Мерч",
        "section": "Что нужно согласовать для мерча (кружки, ручки, футболки и др.)",
        "is_macro": True,
        "macro_type": "brief",
        "body": (
            "Для производства мерча необходимо согласовать с клиентом следующее (бриф):\n\n"
            "РУЧКИ: модель ручки, цвет корпуса, количество, способ нанесения логотипа (тампопечать/гравировка/шелкография), "
            "площадь нанесения, срок производства.\n\n"
            "КРУЖКИ: модель кружки, цвет, объём (250/300/350 мл), количество, способ нанесения (сублимация/УФ-печать), "
            "площадь нанесения, срок.\n\n"
            "ФУТБОЛКИ: модель/фасон, размерная сетка, цвет ткани, количество по размерам, способ нанесения "
            "(шелкография/термотрансфер/вышивка), место нанесения (грудь/спина/рукав), срок.\n\n"
            "ШОППЕРЫ: материал (спанбонд/нетканый/х/б), размер, цвет, количество, способ нанесения, срок.\n\n"
            "ПОВЕРБАНКИ: модель, ёмкость (mAh), цвет, количество, способ нанесения логотипа, срок.\n\n"
            "Для всего мерча также нужен дизайн-бриф: логотип (файл AI/EPS/PDF), шрифты, название организации, "
            "сфера деятельности, контакты для нанесения, референсы (примеры понравившихся работ)."
        ),
    },
    {
        "id": "brief_print",
        "source": "briefs_anchor",
        "source_label": "Брифы по продуктам — Полиграфия",
        "section": "Что нужно согласовать для полиграфии (визитки, листовки, буклеты)",
        "is_macro": True,
        "macro_type": "brief",
        "body": (
            "Для производства полиграфии необходимо согласовать с клиентом (бриф):\n\n"
            "ВИЗИТКИ: размер (стандарт 90×50 мм / евро 85×55 мм), тираж, бумага (мелованная 300/350 г, "
            "Touch Cover, дизайнерская), ламинация (матовая/глянцевая/без), вид печати (цифровая/офсет), "
            "количество цветов, двусторонняя или нет, срок.\n\n"
            "ЛИСТОВКИ / ФЛАЕРЫ: формат (А3/А4/А5/А6), тираж, бумага (80-130 г/м²), ламинация, "
            "один или два цвета, двусторонняя, срок.\n\n"
            "БУКЛЕТЫ: формат развёртки (А3 или А4), количество фальцев, тираж, бумага, ламинация, срок.\n\n"
            "НАКЛЕЙКИ / САМОКЛЕЙКА: размер, форма (прямоугольник/круг/фигурная вырубка), тираж, "
            "материал (белая/прозрачная/матовая), срок.\n\n"
            "Для дизайна: логотип (AI/EPS/PDF), цвета бренда, текст, контакты, адрес, QR-код (если нужен), "
            "референсы стиля."
        ),
    },
    {
        "id": "brief_signage",
        "source": "briefs_anchor",
        "source_label": "Брифы по продуктам — Вывески и наружная реклама",
        "section": "Что нужно согласовать для вывески и наружной рекламы",
        "is_macro": True,
        "macro_type": "brief",
        "body": (
            "Для производства и монтажа вывесок / наружной рекламы необходимо согласовать (бриф):\n\n"
            "СВЕТОВЫЕ КОРОБА / ЛАЙТБОКСЫ: размер (ШxВ в мм), толщина короба, вид подсветки (LED/неон), "
            "одностороннее/двустороннее, материал лицевой части (акрил/ПВХ), цвет фона, тираж, срок.\n\n"
            "ОБЪЁМНЫЕ БУКВЫ: высота букв (мм), материал (акрил/металл/ПВХ), способ подсветки "
            "(торцевая LED/обратная/без подсветки), цвет, количество букв/символов, текст, монтаж (да/нет), срок.\n\n"
            "БАННЕРЫ: размер (ШxВ в мм или мп), материал (баннерная ткань/лакша/сетка), "
            "способ крепления (карманы/люверсы/планка), тираж, срок.\n\n"
            "МОНТАЖ: адрес объекта, тип фасада (кирпич/сэндвич/стекло/профнастил), "
            "высота установки (нужна ли автовышка), фото фасада, условия доступа.\n\n"
            "Для дизайна вывески: логотип (AI/EPS), фирменные цвета (Pantone/CMYK/HEX), "
            "текст для вывески, фото фасада здания, примеры понравившихся вывесок."
        ),
    },
    {
        "id": "roi_key_facts",
        "source": "roi_anchor",
        "source_label": "Конверсия и ROI labus.pro — ключевые показатели",
        "section": "ROMI и конверсия по направлениям Labus.pro",
        "is_macro": False,
        "macro_type": "",
        "body": (
            "Ключевые показатели эффективности рекламных услуг Labus.pro (2024-2026):\n\n"
            "БРЕНДИНГ И АЙДЕНТИКА (ROMI):\n"
            "  Разработка логотипа: мин. 100%, средний 180%, оптимальный 350%, идеальный >600%\n"
            "  Фирменный стиль (комплекс): мин. 120%, средний 220%, оптимальный 400%, идеальный >750%\n"
            "  Разработка брендбука: мин. 150%, средний 250%, оптимальный 450%, идеальный >800%\n"
            "  Корпоративные иллюстрации: мин. 80%, средний 140%, оптимальный 280%, идеальный >500%\n\n"
            "НАРУЖНАЯ РЕКЛАМА (конверсия в трафик):\n"
            "  Брандирование транспорта: 30 000–70 000 просмотров/сутки, конверсия в лид 0.5–1.5%\n"
            "  Билборды 3×6 м: охват 15 000–25 000 просмотров/сутки, конверсия 0.3–1.0%\n"
            "  Световые вывески: CR 1–3% (прямой заход в заведение), ROMI 200–500%\n\n"
            "ПОЛИГРАФИЯ (конверсия листовок):\n"
            "  Листовки А5 (холодная рассылка): CR 0.5–2%, ROMI 100–300%\n"
            "  Листовки А4 (таргетированная): CR 2–5%, ROMI 200–500%\n"
            "  Буклеты и каталоги: CR 3–8% в повторный контакт, ROMI 150–400%\n\n"
            "МЕРЧ (корпоративные подарки):\n"
            "  Брендированный мерч: ROMI 120–350%, срок окупаемости 6–12 месяцев\n\n"
            "ДИЗАЙН:\n"
            "  Дизайн упаковки: ROMI 150–1000%, влияет на конверсию продаж +15–40%\n"
            "  3D-визуализация: ROMI 100–700%\n"
            "  Фотосъёмка: влияние на конверсию сайта +5–45%\n"
            "  SMM-визуал: Engagement Rate 1.5–10%"
        ),
    },
    {
        "id": "objection_handling",
        "source": "sales_anchor",
        "source_label": "Работа с возражениями клиентов",
        "section": "Скрипты ответов на ключевые возражения клиентов",
        "is_macro": True,
        "macro_type": "objection",
        "body": (
            "Ключевые возражения клиентов и как на них отвечать:\n\n"
            "ВОЗРАЖЕНИЕ 'СЛИШКОМ ДОРОГО':\n"
            "  — Разбейте стоимость на составляющие: 'Давайте посмотрим, из чего складывается цена...'\n"
            "  — Сравните с убытками от плохой рекламы: 'Некачественная вывеска отпугнёт в 3 раза больше клиентов...'\n"
            "  — Предложите рассрочку или поэтапную оплату\n"
            "  — Покажите ROI: 'Вывеска работает 5 лет, значит в день это стоит...'\n\n"
            "ВОЗРАЖЕНИЕ 'Я ПОДУМАЮ':\n"
            "  — 'Хорошо, а что именно вас останавливает?' (выяснить реальное возражение)\n"
            "  — Предложить дополнительную информацию: 'Давайте я пришлю примеры наших работ'\n"
            "  — Установить срок: 'Акция действует до пятницы — успеем зафиксировать цену?'\n\n"
            "ВОЗРАЖЕНИЕ 'У КОНКУРЕНТОВ ДЕШЕВЛЕ':\n"
            "  — 'Уточните у них: из чего сделано? Какой срок гарантии? Кто будет монтировать?'\n"
            "  — Объяснить разницу в качестве материалов и сервисе\n"
            "  — Предложить сравнить конкретные характеристики\n\n"
            "ВОЗРАЖЕНИЕ 'ЗАЧЕМ ДИЗАЙН — У НАС УЖЕ ЕСТЬ ЛОГОТИП':\n"
            "  — 'Нам нужно адаптировать его под конкретный носитель, иначе могут быть искажения цвета и пропорций'\n"
            "  — Объяснить разницу между логотипом для экрана и для печати/вывески\n\n"
            "ВОЗРАЖЕНИЕ 'СДЕЛАЙТЕ СНАЧАЛА — ПОТОМ ЗАПЛАЧУ':\n"
            "  — 'Наш стандарт — 50% предоплата. Это защищает обе стороны: мы гарантируем срок, вы — приоритет в очереди.'"
        ),
    },
]


def build_anchor_docs() -> list[dict]:
    """Build hardcoded critical-fact knowledge docs that must always be retrievable."""
    docs = []
    for anchor in ANCHOR_DOCS:
        doc_id = f"knowledge_anchor_{anchor['id']}"
        searchable = f"[{anchor['source_label']} → {anchor['section']}]\n\n{anchor['body']}"
        docs.append({
            "doc_id": doc_id,
            "doc_type": "knowledge",
            "searchable_text": searchable,
            "payload": {
                "doc_id": doc_id,
                "doc_type": "knowledge",
                "source": anchor["source"],
                "source_label": anchor["source_label"],
                "section": anchor["section"],
                "parent_section": anchor["source_label"],
                "content": anchor["body"],
                "searchable_text": searchable,
                # P12.3.C: manager-script routing metadata
                "is_macro": anchor.get("is_macro", False),
                "macro_type": anchor.get("macro_type", ""),
            },
            "provenance": {"source": "anchor_knowledge", "generated_at": GENERATED_AT},
        })
    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--verbose", is_flag=True, default=False)
def main(verbose: bool):
    """Ingest knowledge base docs into JSONL files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Ingesting FAQs ===")
    faq_docs = ingest_faqs(verbose)
    faq_path = OUTPUT_DIR / "faq_docs.jsonl"
    with open(faq_path, "w", encoding="utf-8") as f:
        for doc in faq_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Written: {faq_path} ({len(faq_docs)} docs)")

    print("\n=== Ingesting Knowledge Base ===")
    knowledge_docs = ingest_knowledge(verbose)

    print("\n=== Building Anchor Docs (critical facts) ===")
    anchor_docs = build_anchor_docs()
    print(f"  Anchor docs: {len(anchor_docs)}")
    all_knowledge = knowledge_docs + anchor_docs

    kb_path = OUTPUT_DIR / "knowledge_docs.jsonl"
    with open(kb_path, "w", encoding="utf-8") as f:
        for doc in all_knowledge:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  Written: {kb_path} ({len(all_knowledge)} docs)")

    total = len(faq_docs) + len(all_knowledge)
    print(f"\nTotal new knowledge docs: {total}")


if __name__ == "__main__":
    main()
