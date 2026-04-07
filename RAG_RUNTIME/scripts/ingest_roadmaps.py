#!/usr/bin/env python3
"""
Ingest ROADMAP markdown files into RAG-compatible JSONL.

Reads all .md files from RAG_DATA/ROADMAPS/, chunks by ## headings (stages),
extracts structured metadata (direction, category, prices, timelines, ROI),
and writes data/roadmap_docs.jsonl.

Usage:
    python scripts/ingest_roadmaps.py [--verbose]
"""
import json
import re
import click
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAG_DATA = PROJECT_ROOT.parent / "RAG_DATA"
ROADMAPS_DIR = RAG_DATA / "ROADMAPS"
OUTPUT_PATH = PROJECT_ROOT / "data" / "roadmap_docs.jsonl"

GENERATED_AT = datetime.now(timezone.utc).isoformat()

# ── Direction mapping: category/title keywords → canonical direction ─────────

DIRECTION_MAP = {
    # Цех — signs, outdoor
    "наружная реклама": "Цех",
    "вывеск": "Цех",
    "световых коробов": "Цех",
    "световых панелей": "Цех",
    "объемных": "Цех",
    "плоских букв": "Цех",
    "неонов": "Цех",
    "фасадн": "Цех",
    "табличк": "Цех",
    "кронштейн": "Цех",
    "штендер": "Цех",
    "пресс-волл": "Цех",
    "press-wall": "Цех",
    "оклейка транспорта": "Цех",
    "брендирование авто": "Цех",
    # Сольвент — wide format
    "широкоформатн": "Сольвент",
    "интерьерн": "Сольвент",
    "баннер": "Сольвент",
    "печать на холсте": "Сольвент",
    "виниловых пленок": "Сольвент",
    # Печатная — standard print
    "полиграфи": "Печатная",
    "типографи": "Печатная",
    "визитк": "Печатная",
    "листовк": "Печатная",
    "флаер": "Печатная",
    "буклет": "Печатная",
    "брошюр": "Печатная",
    "каталог": "Печатная",
    "этикет": "Печатная",
    "стикер": "Печатная",
    "3d стикер": "Печатная",
    "3d-стикер": "Печатная",
    "бланк": "Печатная",
    "календар": "Печатная",
    "меню": "Печатная",
    "книгоиздан": "Печатная",
    "офсетн": "Печатная",
    "цифровая печать": "Печатная",
    "пластиков": "Печатная",
    "бейдж": "Печатная",
    # POS — стенды, навигация, информационные
    "информационн": "POS",
    "напольн": "POS",
    "стенд": "POS",
    # Дизайн — design services
    "дизайн": "Дизайн",
    "логотип": "Дизайн",
    "брендбук": "Дизайн",
    "фирменного стиля": "Дизайн",
    "иллюстрац": "Дизайн",
    "web-дизайн": "Дизайн",
    "web-разработк": "Дизайн",
    "smm": "Дизайн",
    "контент-план": "Дизайн",
    "media production": "Дизайн",
    "фотосъемк": "Дизайн",
    "упаковк": "Дизайн",
    # Мерч — merchandise
    "мерч": "Мерч",
    "сувенир": "Мерч",
    "брендированн": "Мерч",
    "бейсболк": "Мерч",
    "бутылк": "Мерч",
    "магнит": "Мерч",
    "повербанк": "Мерч",
    "ручк": "Мерч",
    "свитшот": "Мерч",
    "термос": "Мерч",
    "термокружк": "Мерч",
    "шоппер": "Мерч",
    "кружк": "Мерч",
    "блокнот": "Мерч",
    "ежедневник": "Мерч",
    "папк": "Мерч",
    "шоколад": "Мерч",
    "скотч": "Мерч",
    "пакет": "Мерч",
    "крафт-пакет": "Мерч",
}


def slugify(text: str) -> str:
    """Create a short ASCII slug from Russian text for doc_id."""
    tr = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
        "ж": "zh", "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
    result = []
    for ch in text.lower():
        if ch in tr:
            result.append(tr[ch])
        elif ch.isascii() and ch.isalnum():
            result.append(ch)
        elif ch in (" ", "-", "_"):
            result.append("_")
    slug = re.sub(r"_+", "_", "".join(result)).strip("_")
    return slug[:60]


def detect_direction(title: str, category: str) -> str:
    """Map roadmap title + category to canonical direction."""
    combined = f"{title} {category}".lower()
    for keyword, direction in DIRECTION_MAP.items():
        if keyword in combined:
            return direction
    return ""


def clean_markdown(text: str) -> str:
    """Strip markdown formatting for cleaner embeddings."""
    # Remove bold markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # Remove italic markers
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # Remove bullet markers at line start
    text = re.sub(r"^\s*\*\s+", "", text, flags=re.MULTILINE)
    # Remove escaped dots
    text = text.replace("\\.", ".")
    # Remove escaped underscores
    text = text.replace("\\_", "_")
    # Remove reference markers like [1], [2], [3]
    text = re.sub(r"\s*\[[\d,\s]+\]", "", text)
    # Remove backticks
    text = text.replace("`", "")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_prices(text: str) -> list[dict]:
    """Extract price ranges/values from text."""
    prices = []
    # Pattern: "27 500 – 48 500 руб." or "от 5 000 рублей" or "5 000 руб"
    # Match package-style: «Название» (XX XXX – YY YYY руб.)
    pkg_pattern = r"[«\"«]([^»\"»]+)[»\"»]\s*\(?\s*([\d\s]+(?:\s*[-–]\s*[\d\s]+)?)\s*руб"
    for m in re.finditer(pkg_pattern, text, re.IGNORECASE):
        name = m.group(1).strip()
        price_str = m.group(2).strip()
        parts = re.split(r"\s*[-–]\s*", price_str)
        nums = [int(p.replace(" ", "").replace("\u00a0", "")) for p in parts if p.replace(" ", "").replace("\u00a0", "").isdigit()]
        if nums:
            prices.append({"package": name, "min": nums[0], "max": nums[-1]})

    # Standalone price patterns: "от X руб" or "X руб" or "X–Y руб"
    if not prices:
        standalone = re.findall(r"([\d][\d\s]*[\d])\s*(?:[-–]\s*([\d][\d\s]*[\d])\s*)?руб", text)
        for match in standalone:
            lo = int(match[0].replace(" ", "").replace("\u00a0", ""))
            hi = int(match[1].replace(" ", "").replace("\u00a0", "")) if match[1] else lo
            if lo >= 100:  # filter noise like "2 руб"
                prices.append({"package": "", "min": lo, "max": hi})

    return prices


def extract_timelines(text: str) -> list[str]:
    """Extract production timeline mentions."""
    timelines = []
    patterns = [
        r"(\d+\s*[-–]\s*\d+\s*(?:рабочих\s+)?(?:дней|дня|день))",
        r"(от\s+\d+\s+до\s+\d+\s+(?:рабочих\s+)?(?:дней|дня|день))",
        r"(\d+\s*[-–]\s*\d+\s*(?:рабочих\s+)?(?:недел[ьи]))",
        r"(\d+\s+час(?:а|ов)?)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val = m.group(1).strip()
            if val not in timelines:
                timelines.append(val)
    return timelines


def extract_roi(text: str) -> dict | None:
    """Extract ROI/ROMI figures from text."""
    result = {}

    # ROMI percentage — various formats
    romi_vals = re.findall(
        r"ROMI?\s*(?:[\(\)]?\s*возврат[^)]*\)?\s*)?(?:от\s+|~|≈|около\s+|составляет\s+|более\s+чем\s+)?(\d+)\s*%",
        text, re.IGNORECASE
    )
    if romi_vals:
        nums = [int(v) for v in romi_vals]
        result["romi_min"] = min(nums)
        result["romi_max"] = max(nums)

    # ROI percentage
    roi_vals = re.findall(
        r"ROI\s*(?:от\s+|~|≈|около\s+)?(\d+)\s*%",
        text, re.IGNORECASE
    )
    if roi_vals:
        nums = [int(v) for v in roi_vals]
        result["roi_min"] = min(nums)
        result["roi_max"] = max(nums)

    # Conversion rate
    conv_vals = re.findall(
        r"(?:конверси[яюей]|CR)\s*(?:[\w\s]*?)(?:от\s+|до\s+|~|≈|около\s+)?(\d+(?:[.,]\d+)?)\s*%",
        text, re.IGNORECASE
    )
    if conv_vals:
        nums = [float(v.replace(",", ".")) for v in conv_vals]
        result["conversion_min"] = min(nums)
        result["conversion_max"] = max(nums)

    return result if result else None


def extract_doc_metadata(lines: list[str]) -> dict:
    """Extract metadata block from top of roadmap (Путь, Категория, Услуга, Тип)."""
    meta = {}
    for line in lines[:15]:
        m = re.match(r"^\*\s*\*\*Категория:\*\*\s*(.+?)(?:\s*\\?\s*)?$", line)
        if m:
            meta["category"] = m.group(1).strip().rstrip("\\")
            continue
        m = re.match(r"^\*\s*\*\*Услуга:\*\*\s*(.+?)(?:\s*\\?\s*)?$", line)
        if m:
            meta["service"] = m.group(1).strip().rstrip("\\")
            continue
    return meta


def chunk_roadmap(text: str, filename: str) -> tuple[list[dict], dict]:
    """Split roadmap markdown by ## headings into chunks.

    Returns (chunks, file_metadata) where file_metadata has category, service, etc.
    """
    lines = text.split("\n")

    # Extract roadmap title from H1
    roadmap_title = filename.replace(".md", "")
    for line in lines:
        m = re.match(r"^#\s+\**(.+?)\**\s*$", line)
        if m:
            raw_title = m.group(1).strip()
            roadmap_title = re.sub(
                r"^Дорожная карта услуги:\s*", "", raw_title
            ).strip()
            break

    # Extract document-level metadata
    doc_meta = extract_doc_metadata(lines)

    chunks = []
    current_section = ""
    current_lines: list[str] = []

    def flush():
        nonlocal current_section, current_lines
        if current_section and current_lines:
            body = "\n".join(current_lines).strip()
            if len(body) >= 50:
                chunks.append({
                    "title": roadmap_title,
                    "section": current_section,
                    "body": body,
                })
        current_lines = []

    for line in lines:
        m = re.match(r"^##\s+\**(.+?)\**\s*$", line)
        if m:
            flush()
            current_section = m.group(1).strip()
            current_section = current_section.replace("\\.", ".")
            continue

        if re.match(r"^#\s+", line):
            continue
        if not current_section and re.match(r"^\*\s+\*\*(Путь|Категория|Услуга|Тип документа):", line):
            continue

        current_lines.append(line)

    flush()

    if not chunks:
        body = text.strip()
        if len(body) >= 50:
            chunks.append({
                "title": roadmap_title,
                "section": "Полный документ",
                "body": body,
            })

    return chunks, {**doc_meta, "roadmap_title": roadmap_title}


@click.command()
@click.option("--verbose", is_flag=True, default=False)
def main(verbose: bool):
    """Ingest ROADMAP markdown files into roadmap_docs.jsonl."""
    if not ROADMAPS_DIR.exists():
        print(f"ERROR: ROADMAPS directory not found: {ROADMAPS_DIR}")
        return

    md_files = sorted(ROADMAPS_DIR.glob("*.md"))
    print(f"Found {len(md_files)} roadmap files in {ROADMAPS_DIR}")

    docs = []
    stats = {"total_chunks": 0, "with_prices": 0, "with_timelines": 0, "with_roi": 0}

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        slug = slugify(md_path.stem)
        chunks, file_meta = chunk_roadmap(text, md_path.name)

        category = file_meta.get("category", "")
        service = file_meta.get("service", "")
        roadmap_title = file_meta.get("roadmap_title", md_path.stem)
        direction = detect_direction(roadmap_title, category)

        # Extract file-level prices, timelines, ROI from full text
        file_prices = extract_prices(text)
        file_timelines = extract_timelines(text)
        file_roi = extract_roi(text)

        for idx, chunk in enumerate(chunks):
            doc_id = f"roadmap_{slug}_{idx:04d}"

            # Clean body for better embeddings
            clean_body = clean_markdown(chunk["body"])

            # Extract chunk-level structured data
            chunk_prices = extract_prices(chunk["body"])
            chunk_timelines = extract_timelines(chunk["body"])
            chunk_roi = extract_roi(chunk["body"])

            # Build searchable text: clean, no markdown noise
            header = f"[Дорожная карта: {chunk['title']} → {chunk['section']}]"
            searchable = f"{header}\n\n{clean_body}"

            # Build enriched metadata
            metadata = {
                "source": "roadmap",
                "source_file": md_path.name,
                "roadmap_title": roadmap_title,
                "section": chunk["section"],
                "direction": direction,
                "category": category,
                "service": service,
            }

            if chunk_prices:
                metadata["prices"] = chunk_prices
                stats["with_prices"] += 1
            if chunk_timelines:
                metadata["timelines"] = chunk_timelines
                stats["with_timelines"] += 1
            if chunk_roi:
                metadata["roi"] = chunk_roi
                stats["with_roi"] += 1

            # For pricing/timeline sections, add file-level data if chunk has none
            section_lower = chunk["section"].lower()
            if "ценообразован" in section_lower or "пакет" in section_lower:
                if not chunk_prices and file_prices:
                    metadata["prices"] = file_prices
                    stats["with_prices"] += 1
            if "срок" in section_lower or "производств" in section_lower or "логистик" in section_lower:
                if not chunk_timelines and file_timelines:
                    metadata["timelines"] = file_timelines
                    stats["with_timelines"] += 1
            if "презентац" in section_lower or "квалификац" in section_lower or "консультац" in section_lower:
                if not chunk_roi and file_roi:
                    metadata["roi"] = file_roi
                    stats["with_roi"] += 1

            doc = {
                "doc_id": doc_id,
                "doc_type": "roadmap",
                "searchable_text": searchable,
                "metadata": metadata,
                "provenance": {
                    "source": str(md_path),
                    "generated_at": GENERATED_AT,
                },
            }
            docs.append(doc)

        stats["total_chunks"] += len(chunks)

        if verbose:
            dir_tag = f" [{direction}]" if direction else ""
            print(f"  {md_path.name[:60]:60s}: {len(chunks)} chunks{dir_tag}")

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nWritten: {OUTPUT_PATH} ({len(docs)} docs from {len(md_files)} files)")
    print(f"  Chunks with prices: {stats['with_prices']}")
    print(f"  Chunks with timelines: {stats['with_timelines']}")
    print(f"  Chunks with ROI: {stats['with_roi']}")
    print(f"  Directions mapped: {sum(1 for d in docs if d['metadata'].get('direction'))}/{len(docs)}")


if __name__ == "__main__":
    main()
