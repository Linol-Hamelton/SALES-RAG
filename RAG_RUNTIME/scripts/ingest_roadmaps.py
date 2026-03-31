#!/usr/bin/env python3
"""
Ingest ROADMAP markdown files into RAG-compatible JSONL.

Reads all .md files from RAG_DATA/ROADMAPS/, chunks by ## headings (stages),
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


def slugify(text: str) -> str:
    """Create a short ASCII slug from Russian text for doc_id."""
    # transliterate common chars
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


def chunk_roadmap(text: str, filename: str) -> list[dict]:
    """Split roadmap markdown by ## headings into chunks.

    Returns list of dicts with keys: title (roadmap name), section, body.
    """
    lines = text.split("\n")

    # Extract roadmap title from H1
    roadmap_title = filename.replace(".md", "")
    for line in lines:
        m = re.match(r"^#\s+\**(.+?)\**\s*$", line)
        if m:
            raw_title = m.group(1).strip()
            # Remove "Дорожная карта услуги: " prefix if present
            roadmap_title = re.sub(
                r"^Дорожная карта услуги:\s*", "", raw_title
            ).strip()
            break

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
        # Match ## headings (with optional bold markers)
        m = re.match(r"^##\s+\**(.+?)\**\s*$", line)
        if m:
            flush()
            current_section = m.group(1).strip()
            # Clean backslash-escaped dots: "Этап 1\." → "Этап 1."
            current_section = current_section.replace("\\.", ".")
            continue

        # Skip H1 and metadata lines before first section
        if re.match(r"^#\s+", line):
            continue
        if not current_section and re.match(r"^\*\s+\*\*(Путь|Категория|Услуга|Тип документа):", line):
            continue

        current_lines.append(line)

    flush()

    # If no ## headings found, treat entire content as one chunk
    if not chunks:
        body = text.strip()
        if len(body) >= 50:
            chunks.append({
                "title": roadmap_title,
                "section": "Полный документ",
                "body": body,
            })

    return chunks


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
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        slug = slugify(md_path.stem)
        chunks = chunk_roadmap(text, md_path.name)

        for idx, chunk in enumerate(chunks):
            doc_id = f"roadmap_{slug}_{idx:04d}"
            header = f"[Дорожная карта: {chunk['title']} → {chunk['section']}]"
            searchable = f"{header}\n\n{chunk['body']}"

            doc = {
                "doc_id": doc_id,
                "doc_type": "roadmap",
                "searchable_text": searchable,
                "metadata": {
                    "source": "roadmap",
                    "source_file": md_path.name,
                    "roadmap_title": chunk["title"],
                    "section": chunk["section"],
                },
                "provenance": {
                    "source": str(md_path),
                    "generated_at": GENERATED_AT,
                },
            }
            docs.append(doc)

        if verbose:
            print(f"  {md_path.name[:60]:60s}: {len(chunks)} chunks")

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nWritten: {OUTPUT_PATH} ({len(docs)} docs from {len(md_files)} files)")


if __name__ == "__main__":
    main()
