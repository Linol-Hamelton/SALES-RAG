#!/usr/bin/env python3
"""
Fetch Site Images v2 — Product-Centric архитектура.

Стратегия:
  Фаза 1: Скачать sitemap.xml → для каждой уникальной страницы товара
           забрать ВСЕ фото (og:image + карусель) → сохранить по slug
  Фаза 2: Замапить все deal_id из deals.json на slug товара (fuzzy-match)
  Результат: ~200 папок с фото вместо 18000 дубликатов

Хранение:
  OFFERS_IMAGE/{section}/{slug}/cover.webp
  OFFERS_IMAGE/{section}/{slug}/gallery_1.webp
  OFFERS_IMAGE/{section}/{slug}/gallery_2.webp
  OFFERS_IMAGE/index.json  — маппинг products + deal_to_product

Usage:
    python scripts/fetch_site_images.py [--limit N] [--min-score 0.6] [--skip-download]
"""
import csv
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from difflib import SequenceMatcher
from urllib.parse import urlparse, unquote

import requests
import click
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
OFFERS_CSV = PROJECT_ROOT.parent / "RAG_DATA" / "offers.csv"
OFFERS_IMAGE_DIR = PROJECT_ROOT.parent / "RAG_DATA" / "OFFERS_IMAGE"
INDEX_FILE = OFFERS_IMAGE_DIR / "index.json"

SITEMAP_URL = "https://labus.pro/sitemap.xml"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
REQUEST_TIMEOUT = 25

# ─── Транслитерация ───────────────────────────────────────────────────────────
TRANSLIT_MAP = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'j', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'x', 'ц': 'cz', 'ч': 'ch', 'ш': 'sh', 'щ': 'shh',
    'ъ': '', 'ы': 'yi', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    'А': 'a', 'Б': 'b', 'В': 'v', 'Г': 'g', 'Д': 'd', 'Е': 'e', 'Ё': 'yo',
    'Ж': 'zh', 'З': 'z', 'И': 'i', 'Й': 'j', 'К': 'k', 'Л': 'l', 'М': 'm',
    'Н': 'n', 'О': 'o', 'П': 'p', 'Р': 'r', 'С': 's', 'Т': 't', 'У': 'u',
    'Ф': 'f', 'Х': 'x', 'Ц': 'cz', 'Ч': 'ch', 'Ш': 'sh', 'Щ': 'shh',
    'Ъ': '', 'Ы': 'yi', 'Ь': '', 'Э': 'e', 'Ю': 'yu', 'Я': 'ya',
}


def transliterate(text: str) -> str:
    """Convert Russian text to a URL-friendly slug (MODX-compatible)."""
    result = []
    for ch in text.lower():
        if ch in TRANSLIT_MAP:
            result.append(TRANSLIT_MAP[ch])
        elif ch.isalnum():
            result.append(ch)
        elif ch in (' ', '_', '-'):
            result.append('-')
        else:
            result.append(ch)
    slug = re.sub(r'-{2,}', '-', ''.join(result)).strip('-')
    return slug


def extract_slug(url: str) -> str:
    """Extract the last path segment (slug) from a URL."""
    path = unquote(urlparse(url).path).rstrip('/')
    return path.split('/')[-1] if path else ""


def extract_section(url: str) -> str:
    """Extract the top-level section from URL."""
    parts = urlparse(url).path.strip('/').split('/')
    return parts[0] if parts else "other"


# ─── Sitemap ──────────────────────────────────────────────────────────────────

def fetch_sitemap(url: str) -> list[str]:
    """Download sitemap.xml and return PRODUCT pages (depth >= 3).
    Level 1: /viveska/           — section (skip)
    Level 2: /viveska/bukva/     — subcategory (skip)
    Level 3: /viveska/bukva/obemnaya-svetovaya-bukva/ — PRODUCT ✓
    """
    print(f"  Downloading sitemap from {url}...")
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    all_urls = [el.text.strip() for el in root.findall('.//sm:url/sm:loc', ns) if el.text]

    # Product pages have 3+ path segments: section/subcategory/product
    EXCLUDED_SLUGS = {'about', 'cabinet', 'design', 'search', 'contacts',
                      'clients', 'accounts', 'edit', 'blog', 'news', 'faq',
                      'bilboard'}
    product_urls = []
    for u in all_urls:
        parts = urlparse(u).path.strip('/').split('/')
        if len(parts) >= 3 and not EXCLUDED_SLUGS.intersection(parts):
            product_urls.append(u)

    print(f"  Sitemap: {len(all_urls)} total, {len(product_urls)} product pages (depth>=3)")
    return product_urls


# ─── Image Extraction ─────────────────────────────────────────────────────────

def extract_page_data(page_url: str) -> dict:
    """Fetch product page and extract title + ALL images (og:image + gallery).
    Returns: {"title": str, "images": [url, ...]}
    """
    result = {"title": "", "images": []}
    try:
        resp = requests.get(page_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return result
    except Exception as e:
        print(f"  [WARN] Failed to fetch {page_url}: {e}")
        return result

    html = resp.text
    images = []

    # Extract product title from <h1> or og:title
    h1 = re.search(r'<h1[^>]*>([^<]+)</h1>', html, re.I)
    if h1:
        result["title"] = h1.group(1).strip()
    else:
        og_title = re.search(
            r'<meta\s+(?:property|name)=["\']og:title["\']\s+content=["\']([^"\']+)', html, re.I
        )
        if og_title:
            result["title"] = og_title.group(1).strip()

    BASE_URL = "https://labus.pro"

    # 1. Extract ALL image paths containing /assets/images/resources/
    #    Matches both absolute (https://labus.pro/assets/...) and relative (/assets/...)
    all_raw = re.findall(
        r'((?:https?://labus\.pro)?/assets/images/resources/\d+/\w+/[a-f0-9]+\.(?:webp|jpg|jpeg|png))',
        html, re.I
    )

    # Normalize: convert relative paths to absolute, prefer mediumwebp over smallwebp
    seen_resource_ids = set()
    for raw_url in all_raw:
        # Make absolute
        if raw_url.startswith('/'):
            url = BASE_URL + raw_url
        elif not raw_url.startswith('http'):
            url = BASE_URL + '/' + raw_url
        else:
            url = raw_url

        # Extract resource ID to deduplicate (e.g. /resources/4170/smallwebp/...)
        rid_match = re.search(r'/resources/(\d+)/', url)
        if rid_match:
            rid = rid_match.group(1)
            if rid in seen_resource_ids:
                continue
            seen_resource_ids.add(rid)

        # Prefer mediumwebp over smallwebp for better quality
        better_url = url.replace('/smallwebp/', '/mediumwebp/')
        images.append(better_url)

    # 2. Fallback: og:image if no gallery images found
    if not images:
        og = re.search(
            r'<meta\s+(?:property|name)=["\']og:image["\']\s+content=["\']([^"\']+)', html, re.I
        ) or re.search(
            r'<meta\s+content=["\']([^"\']+)["\'].*?(?:property|name)=["\']og:image', html, re.I
        )
        if og:
            images.append(og.group(1))

    # Limit to 10 most relevant images
    result["images"] = images[:10]
    return result


def download_image(image_url: str, save_path: Path) -> bool:
    """Download an image and save to disk."""
    try:
        resp = requests.get(image_url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
        if resp.status_code != 200:
            return False
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception:
        return False


# ─── Matching ─────────────────────────────────────────────────────────────────

# ─── Deal Title Normalization ─────────────────────────────────────────────────

def normalize_deal_title(title: str) -> list[str]:
    """Normalize deal title into multiple candidate slugs for matching.
    Deals often contain options like 'Визитки матовые. Оперативная полиграфия.'
    We need to strip options and try multiple variants.
    Returns list of candidate slugs, best first."""
    candidates = []

    # 1. Strip everything after first period: 'Визитки матовые. Операт...' → 'Визитки матовые'
    base = title.split('.')[0].strip()
    if base:
        candidates.append(transliterate(base))

    # 2. Strip common suffixes: sizes, quantities, paper weights
    cleaned = re.sub(
        r'\s*\d+[xх×]\d+\s*(?:см|мм)?'  # sizes like 30x40 см
        r'|\s*\d+\s*(?:гр/?м2|мкр|шт|мм|см|экз)'  # weights/quantities
        r'|\s*(?:эконом|стандарт|премиум|офсет|двухсторонн\S*|односторонн\S*|матов\S*|глянц\S*)'
        r'|\s*(?:оперативная полиграфия|бумага|пружине)',
        '', base, flags=re.I
    ).strip()
    if cleaned and cleaned != base:
        candidates.append(transliterate(cleaned))

    # 3. First 2 words only: 'Визитки матовые' → 'vizitki-matovyie'
    words = base.split()
    if len(words) >= 2:
        candidates.append(transliterate(' '.join(words[:2])))

    # 4. First word only: 'Визитки' → 'vizitki'
    if words:
        candidates.append(transliterate(words[0]))

    # 5. Full original title as fallback
    full_slug = transliterate(title)
    if full_slug not in candidates:
        candidates.append(full_slug)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def find_best_match(deal_candidates: list[str], slug_index: dict[str, str],
                    min_score: float = 0.55) -> tuple[str, str, float] | None:
    """Find best matching product slug using multiple candidate slugs.
    Tries each candidate in order (best first), returns first good match."""

    overall_best = None  # (slug, url, score)

    for deal_slug in deal_candidates:
        # 1. Exact match
        if deal_slug in slug_index:
            return deal_slug, slug_index[deal_slug], 1.0

        # 2. Substring: product slug starts with deal slug or contains it
        for site_slug, url in slug_index.items():
            if len(deal_slug) >= 4 and deal_slug in site_slug:
                score = len(deal_slug) / max(len(site_slug), 1)
                if score >= 0.4:  # lower threshold for substring
                    return site_slug, url, min(score + 0.2, 1.0)
            if len(site_slug) >= 4 and site_slug in deal_slug:
                score = len(site_slug) / max(len(deal_slug), 1)
                if score >= 0.4:
                    return site_slug, url, min(score + 0.2, 1.0)

        # 3. Fuzzy matching
        for site_slug, url in slug_index.items():
            score = SequenceMatcher(None, deal_slug, site_slug).ratio()
            if overall_best is None or score > overall_best[2]:
                overall_best = (site_slug, url, score)

    if overall_best and overall_best[2] >= min_score:
        return overall_best
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--limit", default=0, type=int, help="Limit product pages to download (0=all)")
@click.option("--min-score", default=0.60, type=float, help="Min fuzzy match score")
@click.option("--skip-download", is_flag=True, help="Skip Phase 1 (image download), only do mapping")
def main(limit, min_score, skip_download):
    """Fetch product images from labus.pro using Product-Centric strategy."""

    # Load or create index
    index = {"products": {}, "deal_to_product": {}}
    if INDEX_FILE.exists():
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            index = json.load(f)
            # Migrate from v1 format if needed
            if "products" not in index:
                index = {"products": {}, "deal_to_product": {}}

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Download images by unique product pages
    # ═══════════════════════════════════════════════════════════════════════════
    if not skip_download:
        print("=" * 60)
        print("ФАЗА 1: Скачивание изображений по уникальным страницам товаров")
        print("=" * 60)

        product_urls = fetch_sitemap(SITEMAP_URL)

        # Filter out already downloaded products
        already_done = set(index["products"].keys())
        to_download = []
        for url in product_urls:
            slug = extract_slug(url)
            if slug and slug not in already_done:
                to_download.append((slug, url))

        print(f"  Уже скачано: {len(already_done)}, осталось: {len(to_download)}")

        if limit > 0:
            to_download = to_download[:limit]
            print(f"  Лимит: {limit}")

        stats = {"downloaded": 0, "no_images": 0, "errors": 0}
        OFFERS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        for slug, page_url in tqdm(to_download, desc="Phase 1: Downloading"):
            section = extract_section(page_url)
            page_data = extract_page_data(page_url)
            image_urls = page_data["images"]
            product_title = page_data["title"]

            if not image_urls:
                stats["no_images"] += 1
                index["products"][slug] = {
                    "title": product_title,
                    "page_url": page_url,
                    "section": section,
                    "images": [],
                    "deal_ids": [],
                    "status": "no_images"
                }
                continue

            # Download images into OFFERS_IMAGE/{section}/{slug}/
            product_dir = OFFERS_IMAGE_DIR / section / slug
            local_files = []

            for idx, img_url in enumerate(image_urls):
                ext = Path(urlparse(img_url).path).suffix or ".webp"
                filename = f"cover{ext}" if idx == 0 else f"gallery_{idx}{ext}"
                save_path = product_dir / filename

                if download_image(img_url, save_path):
                    local_files.append(filename)

            if local_files:
                stats["downloaded"] += 1
                index["products"][slug] = {
                    "title": product_title,
                    "page_url": page_url,
                    "section": section,
                    "images": local_files,
                    "image_urls": image_urls[:len(local_files)],
                    "deal_ids": [],
                    "status": "ok"
                }
            else:
                stats["errors"] += 1
                index["products"][slug] = {
                    "title": product_title,
                    "page_url": page_url,
                    "section": section,
                    "images": [],
                    "deal_ids": [],
                    "status": "download_error"
                }

            # Incremental save every 20 products
            if (stats["downloaded"] + stats["no_images"]) % 20 == 0:
                with open(INDEX_FILE, 'w', encoding='utf-8') as f:
                    json.dump(index, f, ensure_ascii=False, indent=2)

            # Polite delay
            time.sleep(0.5)

        print(f"\n  Phase 1 Results:")
        print(f"    Downloaded: {stats['downloaded']}")
        print(f"    No images:  {stats['no_images']}")
        print(f"    Errors:     {stats['errors']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Map offer IDs to product slugs (using offers.csv)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ФАЗА 2: Маппинг товаров (offers.csv) на страницы сайта")
    print("=" * 60)

    if not OFFERS_CSV.exists():
        print(f"  ERROR: {OFFERS_CSV} not found!")
        return

    # Read unique offers from CSV (deduplicate by ID)
    offers = {}  # {offer_id: title}
    with open(OFFERS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            oid = row.get('ID', '').strip()
            title = row.get('TITLE', '').strip()
            if oid and title and oid not in offers:
                offers[oid] = title
    print(f"  Уникальных товаров (offers): {len(offers)}")

    # Build slug index from downloaded products (only those with images)
    slug_index = {}
    for slug, info in index["products"].items():
        if info.get("status") == "ok" and info.get("images"):
            slug_index[slug] = info["page_url"]
    print(f"  Товаров с фото на сайте: {len(slug_index)}")

    # Clear existing deal_ids from products
    for prod in index["products"].values():
        prod["deal_ids"] = []

    match_stats = {"matched": 0, "unmatched": 0}
    index["deal_to_product"] = {}

    for offer_id, title in tqdm(offers.items(), desc="Phase 2: Mapping"):
        deal_candidates = normalize_deal_title(title)
        match = find_best_match(deal_candidates, slug_index, min_score=min_score)

        if match:
            matched_slug, _, score = match
            index["deal_to_product"][offer_id] = {
                "product_slug": matched_slug,
                "score": round(score, 3)
            }
            # Add offer_id to product's list
            if matched_slug in index["products"]:
                if offer_id not in index["products"][matched_slug]["deal_ids"]:
                    index["products"][matched_slug]["deal_ids"].append(offer_id)
            match_stats["matched"] += 1
        else:
            match_stats["unmatched"] += 1

    print(f"\n  Phase 2 Results:")
    print(f"    Matched:   {match_stats['matched']}")
    print(f"    Unmatched: {match_stats['unmatched']}")

    # Stats on product usage
    used_products = sum(1 for p in index["products"].values() if p.get("deal_ids"))
    total_mapped = sum(len(p.get("deal_ids", [])) for p in index["products"].values())
    print(f"    Products with offers: {used_products}")
    print(f"    Total mappings:       {total_mapped}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Save final index
    # ═══════════════════════════════════════════════════════════════════════════
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    total_images = sum(len(p.get("images", [])) for p in index["products"].values() if p.get("status") == "ok")
    print(f"\n{'=' * 60}")
    print(f"ИТОГО:")
    print(f"  Уникальных товаров с фото: {used_products}")
    print(f"  Всего файлов изображений:  {total_images}")
    print(f"  Offers с привязкой:        {match_stats['matched']}")
    print(f"  Offers без привязки:       {match_stats['unmatched']}")
    print(f"  Индекс: {INDEX_FILE}")
    print(f"  Папка:  {OFFERS_IMAGE_DIR}")


if __name__ == "__main__":
    main()
