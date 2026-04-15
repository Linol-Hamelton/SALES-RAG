#!/usr/bin/env python3
"""
P10-B1: Rank SmetaEngine categories by business volume to prioritize
bridge-document coverage.

Output: CSV sorted by revenue_proxy DESC with:
  category_name, deals_count, avg_deal_value, revenue_proxy,
  has_bridge, has_roadmap, has_packages_in_roadmap, roadmap_file

Usage:
    python scripts/rank_categories_by_volume.py

Sources:
  RAG_ANALYTICS/output/smeta_templates.json  — ground-truth category stats
  RAG_RUNTIME/data/bridge_docs.jsonl          — existing bridges (has_bridge)
  RAG_DATA/ROADMAPS/*.md                     — markdown roadmaps (has_roadmap,
                                                has_packages_in_roadmap)
"""
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REPO_ROOT = PROJECT_ROOT.parent
SMETA_TEMPLATES = REPO_ROOT / "RAG_ANALYTICS" / "output" / "smeta_templates.json"
BRIDGE_JSONL = PROJECT_ROOT / "data" / "bridge_docs.jsonl"
ROADMAPS_DIR = REPO_ROOT / "RAG_DATA" / "ROADMAPS"
OUTPUT_CSV = PROJECT_ROOT / "data" / "category_coverage.csv"

# Regex: "Пакет «Стандарт» (27 500 – 48 500 руб.):" and variants
PACKAGE_PRICE_RX = re.compile(
    r"Пакет\s*[«\"']([^»\"']+)[»\"']\s*\(([\d\s]+)\s*[–\-−]\s*([\d\s]+)\s*руб",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    """Aggressive normalization for matching: lowercase, strip accents/yo/punct."""
    s = unicodedata.normalize("NFKD", s or "").casefold()
    s = s.replace("ё", "е")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_bridges() -> set[str]:
    """Return set of normalized service names that already have bridges."""
    out = set()
    if not BRIDGE_JSONL.exists():
        return out
    with open(BRIDGE_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            svc = (doc.get("metadata") or {}).get("service", "")
            if svc:
                out.add(_norm(svc))
    return out


def load_roadmaps() -> dict[str, dict]:
    """Return {normalized_stem: {"path": Path, "has_packages": bool}} for each .md roadmap."""
    out: dict[str, dict] = {}
    if not ROADMAPS_DIR.is_dir():
        return out
    for p in ROADMAPS_DIR.glob("*.md"):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        has_pkgs = bool(PACKAGE_PRICE_RX.search(text))
        # Use several normalized keys per roadmap to maximize category match:
        # full stem, stem without leading "Дизайн"/"Производство"/etc., and
        # tokens of the stem.
        stem = p.stem
        out[_norm(stem)] = {"path": p, "has_packages": has_pkgs}
    return out


def match_roadmap(cat_name: str, roadmaps: dict[str, dict]) -> tuple[str, bool]:
    """Return (roadmap_file, has_packages) for best match, or ("", False)."""
    cat_n = _norm(cat_name)
    cat_tokens = set(cat_n.split())
    if not cat_tokens:
        return "", False
    best_path, best_has, best_score = "", False, 0
    for rm_n, info in roadmaps.items():
        rm_tokens = set(rm_n.split())
        if not rm_tokens:
            continue
        overlap = cat_tokens & rm_tokens
        # Require at least one meaningful token overlap; prefer max overlap
        # and longer token-length match.
        if not overlap:
            continue
        # Exclude noise words from scoring so "Дизайн X" doesn't match every
        # roadmap that starts with "Дизайн".
        noise = {"дизайн", "производство", "печать", "печати", "брендированный",
                 "брендированные", "брендированная", "брендированного", "карта",
                 "дорожная", "и", "для"}
        strong = overlap - noise
        score = len(strong) * 10 + len(overlap)
        if score > best_score:
            best_score = score
            best_path = info["path"].name
            best_has = info["has_packages"]
    return best_path, best_has


def main() -> int:
    if not SMETA_TEMPLATES.exists():
        print(f"ERROR: {SMETA_TEMPLATES} not found", file=sys.stderr)
        return 1

    templates = json.loads(SMETA_TEMPLATES.read_text(encoding="utf-8"))
    categories = templates.get("categories", [])

    bridges = load_bridges()
    roadmaps = load_roadmaps()

    rows = []
    for c in categories:
        name = c.get("category_name", "")
        if not name:
            continue
        deals = int(c.get("deals_count") or 0)
        avg = float(c.get("avg_deal_value") or 0.0)
        revenue = deals * avg
        has_bridge = any(
            _norm(name) in b or b in _norm(name) or bool(set(_norm(name).split()) & set(b.split()))
            for b in bridges
        )
        rm_file, rm_has_pkgs = match_roadmap(name, roadmaps)
        rows.append({
            "category_name": name,
            "deals_count": deals,
            "avg_deal_value": round(avg, 2),
            "revenue_proxy": round(revenue, 2),
            "has_bridge": "1" if has_bridge else "0",
            "has_roadmap": "1" if rm_file else "0",
            "has_packages_in_roadmap": "1" if rm_has_pkgs else "0",
            "roadmap_file": rm_file,
        })

    rows.sort(key=lambda r: -r["revenue_proxy"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()
        w.writerows(rows)

    total_deals = sum(r["deals_count"] for r in rows)
    total_revenue = sum(r["revenue_proxy"] for r in rows)
    covered_revenue = sum(r["revenue_proxy"] for r in rows if r["has_bridge"] == "1")
    covered_pct = (covered_revenue / total_revenue * 100) if total_revenue else 0.0

    # Find top-25 bridge candidates: high revenue, no bridge yet
    candidates = [r for r in rows if r["has_bridge"] == "0"][:25]

    summary_path = OUTPUT_CSV.with_suffix(".summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total categories: {len(rows)}\n")
        f.write(f"Total deals: {total_deals:,}\n")
        f.write(f"Total revenue proxy: {total_revenue:,.0f} RUB\n")
        f.write(f"Already covered by bridges: {covered_revenue:,.0f} RUB ({covered_pct:.1f}%)\n")
        f.write(f"\nTop-25 bridge candidates (highest revenue WITHOUT bridge):\n")
        f.write(f"{'rank':<5}{'revenue_proxy':>16}  {'deals':>6}  "
                f"{'rm':>3} {'pkg':>4}  category_name\n")
        for i, r in enumerate(candidates, 1):
            f.write(
                f"{i:<5}{r['revenue_proxy']:>16,.0f}  {r['deals_count']:>6}  "
                f"{r['has_roadmap']:>3} {r['has_packages_in_roadmap']:>4}  "
                f"{r['category_name']}\n"
            )
        f.write(f"\nCoverage target (25 bridges): forecast revenue share ~"
                f"{(sum(r['revenue_proxy'] for r in candidates) + covered_revenue) / total_revenue * 100 if total_revenue else 0:.1f}%\n")

    print(f"Written: {OUTPUT_CSV}")
    print(f"Written: {summary_path}")
    print(f"Categories: {len(rows)}")
    print(f"Currently covered by bridges: {covered_pct:.1f}% of revenue")
    return 0


if __name__ == "__main__":
    sys.exit(main())
