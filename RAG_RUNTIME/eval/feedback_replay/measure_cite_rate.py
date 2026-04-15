#!/usr/bin/env python3
"""P11-R4: Measure roadmap citation rate before vs after P11.

Compares OLD responses (from DB, pre-P11) vs NEW responses (from prod replay)
on the metric: does the response cite регламент/этап/сроки?

Usage:
    # 1. Run replay first (against current prod):
    python replay.py --input r4_test_pairs.json --output r4_replays.json

    # 2. Measure cite rate:
    python measure_cite_rate.py --input r4_replays.json

    # Or against the default replays.json:
    python measure_cite_rate.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Windows console cp1251 can't encode arrows/Cyrillic special chars
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent

ROADMAP_KEYWORDS = re.compile(
    r"регламент|этапы?|сроки?|занимает\s+\d|в\s+течение"
    r"|порядок\s+работ|процесс\s+(создани|разработ|производ)"
    r"|шаг\s+\d|стадия|фаза",
    re.IGNORECASE,
)


def extract_text(entry: dict) -> str:
    kf = entry.get("key_fields") or {}
    return " ".join(filter(None, [
        kf.get("summary", ""),
        kf.get("reasoning", ""),
        " ".join(kf.get("flags", [])),
    ]))


def cites_roadmap(text: str) -> bool:
    return bool(ROADMAP_KEYWORDS.search(text))


def main():
    ap = argparse.ArgumentParser(description="Measure P11 roadmap citation rate")
    ap.add_argument("--input", default=str(ROOT / "replays.json"),
                    help="Path to replays.json (output of replay.py)")
    ap.add_argument("--output", default=str(ROOT / "r4_cite_rate_report.txt"),
                    help="Path to write text report")
    args = ap.parse_args()

    replays_path = Path(args.input)
    if not replays_path.exists():
        print(f"ERROR: {replays_path} not found. Run replay.py first.")
        return 1

    replays = json.loads(replays_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(replays)} replay pairs from {replays_path}")

    old_cited = 0
    new_cited = 0
    improved = 0   # old=False, new=True
    regressed = 0  # old=True, new=False
    errors = 0
    per_pair = []

    for r in replays:
        fb_id = r.get("feedback_id", "?")
        query = r.get("user_query", "")[:80]

        new_entry = r.get("new", {})
        if new_entry.get("error"):
            errors += 1
            per_pair.append({"id": fb_id, "query": query, "old": None, "new": None, "verdict": "error"})
            continue

        old_text = extract_text(r.get("old", {}))
        # new entry has key_fields directly (same as old), or nested under response
        new_text = extract_text(new_entry)

        old_ok = cites_roadmap(old_text)
        new_ok = cites_roadmap(new_text)

        if old_ok:
            old_cited += 1
        if new_ok:
            new_cited += 1
        if not old_ok and new_ok:
            improved += 1
        if old_ok and not new_ok:
            regressed += 1

        verdict = "improved" if (not old_ok and new_ok) else \
                  "regressed" if (old_ok and not new_ok) else \
                  "same_yes" if (old_ok and new_ok) else "same_no"

        per_pair.append({
            "id": fb_id,
            "query": query,
            "old": old_ok,
            "new": new_ok,
            "verdict": verdict,
        })

    total = len(replays)
    valid = total - errors
    old_pct = 100 * old_cited / valid if valid else 0
    new_pct = 100 * new_cited / valid if valid else 0
    delta = new_pct - old_pct

    lines = [
        "=" * 60,
        "  P11-R4: Roadmap Citation Rate Report",
        "=" * 60,
        f"  Input:  {replays_path}",
        f"  Pairs:  {total} total, {errors} errors, {valid} valid",
        "",
        f"  Before P11:  {old_cited}/{valid} ({old_pct:.1f}%) cite регламент/этап/сроки",
        f"  After  P11:  {new_cited}/{valid} ({new_pct:.1f}%) cite регламент/этап/сроки",
        f"  Delta:       {delta:+.1f}%",
        "",
        f"  Improved  (0→1):  {improved}",
        f"  Regressed (1→0):  {regressed}",
        f"  Same-Yes  (1→1):  {sum(1 for p in per_pair if p['verdict']=='same_yes')}",
        f"  Same-No   (0→0):  {sum(1 for p in per_pair if p['verdict']=='same_no')}",
        f"  Errors:           {errors}",
        "",
        "  Per-pair breakdown:",
        f"  {'ID':>5}  {'Old':>5}  {'New':>5}  Verdict    Query",
        "  " + "-" * 80,
    ]

    for p in per_pair:
        old_s = ("yes" if p["old"] else "no") if p["old"] is not None else "err"
        new_s = ("yes" if p["new"] else "no") if p["new"] is not None else "err"
        lines.append(f"  {p['id']:>5}  {old_s:>5}  {new_s:>5}  {p['verdict']:<12} {p['query']}")

    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    out_path = Path(args.output)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
