#!/usr/bin/env python3
"""P11-R4: Select up to 27 feedback pairs relevant to roadmap/process queries.

Selects pairs where user_query OR expert_comment mentions process/roadmap keywords,
plus a random sample of general pairs to measure baseline false-positive rate.

Output: r4_test_pairs.json (subset of feedback_pairs.json)
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent

ROADMAP_QUERY_PATTERN = re.compile(
    r"этапы?|процесс|регламент|сроки?|как\s+(вы\s+)?(делает|работает|производит)"
    r"|порядок\s+работ|схема\s+работ|как\s+происходит|расскажите\s+про",
    re.IGNORECASE,
)

ROADMAP_COMMENT_PATTERN = re.compile(
    r"регламент|этапы?|процесс|сроки?|дорожн\w+\s+карт",
    re.IGNORECASE,
)


def main():
    pairs_path = ROOT / "feedback_pairs.json"
    if not pairs_path.exists():
        print(f"ERROR: {pairs_path} not found")
        return 1

    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(pairs)} feedback pairs")

    # Priority 1: roadmap-relevant by query
    roadmap_query = [
        p for p in pairs
        if ROADMAP_QUERY_PATTERN.search(p.get("user_query", ""))
    ]

    # Priority 2: roadmap-relevant by expert comment
    roadmap_comment = [
        p for p in pairs
        if p not in roadmap_query
        and ROADMAP_COMMENT_PATTERN.search(p.get("expert_comment", ""))
    ]

    # Priority 3: any expert-commented pairs (general baseline)
    expert_commented = [
        p for p in pairs
        if p.get("expert_comment", "").strip()
        and p not in roadmap_query
        and p not in roadmap_comment
    ]

    selected = []
    selected.extend(roadmap_query[:15])
    selected.extend(roadmap_comment[:6])
    remaining = 27 - len(selected)
    selected.extend(expert_commented[:remaining])

    print(f"Selected {len(selected)} pairs:")
    print(f"  {len(roadmap_query[:15])} by roadmap query keywords")
    print(f"  {len(roadmap_comment[:6])} by roadmap expert comment")
    print(f"  {len(selected) - len(roadmap_query[:15]) - len(roadmap_comment[:6])} general baseline")

    out_path = ROOT / "r4_test_pairs.json"
    out_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written to {out_path}")

    print("\nSelected pair IDs:", [p["feedback_id"] for p in selected])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
