"""P21.B.1: prepare embedding fine-tune dataset from existing reranker training pairs.

Use case: reuse P16.A.2 reranker_training_pairs_v3.jsonl (collision-aware) для
обучения BGE-M3 embedding модели через MultipleNegativesRankingLoss (MNRL).

MNRL берёт только positives (query, positive_passage), in-batch other pairs
автоматически становятся negatives. Так что:
  - Filter: только label > 0.5 (positives)
  - Filter: passage length >= 30 chars
  - Cap: 5 positives per unique query (uniform diversity, avoid overfit)
  - Deduplicate by (query[:200], passage[:200])

Input: SOFT_TUNE_DATA/reranker_training_pairs_v3.jsonl (25,721 pairs)
Output: SOFT_TUNE_DATA/embedding_training_pairs_v1.jsonl
        — list of {"query": str, "passage": str} ready for SentenceTransformer
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "SOFT_TUNE_DATA"
IN_PATH = DATA / "reranker_training_pairs_v3.jsonl"
OUT_PATH = DATA / "embedding_training_pairs_v1.jsonl"

# Filtering parameters
POSITIVE_THRESHOLD = 0.5  # label > this == positive
MIN_PASSAGE_LEN = 30
MAX_POSITIVES_PER_QUERY = 5
SEED = 42


def main():
    if not IN_PATH.exists():
        print(f"ERROR: {IN_PATH} not found. Run P16.A.2 reranker_train_prep_v3.py first.")
        return

    print(f"Loading {IN_PATH}...")
    rows = []
    for line in IN_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    print(f"  loaded {len(rows)} pairs")

    # Filter positives + length
    positives = [
        r for r in rows
        if r.get("label", 0) > POSITIVE_THRESHOLD
        and len(r.get("passage") or "") >= MIN_PASSAGE_LEN
        and len(r.get("query") or "") >= 5
    ]
    print(f"  positives (label > {POSITIVE_THRESHOLD}, len >= {MIN_PASSAGE_LEN}): {len(positives)}")

    # Group by query, cap N per query
    by_query: dict[str, list[dict]] = defaultdict(list)
    for r in positives:
        q = r["query"][:500].strip()  # normalize query
        by_query[q].append(r)
    print(f"  unique queries: {len(by_query)}")

    # Sort each group by label desc (best positives first), cap at MAX_POSITIVES
    capped = []
    for q, group in by_query.items():
        group.sort(key=lambda r: r.get("label", 0), reverse=True)
        for r in group[:MAX_POSITIVES_PER_QUERY]:
            capped.append({
                "query": r["query"],
                "passage": r["passage"],
                "label": r.get("label", 1.0),
                "topic": r.get("topic", "?"),
            })
    print(f"  after cap ({MAX_POSITIVES_PER_QUERY} per query): {len(capped)}")

    # Dedup (query[:200], passage[:200])
    seen = set()
    deduped = []
    for r in capped:
        key = (r["query"][:200], r["passage"][:200])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    print(f"  after dedup: {len(deduped)}")

    # Stats by topic
    topic_counts: dict[str, int] = defaultdict(int)
    for r in deduped:
        topic_counts[r["topic"]] += 1
    print(f"\nTopic distribution:")
    for topic, n in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic:<24} {n:>5}")

    # Write
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ wrote {len(deduped)} pairs → {OUT_PATH}")


if __name__ == "__main__":
    main()
