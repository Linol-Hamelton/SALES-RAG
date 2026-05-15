"""P15.D-prep v2: PROPER hard-negative mining for reranker fine-tune.

v1 mistake: labeled ALL refs as 1 if winner=rag, 0 if no_rag → noisy labels →
broken model (0.99 for irrelevant docs).

v2 methodology:
  POSITIVES (label=1.0): top-5 refs that baseline retrieved & ranked for a query.
    Baseline already filtered these as relevant. Implicit ground truth.
    Refinement: if judge.winner=rag → label=1.0; if no_rag → label=0.7;
                tie/missing → label=0.85.
  NEGATIVES (label=0.0): refs from OTHER queries with DIFFERENT topic.
    Per positive — sample 1-2 hard negatives (close in embedding but wrong topic).

Format: SOFT_TUNE_DATA/reranker_training_pairs_v2.jsonl
  {"query": str, "passage": str, "label": float, "type": "pos"|"neg"}

Expected: ~6000 positives + ~12000 negatives = ~18k balanced training pairs.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "SOFT_TUNE_DATA"
OUT = DATA / "reranker_training_pairs_v2.jsonl"

VERSIONS = ["", "_v2", "_v3", "_v4", "_v5", "_v6", "_v7", "_v10", "_v11"]
TOP_REFS_AS_POS = 5
NEG_PER_POS = 2
MAX_SNIPPET_CHARS = 400
SEED = 42


def collect_all_queries():
    """Returns list of dicts: {id, query, topic, persona, refs:[snippet,...], winner}."""
    out = []
    for ver in VERSIONS:
        scores_p = DATA / f"scores{ver}.jsonl"
        ans_p = DATA / f"answers_rag{ver}.jsonl"
        if not scores_p.exists() or not ans_p.exists():
            continue
        scores = {}
        for line in scores_p.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            s = json.loads(line)
            if "error" in s: continue
            if s.get("winner"):
                scores[s["id"]] = s["winner"]
        for line in ans_p.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            a = json.loads(line)
            if a.get("error"): continue
            fr = a.get("full_response") or {}
            refs = (fr.get("references") or [])[:TOP_REFS_AS_POS]
            snippets = []
            for r in refs:
                snip = (r.get("snippet") or "").strip()
                if snip and len(snip) >= 20:
                    snippets.append(snip[:MAX_SNIPPET_CHARS])
            if not snippets:
                continue
            out.append({
                "id": a["id"],
                "query": a["query"],
                "topic": a.get("topic", "unk"),
                "persona": a.get("persona", "unk"),
                "snippets": snippets,
                "winner": scores.get(a["id"], "tie"),
                "version": ver or "v1",
            })
    return out


def main():
    random.seed(SEED)
    print("Collecting queries...")
    queries = collect_all_queries()
    print(f"  {len(queries)} queries total")

    # Index passages by topic for cross-topic negative sampling
    passages_by_topic = defaultdict(list)
    for q in queries:
        for snip in q["snippets"]:
            passages_by_topic[q["topic"]].append(snip)
    print(f"  topics: {len(passages_by_topic)}")
    for t in sorted(passages_by_topic.keys())[:5]:
        print(f"    {t}: {len(passages_by_topic[t])} snippets")

    # Build training pairs
    pairs = []
    seen_pairs = set()

    label_map = {"rag": 1.0, "no_rag": 0.7, "tie": 0.85}

    for q in queries:
        pos_label = label_map.get(q["winner"], 0.85)

        # 1) Positive pairs (query, top-ref)
        for snip in q["snippets"]:
            key = (q["query"][:200], snip[:200])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs.append({
                "query": q["query"],
                "passage": snip,
                "label": pos_label,
                "type": "pos",
                "topic": q["topic"],
            })

            # 2) Hard negatives: refs from DIFFERENT topic queries
            other_topics = [t for t in passages_by_topic.keys() if t != q["topic"]]
            if not other_topics:
                continue
            for _ in range(NEG_PER_POS):
                neg_topic = random.choice(other_topics)
                neg_candidates = passages_by_topic[neg_topic]
                if not neg_candidates:
                    continue
                neg_snip = random.choice(neg_candidates)
                # Ensure we're not accidentally adding a self-positive
                if neg_snip in q["snippets"]:
                    continue
                neg_key = (q["query"][:200], neg_snip[:200])
                if neg_key in seen_pairs:
                    continue
                seen_pairs.add(neg_key)
                pairs.append({
                    "query": q["query"],
                    "passage": neg_snip,
                    "label": 0.0,
                    "type": "neg",
                    "topic_query": q["topic"],
                    "topic_passage": neg_topic,
                })

    random.shuffle(pairs)
    pos_count = sum(1 for p in pairs if p["type"] == "pos")
    neg_count = sum(1 for p in pairs if p["type"] == "neg")
    print(f"\nFinal pairs: {len(pairs)}")
    print(f"  positives: {pos_count}")
    print(f"  negatives: {neg_count}  (ratio neg:pos = {neg_count/max(1,pos_count):.2f})")

    with OUT.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n✓ wrote {len(pairs)} → {OUT}")


if __name__ == "__main__":
    main()
