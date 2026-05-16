"""P16.A.2-prep v3: collision-aware hard negatives for reranker fine-tune.

v2 mistake: same-direction near-products (буклет vs листовка, визитка vs флаер)
treated as completely irrelevant → model over-penalized cross-product matching.
v12 print_flyer regressed -15.2pp vs v10.

v3 methodology:
  POSITIVES (label 0.7-1.0): top-5 refs that baseline retrieved & ranked per query.
    Same as v2 — refinement: rag-win = 1.0, no_rag-win = 0.7, tie = 0.85.

  HARD NEGATIVES (label=0.0): refs from CROSS-DIRECTION queries.
    Buklet (print direction) ↔ pricing_justify or signage — far enough.

  SOFT NEGATIVES (label=0.3): refs from SAME direction but DIFFERENT product.
    Visit ↔ flyer: both print, but distinct product. Soft-label avoids
    over-penalty while still teaching distinction.

  COLLISION WHITELIST: if query explicitly mentions BOTH products (e.g.
    "буклет и листовка"), neither is a negative for the other.

Format: SOFT_TUNE_DATA/reranker_training_pairs_v3.jsonl
  {"query": str, "passage": str, "label": float, "type": "pos"|"hard_neg"|"soft_neg"}

Expected: ~8.5k pos + ~12k hard_neg + ~5k soft_neg = ~25.5k pairs.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "SOFT_TUNE_DATA"
OUT = DATA / "reranker_training_pairs_v3.jsonl"

VERSIONS = ["", "_v2", "_v3", "_v4", "_v5", "_v6", "_v7", "_v10", "_v11"]
TOP_REFS_AS_POS = 5
HARD_NEG_PER_POS = 1.5  # avg, exact ratio via random
SOFT_NEG_PER_POS = 0.6
MAX_SNIPPET_CHARS = 400
SEED = 42

# Direction grouping — topics that share a direction. Same direction = soft neg.
DIRECTION_OF_TOPIC = {
    # Print direction
    "print_visitka": "print",
    "print_flyer": "print",
    "print_misc": "print",
    # Signage direction
    "ad_banner": "signage",
    "ad_signboard": "signage",
    "ad_misc": "signage",
    "signage_specs": "signage",
    # Design direction
    "design_logo": "design",
    "design_brand": "design",
    # Merch direction
    "merch": "merch",
    # Stickers
    "stickers": "stickers",
    # Manager workflow topics — direction-agnostic
    "objection_handling": "workflow",
    "objections": "workflow",
    "discovery": "workflow",
    "upsell": "workflow",
    "consultation": "workflow",
    "pricing_justify": "workflow",
    "historical_links": "workflow",
    "smeta_assist": "workflow",
    "directions": "workflow",
    "logistics": "workflow",
    "vocabulary": "workflow",
}

# Product token aliases for collision whitelist. If query contains 2+ keywords
# from DIFFERENT groups simultaneously, don't treat those products as negatives
# for each other.
COLLISION_GROUPS = {
    "buklet": ["буклет", "брошюр"],
    "listovka": ["листовк", "флаер"],
    "visitka": ["визитк"],
    "merch_textile": ["футболк", "толстовк"],
    "merch_drink": ["кружк", "термокружк"],
    "signage_neon": ["неон"],
    "signage_letters": ["объёмные буквы", "объемные буквы"],
}


def _which_collision_groups(text: str) -> set[str]:
    """Return which collision groups are mentioned in text."""
    text_lower = text.lower()
    hits = set()
    for group_name, kws in COLLISION_GROUPS.items():
        if any(kw in text_lower for kw in kws):
            hits.add(group_name)
    return hits


def collect_all_queries():
    """Returns list of dicts: {id, query, topic, direction, persona, snippets, winner}."""
    out = []
    for ver in VERSIONS:
        scores_p = DATA / f"scores{ver}.jsonl"
        ans_p = DATA / f"answers_rag{ver}.jsonl"
        if not scores_p.exists() or not ans_p.exists():
            continue
        scores = {}
        for line in scores_p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            if "error" in s:
                continue
            if s.get("winner"):
                scores[s["id"]] = s["winner"]
        for line in ans_p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            a = json.loads(line)
            if a.get("error"):
                continue
            fr = a.get("full_response") or {}
            refs = (fr.get("references") or [])[:TOP_REFS_AS_POS]
            snippets = []
            for r in refs:
                snip = (r.get("snippet") or "").strip()
                if snip and len(snip) >= 20:
                    snippets.append(snip[:MAX_SNIPPET_CHARS])
            if not snippets:
                continue
            topic = a.get("topic", "unk")
            out.append({
                "id": a["id"],
                "query": a["query"],
                "topic": topic,
                "direction": DIRECTION_OF_TOPIC.get(topic, "other"),
                "persona": a.get("persona", "unk"),
                "snippets": snippets,
                "winner": scores.get(a["id"], "tie"),
                "version": ver or "v1",
                "collision_groups": _which_collision_groups(a["query"]),
            })
    return out


def main():
    random.seed(SEED)
    print("Collecting queries...")
    queries = collect_all_queries()
    print(f"  {len(queries)} queries total")

    # Index passages two ways:
    #   - by topic (for soft-neg same-direction-different-product)
    #   - by direction (for hard-neg cross-direction)
    passages_by_topic = defaultdict(list)
    passages_by_direction = defaultdict(list)
    for q in queries:
        for snip in q["snippets"]:
            passages_by_topic[q["topic"]].append(snip)
            passages_by_direction[q["direction"]].append(snip)
    print(f"  topics: {len(passages_by_topic)}")
    print(f"  directions: {sorted(passages_by_direction.keys())}")
    for d in sorted(passages_by_direction.keys()):
        print(f"    {d}: {len(passages_by_direction[d])} snippets")

    pairs = []
    seen_pairs = set()
    label_map = {"rag": 1.0, "no_rag": 0.7, "tie": 0.85}

    stats = {"pos": 0, "hard_neg": 0, "soft_neg": 0, "collision_skip": 0}

    for q in queries:
        pos_label = label_map.get(q["winner"], 0.85)
        q_dir = q["direction"]
        q_topic = q["topic"]
        q_collisions = q["collision_groups"]

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
                "topic": q_topic,
                "direction": q_dir,
            })
            stats["pos"] += 1

            # Hard negatives: from a DIFFERENT direction (truly far).
            other_dirs = [d for d in passages_by_direction if d != q_dir and d != "workflow"]
            for _ in range(int(HARD_NEG_PER_POS) + (1 if random.random() < (HARD_NEG_PER_POS - int(HARD_NEG_PER_POS)) else 0)):
                if not other_dirs:
                    break
                neg_dir = random.choice(other_dirs)
                neg_pool = passages_by_direction[neg_dir]
                if not neg_pool:
                    continue
                neg_snip = random.choice(neg_pool)
                if neg_snip in q["snippets"]:
                    continue
                # collision check: skip if query mentions same product group as neg snippet
                neg_groups = _which_collision_groups(neg_snip)
                if q_collisions and neg_groups and (q_collisions & neg_groups):
                    stats["collision_skip"] += 1
                    continue
                neg_key = (q["query"][:200], neg_snip[:200])
                if neg_key in seen_pairs:
                    continue
                seen_pairs.add(neg_key)
                pairs.append({
                    "query": q["query"],
                    "passage": neg_snip,
                    "label": 0.0,
                    "type": "hard_neg",
                    "topic_query": q_topic,
                    "direction_query": q_dir,
                    "direction_passage": neg_dir,
                })
                stats["hard_neg"] += 1

            # Soft negatives: SAME direction, DIFFERENT topic. Label 0.3 (not 0.0).
            same_dir_topics = [
                t for t, td in DIRECTION_OF_TOPIC.items()
                if td == q_dir and t != q_topic and t in passages_by_topic
            ]
            if same_dir_topics and random.random() < SOFT_NEG_PER_POS:
                soft_topic = random.choice(same_dir_topics)
                soft_pool = passages_by_topic[soft_topic]
                if soft_pool:
                    soft_snip = random.choice(soft_pool)
                    if soft_snip not in q["snippets"]:
                        # collision check still applies
                        soft_groups = _which_collision_groups(soft_snip)
                        if not (q_collisions and soft_groups and (q_collisions & soft_groups)):
                            soft_key = (q["query"][:200], soft_snip[:200])
                            if soft_key not in seen_pairs:
                                seen_pairs.add(soft_key)
                                pairs.append({
                                    "query": q["query"],
                                    "passage": soft_snip,
                                    "label": 0.3,
                                    "type": "soft_neg",
                                    "topic_query": q_topic,
                                    "topic_passage": soft_topic,
                                    "direction": q_dir,
                                })
                                stats["soft_neg"] += 1

    random.shuffle(pairs)
    print(f"\nFinal pairs: {len(pairs)}")
    print(f"  positives:  {stats['pos']}")
    print(f"  hard_negs:  {stats['hard_neg']}  (cross-direction)")
    print(f"  soft_negs:  {stats['soft_neg']}  (same-direction, label=0.3)")
    print(f"  collision_skipped: {stats['collision_skip']}  (saved from over-penalty)")
    print(f"  ratio neg:pos = {(stats['hard_neg'] + stats['soft_neg'])/max(1,stats['pos']):.2f}")

    with OUT.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n✓ wrote {len(pairs)} → {OUT}")


if __name__ == "__main__":
    main()
