"""P14.5 Phase 5.A: build expanded intent_prototypes.yaml from labelled v1/v2/v3 data.

Reads questions_*_v*.jsonl from SOFT_TUNE_DATA, groups by `expected_intent`,
dedups via BGE-M3 cosine (sim>0.85 → drop), caps 60 per intent, merges with
existing 58 prototypes.

Output: configs/intent_prototypes.yaml (in-place update).

Maps `expected_intent` labels from question-generation prompt to actual
INTENT_NAMES used by classifier:
  product_query → product_query
  consultation → consultation
  historical_request → historical_request
  persuasion → describe   (closest match in classifier)
  underspec → underspec
  general → (skip — not useful as prototype)

Usage:
  PYTHONIOENCODING=utf-8 python SOFT_TUNE_DATA/scripts/build_intent_prototypes.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "SOFT_TUNE_DATA"
EXISTING_YAML = ROOT / "configs" / "intent_prototypes.yaml"
OUT_YAML = EXISTING_YAML  # in-place update

# Labels in questions_*.jsonl come from deepseek-reasoner generation prompt.
# Map to actual INTENT_NAMES used by classifier (intent_classifier.py:23-32).
LABEL_TO_INTENT = {
    "product_query": "product_query",
    "consultation": "consultation",
    "historical_request": "historical_request",
    "persuasion": "describe",       # «как ответить», «как объяснить»
    "underspec": "underspec",
    "general": None,                 # skip — generic
    # Дополнительные labels из v3 (могут встретиться):
    "smeta_request": "smeta_request",
    "bundle_query": "bundle_query",
    "objection_arguments": "objection_arguments",
    "discovery_assist": "discovery_assist",
    "category_clarify": "category_clarify",
    "referential": "referential",
    "describe": "describe",
    "out_of_scope": "out_of_scope",
    "financial_modifier": "financial_modifier",
    "visualization": "visualization",
    "empty_context_smeta": "empty_context_smeta",
}

# Topic → intent fallback when expected_intent missing or is "general".
# Используем topic для лучшей категоризации.
TOPIC_TO_INTENT_FALLBACK = {
    # client topics
    "design_logo": "product_query",
    "design_brand": "product_query",
    "ad_signboard": "product_query",
    "ad_banner": "product_query",
    "ad_misc": "product_query",
    "print_visitka": "product_query",
    "print_flyer": "product_query",
    "print_misc": "product_query",
    "merch": "product_query",
    "stickers": "product_query",
    "consultation": "consultation",
    "objections": "describe",
    # manager topics
    "objection_handling": "objection_arguments",
    "smeta_assist": "smeta_request",
    "historical_links": "historical_request",
    "upsell": "bundle_query",
    "discovery": "discovery_assist",
    "signage_specs": "product_query",
    "vocabulary": "consultation",
    "directions": "consultation",
    "pricing_justify": "describe",
    "logistics": "consultation",
}

CAP_PER_INTENT = 60
DEDUP_THRESHOLD = 0.85


def load_existing() -> dict:
    if not EXISTING_YAML.exists():
        print(f"!! existing yaml not found: {EXISTING_YAML}")
        return {}
    with EXISTING_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def collect_candidates() -> dict[str, list[str]]:
    """Read all questions_*.jsonl, return {intent_name: [unique queries]}."""
    by_intent: dict[str, list[str]] = defaultdict(list)
    seen_queries: set[str] = set()

    files = sorted(DATA.glob("questions_*.jsonl"))
    for p in files:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = (row.get("question") or "").strip()
            if not q or q in seen_queries or len(q) < 10 or len(q) > 300:
                continue
            seen_queries.add(q)

            label = row.get("expected_intent") or ""
            intent = LABEL_TO_INTENT.get(label)
            # Fallback: if general or unknown, use topic mapping
            if intent is None:
                topic = row.get("topic") or ""
                intent = TOPIC_TO_INTENT_FALLBACK.get(topic)
            if not intent:
                continue
            by_intent[intent].append(q)
        print(f"  read {p.name}: total candidates so far = {sum(len(v) for v in by_intent.values())}")

    return dict(by_intent)


def dedup_with_existing(
    by_intent: dict[str, list[str]],
    existing: dict[str, list[str]],
    embed_fn,
) -> dict[str, list[str]]:
    """For each intent, dedup new candidates against existing + own (cosine > THRESH).

    Returns merged {intent → list[query]} respecting CAP_PER_INTENT.
    """
    import numpy as np

    out: dict[str, list[str]] = {}
    all_intents = set(by_intent.keys()) | set(existing.keys())

    for intent in sorted(all_intents):
        cands = list(existing.get(intent, []))  # existing first (kept)
        new_q = by_intent.get(intent, [])

        if not new_q:
            out[intent] = cands[:CAP_PER_INTENT]
            print(f"  {intent}: existing={len(cands)}, no new → {len(out[intent])}")
            continue

        # Embed everything once
        all_q = cands + new_q
        embeddings = np.stack([
            np.array(embed_fn(q), dtype=np.float32) for q in all_q
        ])
        # BGE-M3 is already L2-normalized

        # Greedy dedup: walk through, keep if not too similar to anything kept
        kept_indices: list[int] = []
        for i in range(len(all_q)):
            if i < len(cands):
                # existing prototypes always kept
                kept_indices.append(i)
                continue
            vec = embeddings[i]
            if not kept_indices:
                kept_indices.append(i)
            else:
                sims = embeddings[kept_indices] @ vec
                if float(sims.max()) < DEDUP_THRESHOLD:
                    kept_indices.append(i)
            if len(kept_indices) >= CAP_PER_INTENT:
                break

        kept_queries = [all_q[i] for i in kept_indices]
        out[intent] = kept_queries
        print(f"  {intent}: existing={len(cands)}, new_input={len(new_q)} → kept={len(kept_queries)}")

    return out


def write_yaml(by_intent: dict[str, list[str]]):
    lines = [
        "# Intent prototypes for BGE-M3 embedding-based classification (Tier 2).",
        "# Each intent has 30-60 prototype queries. At startup, these are embedded",
        "# and used for cosine-similarity matching against user queries.",
        "# Threshold: best match > 0.75 → classify as that intent.",
        "#",
        "# P14.5 expansion: from 58 prototypes (10 intents) → 500+ via v1/v2/v3 labelled",
        "# question data. Dedup via BGE-M3 cosine (sim>0.85 drop), cap 60 per intent.",
        "",
    ]
    for intent in sorted(by_intent.keys()):
        queries = by_intent[intent]
        if not queries:
            continue
        lines.append(f"{intent}:")
        for q in queries:
            # Use single-quoted YAML string with internal '' escape
            escaped = q.replace("'", "''")
            lines.append(f"  - '{escaped}'")
        lines.append("")
    OUT_YAML.write_text("\n".join(lines), encoding="utf-8")
    total = sum(len(v) for v in by_intent.values())
    print(f"\n✓ Wrote {len(by_intent)} intents, {total} total prototypes → {OUT_YAML}")


def main():
    sys.path.insert(0, str(ROOT / "RAG_RUNTIME"))
    # Load BGE-M3 (this takes ~10-30s on first run)
    from sentence_transformers import SentenceTransformer
    from app.config import Settings
    settings = Settings()
    print(f"Loading BGE-M3 from {settings.embedding_model_full_path}...")
    model = SentenceTransformer(str(settings.embedding_model_full_path))

    def embed_fn(q: str):
        v = model.encode([q], normalize_embeddings=True)[0]
        return v.tolist()

    print("\nReading existing prototypes...")
    existing = load_existing()
    print(f"  Found {sum(len(v) for v in existing.values())} prototypes across {len(existing)} intents")

    print("\nCollecting candidates from questions_*.jsonl...")
    candidates = collect_candidates()
    print(f"  Total new candidates: {sum(len(v) for v in candidates.values())}")

    print("\nDeduping + capping...")
    merged = dedup_with_existing(candidates, existing, embed_fn)

    write_yaml(merged)


if __name__ == "__main__":
    main()
