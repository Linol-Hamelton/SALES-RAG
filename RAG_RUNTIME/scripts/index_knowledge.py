#!/usr/bin/env python3
"""
Incrementally index new knowledge docs (faq + knowledge) into existing Qdrant collection.
Uses current Qdrant count as starting point ID — safe to run without recreating collection.
Also rebuilds BM25 over ALL docs (needed for corpus-level scores).

Usage:
    python scripts/index_knowledge.py [--batch-size 64]
"""
import json
import pickle
import sys
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "indexes"
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.text import tokenize_ru

QDRANT_URL = "http://localhost:6333"
COLLECTION = "labus_docs"
MODEL_PATH = PROJECT_ROOT / "models" / "embeddings" / "BAAI_bge-m3"


def load_new_docs() -> list[dict]:
    """Load only the new knowledge JSONL files."""
    new_files = ["faq_docs.jsonl", "knowledge_docs.jsonl"]
    docs = []
    for fname in new_files:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        with open(path, encoding="utf-8") as f:
            file_docs = [json.loads(line) for line in f if line.strip()]
        print(f"  {fname}: {len(file_docs)} docs")
        docs.extend(file_docs)
    return docs


def load_all_docs() -> list[dict]:
    """Load ALL JSONL files for BM25 rebuild."""
    docs = []
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            file_docs = [json.loads(line) for line in fh if line.strip()]
        docs.extend(file_docs)
    return docs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load new docs
    print("\n=== Loading new knowledge docs ===")
    new_docs = load_new_docs()
    print(f"Total new docs: {len(new_docs)}")

    # 2. Embed new docs
    print("\n=== Embedding new docs with BGE-M3 ===")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(MODEL_PATH), device=device)
    texts = [doc["searchable_text"] for doc in new_docs]
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  Embeddings: {embeddings.shape}")

    # 3. Upsert to Qdrant with IDs starting after existing count
    print("\n=== Upserting to Qdrant ===")
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    client = QdrantClient(url=QDRANT_URL, timeout=60)

    # Delete ALL existing faq + knowledge points before re-indexing
    # (avoids duplicates when re-running after ingest_knowledge.py)
    from qdrant_client.models import Filter, FieldCondition, MatchAny, FilterSelector
    print("  Deleting old faq + knowledge points...")
    client.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="doc_type", match=MatchAny(any=["faq", "knowledge"]))]
            )
        ),
    )
    existing_count = client.count(COLLECTION).count
    print(f"  Points after cleanup: {existing_count}")

    BATCH = 256
    for start in tqdm(range(0, len(new_docs), BATCH), desc="Upserting"):
        batch_docs = new_docs[start:start + BATCH]
        batch_embs = embeddings[start:start + BATCH]

        points = []
        for i, (doc, emb) in enumerate(zip(batch_docs, batch_embs)):
            # Use payload directly if present, otherwise build from doc fields
            payload = doc.get("payload", {})
            if not payload:
                payload = {
                    "doc_type": doc["doc_type"],
                    "doc_id": doc["doc_id"],
                    "searchable_text": doc["searchable_text"][:2000],
                }
            payload["doc_type"] = doc["doc_type"]
            payload["doc_id"] = doc["doc_id"]
            payload["searchable_text"] = doc["searchable_text"][:2000]

            points.append(PointStruct(
                id=existing_count + start + i,
                vector=emb.tolist(),
                payload=payload,
            ))

        client.upsert(collection_name=COLLECTION, points=points)

    new_count = client.count(COLLECTION).count
    print(f"  New total points: {new_count} (+{new_count - existing_count})")

    # 4. Rebuild BM25 over ALL docs
    print("\n=== Rebuilding BM25 (all docs) ===")
    from rank_bm25 import BM25Okapi
    all_docs = load_all_docs()
    print(f"  Total docs for BM25: {len(all_docs)}")

    corpus = []
    doc_ids = []
    for doc in tqdm(all_docs, desc="Tokenizing"):
        tokens = tokenize_ru(doc.get("searchable_text", ""))
        corpus.append(tokens)
        doc_ids.append(doc["doc_id"])

    bm25 = BM25Okapi(corpus)
    bm25_path = INDEX_DIR / "bm25_state.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids}, f)
    print(f"  BM25 saved: {bm25_path.stat().st_size / 1024:.0f} KB, {len(doc_ids)} docs")

    print("\n=== Done! Restart the server to load new BM25 state. ===")


if __name__ == "__main__":
    main()
