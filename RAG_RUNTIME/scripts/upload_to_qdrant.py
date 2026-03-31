#!/usr/bin/env python3
"""
Upload pre-computed embeddings to Qdrant. No GPU needed.
Reads JSONL docs + embeddings.npy, creates collection, upserts.

Usage (on VPS):
    python /app/scripts/upload_to_qdrant.py
"""
import json
import sys
import re
import numpy as np
from pathlib import Path
from collections import Counter

import mmh3
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, SparseVectorParams, Modifier, SparseVector,
)


def tokenize_ru(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower()
    tokens = re.split(r"[^\w]+", text)
    return [t for t in tokens if len(t) >= 2]


def main():
    data_dir = Path("/app/data") if Path("/app/data").exists() else Path("data")
    qdrant_url = "http://qdrant:6333" if "KUBERNETES" not in str(Path("/app")) else "http://localhost:6333"
    # Auto-detect environment
    if not Path("/app").exists():
        qdrant_url = "http://localhost:6333"
    else:
        qdrant_url = "http://qdrant:6333"

    collection = "labus_docs"

    # Load docs
    docs = []
    for f in sorted(data_dir.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            file_docs = [json.loads(line) for line in fh if line.strip()]
        print(f"  {f.name}: {len(file_docs)}")
        docs.extend(file_docs)
    print(f"Total docs: {len(docs)}")

    # Load embeddings
    emb_path = data_dir / "embeddings.npy"
    if not emb_path.exists():
        print(f"ERROR: {emb_path} not found. Run embedding on GPU machine first.")
        sys.exit(1)

    embeddings = np.load(str(emb_path))
    print(f"Embeddings: {embeddings.shape}")

    if len(docs) != embeddings.shape[0]:
        print(f"ERROR: doc count ({len(docs)}) != embedding count ({embeddings.shape[0]})")
        sys.exit(1)

    dim = embeddings.shape[1]

    # Connect to Qdrant
    client = QdrantClient(url=qdrant_url, timeout=60)
    print(f"Connected to Qdrant at {qdrant_url}")

    # Recreate collection
    cols = [c.name for c in client.get_collections().collections]
    if collection in cols:
        print(f"Deleting existing '{collection}'...")
        client.delete_collection(collection)

    print(f"Creating '{collection}' (dense={dim} + sparse IDF)...")
    client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={"lexical": SparseVectorParams(modifier=Modifier.IDF)},
    )

    for field in ["doc_type", "direction", "price_mode", "confidence_tier"]:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    # Upsert
    BATCH = 256
    print(f"Upserting {len(docs)} points...")
    for start in range(0, len(docs), BATCH):
        batch_docs = docs[start : start + BATCH]
        batch_emb = embeddings[start : start + BATCH]
        points = []
        for i, (doc, emb) in enumerate(zip(batch_docs, batch_emb)):
            payload = dict(doc.get("metadata", {}))
            payload["doc_type"] = doc["doc_type"]
            payload["doc_id"] = doc["doc_id"]
            full_text = doc.get("searchable_text", "")
            payload["searchable_text"] = full_text[:2000]

            tokens = tokenize_ru(full_text)
            counts = Counter(tokens)
            indices = []
            values = []
            for token, count in counts.items():
                indices.append(mmh3.hash(token, seed=42, signed=False))
                values.append(float(count))

            vec_dict = {"dense": emb.tolist()}
            if indices:
                vec_dict["lexical"] = SparseVector(indices=indices, values=values)

            points.append(PointStruct(id=start + i, vector=vec_dict, payload=payload))

        client.upsert(collection_name=collection, points=points)
        end = start + len(batch_docs)
        print(f"  {end}/{len(docs)}")

    count = client.count(collection).count
    print(f"\nDone! Collection '{collection}': {count} points")


if __name__ == "__main__":
    main()
