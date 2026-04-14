"""Targeted upsert for service_pricing_bridge documents.

Reads /app/data/bridge_docs.jsonl, embeds via the same BGE-M3 used at runtime,
upserts to Qdrant with stable UUID ids (no collisions with existing int-ids).
Idempotent — re-running overwrites by same UUID.

Run inside labus_api container:
    docker exec labus_api python /tmp/upsert_bridges.py
"""
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/app")

import mmh3
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from app.config import settings
from app.utils.text import tokenize_ru

BRIDGE_PATH = Path("/app/data/bridge_docs.jsonl")
NAMESPACE = uuid.UUID("6f1e7b9a-7b4e-4c3a-a9b7-b21d6e000000")  # stable namespace (hex-only)

def load_bridges():
    if not BRIDGE_PATH.exists():
        print(f"ERROR: {BRIDGE_PATH} missing")
        sys.exit(1)
    docs = []
    with open(BRIDGE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def embed_texts(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model_path = Path("/app/models/embeddings/BAAI_bge-m3")
    model = SentenceTransformer(str(model_path), device="cpu")
    emb = model.encode(
        texts,
        batch_size=4,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return emb


def main():
    docs = load_bridges()
    print(f"Loaded {len(docs)} bridge docs from {BRIDGE_PATH}")
    for d in docs:
        print(f"  {d['doc_id']} → service={d['metadata'].get('service')!r}")

    texts = [d.get("searchable_text", "") for d in docs]
    print("\nEmbedding on CPU (4 docs, ~10s)...")
    embeddings = embed_texts(texts)
    print(f"  shape={embeddings.shape}")

    client = QdrantClient(url=settings.qdrant_url)
    col = settings.qdrant_collection

    pre_count = client.count(col).count
    print(f"\nPre-upsert total points: {pre_count}")

    points = []
    for doc, emb in zip(docs, embeddings):
        payload = dict(doc.get("metadata", {}))
        payload["doc_type"] = doc["doc_type"]
        payload["doc_id"] = doc["doc_id"]
        text = doc.get("searchable_text", "")
        payload["searchable_text"] = text[:4000]

        # Sparse (BM25-like)
        counts = Counter(tokenize_ru(text))
        indices, values = [], []
        for tok, cnt in counts.items():
            indices.append(mmh3.hash(tok, seed=42, signed=False))
            values.append(float(cnt))

        vec_dict = {"dense": emb.tolist()}
        if indices:
            vec_dict["lexical"] = SparseVector(indices=indices, values=values)

        # Stable UUID per doc_id
        pid = str(uuid.uuid5(NAMESPACE, doc["doc_id"]))
        points.append(PointStruct(id=pid, vector=vec_dict, payload=payload))
        print(f"  prepared pid={pid} doc_id={doc['doc_id']}")

    client.upsert(collection_name=col, points=points)
    print(f"\nUpserted {len(points)} bridge points.")

    post_count = client.count(col).count
    print(f"Post-upsert total points: {post_count} (delta: {post_count - pre_count})")

    # Verify
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    bridge_total = client.count(
        col,
        count_filter=Filter(must=[FieldCondition(
            key="doc_type", match=MatchValue(value="service_pricing_bridge"))]),
    ).count
    print(f"service_pricing_bridge docs in index now: {bridge_total}")

    for svc in ["Дизайн логотипа", "Дизайн брендбука", "Дизайн фирменного стиля"]:
        res, _ = client.scroll(
            col,
            scroll_filter=Filter(must=[
                FieldCondition(key="doc_type",
                               match=MatchValue(value="service_pricing_bridge")),
                FieldCondition(key="service", match=MatchValue(value=svc)),
            ]),
            limit=1, with_payload=True,
        )
        print(f"  service={svc!r} → {len(res)} hits")


if __name__ == "__main__":
    main()
