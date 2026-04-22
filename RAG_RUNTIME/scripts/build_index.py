#!/usr/bin/env python3
"""
Build vector index: embed docs → upsert into Qdrant natively (dense + sparse).

Usage:
    python scripts/build_index.py [--batch-size 64] [--device cuda] [--recreate]
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Any
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.text import tokenize_ru


SKIP_FILES = {"photo_analysis_raw.jsonl", "bridge_unresolved.jsonl"}  # raw/debug files, not for indexing


def load_all_docs(data_dir: Path) -> list[dict]:
    """Load all JSONL files from data directory."""
    docs = []
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    for f in jsonl_files:
        if f.name in SKIP_FILES:
            print(f"  Skipping {f.name} (raw source)")
            continue
        print(f"  Loading {f.name}...")
        with open(f, encoding="utf-8") as fh:
            file_docs = [json.loads(line) for line in fh if line.strip()]
        print(f"    -> {len(file_docs)} docs")
        docs.extend(file_docs)
    return docs


def load_model(model_path: Path, device: str):
    """Load BGE-M3 embedding model."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model from {model_path} on {device}...")

    if model_path.exists():
        model = SentenceTransformer(str(model_path), device=device)
    else:
        print(f"  Model not found locally, downloading BAAI/bge-m3...")
        model_path.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(
            "BAAI/bge-m3",
            device=device,
            cache_folder=str(model_path.parent)
        )
    print(f"  Model loaded (dim={model.get_sentence_embedding_dimension()})")
    return model


def embed_docs(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed texts using BGE-M3 (no prefix for documents)."""
    print(f"Embedding {len(texts)} texts (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


def upsert_to_qdrant(docs: list[dict], embeddings: np.ndarray, qdrant_url: str, collection: str, recreate: bool):
    """Upsert documents and embeddings to Qdrant (Native Sparse + Dense)."""
    import mmh3
    from collections import Counter
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType, SparseVectorParams, Modifier, SparseVector

    client = QdrantClient(url=qdrant_url, timeout=60)
    dim = embeddings.shape[1]

    print(f"Connecting to Qdrant at {qdrant_url}...")
    try:
        info = client.get_collections()
        print(f"  Connected. Collections: {[c.name for c in info.collections]}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to Qdrant: {e}", file=sys.stderr)
        sys.exit(1)

    collections = [c.name for c in client.get_collections().collections]
    if collection in collections:
        if recreate:
            print(f"  Deleting existing collection '{collection}'...")
            client.delete_collection(collection)
        else:
            print(f"  Collection '{collection}' exists (use --recreate to rebuild)")

    if collection not in [c.name for c in client.get_collections().collections]:
        print(f"  Creating collection '{collection}' (Hybrid: dense dim={dim}, lexical IDF)...")
        client.create_collection(
            collection_name=collection,
            vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={"lexical": SparseVectorParams(modifier=Modifier.IDF)}
        )

    print("  Creating payload indexes...")
    # P10.6 C2: расширенный список KEYWORD-индексов для cross-link фильтров
    # (offer_id IN [...], linked_product_ids IN [...] и т.п.) — без индекса
    # Qdrant падает в full-scan и latency растёт линейно с числом docs.
    indexed_fields = [
        "doc_type", "direction", "price_mode", "confidence_tier",
        "category", "roadmap_title",
        # Cross-link id-поля (P10.6):
        "offer_id", "good_ids",
        "linked_product_ids", "linked_smeta_category_ids",
        "related_roadmap_slugs", "roadmap_slug",
        "source_file",
    ]
    for field in indexed_fields:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    BATCH_SIZE = 256
    print(f"Upserting {len(docs)} points to Qdrant...")

    for start in tqdm(range(0, len(docs), BATCH_SIZE), desc="Upserting"):
        batch_docs = docs[start:start + BATCH_SIZE]
        batch_embeddings = embeddings[start:start + BATCH_SIZE]

        points = []
        for i, (doc, emb) in enumerate(zip(batch_docs, batch_embeddings)):
            # Ingest scripts inconsistently use "metadata" or "payload" as the
            # outer field name. Read both so neither set silently loses fields
            # in Qdrant. payload wins on conflict (newer convention).
            payload = dict(doc.get("metadata", {}))
            payload.update(doc.get("payload", {}))
            payload["doc_type"] = doc["doc_type"]
            payload["doc_id"] = doc["doc_id"]

            full_text = doc.get("searchable_text", "")
            payload["searchable_text"] = full_text[:4000]

            tokens = tokenize_ru(full_text)
            counts = Counter(tokens)
            indices = []
            values = []
            if counts:
                for token, count in counts.items():
                    indices.append(mmh3.hash(token, seed=42, signed=False))
                    values.append(float(count))

            vec_dict = {"dense": emb.tolist()}
            if indices:
                vec_dict["lexical"] = SparseVector(indices=indices, values=values)

            points.append(PointStruct(
                id=start + i,
                vector=vec_dict,
                payload=payload,
            ))

        client.upsert(collection_name=collection, points=points)

    count = client.count(collection).count
    print(f"  Qdrant collection '{collection}': {count} points")


def main():
    """Build Qdrant vector index from canonical JSONL docs."""
    parser = argparse.ArgumentParser(description="Build Qdrant vector index from JSONL docs")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"), help="Directory with JSONL files")
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "embeddings" / "BAAI_bge-m3"), help="Embedding model path")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant server URL")
    parser.add_argument("--collection", default="labus_docs", help="Qdrant collection name")
    parser.add_argument("--batch-size", default=64, type=int, help="Embedding batch size")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device for embeddings")
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection if exists")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant upsert")
    parser.add_argument("--from-npy", default=None,
                        help="Пропустить embedding: прочитать готовый data/embeddings.npy (или указанный путь) "
                             "и только выполнить upsert в Qdrant. Заменяет удалённый upload_to_qdrant.py.")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    device = args.device

    # Load docs
    print(f"\nLoading docs from {data_path}...")
    docs = load_all_docs(data_path)
    if not docs:
        print("ERROR: No docs found. Run `python scripts/ingest.py` first.", file=sys.stderr)
        sys.exit(1)
    print(f"  Total docs: {len(docs)}")

    # P10.6 C1: --from-npy пропускает embed-фазу, читает готовый .npy
    # (раньше это делал удалённый upload_to_qdrant.py)
    if args.from_npy:
        emb_path = Path(args.from_npy) if args.from_npy != "-" else (data_path / "embeddings.npy")
        if not emb_path.exists():
            print(f"ERROR: --from-npy path not found: {emb_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading embeddings from {emb_path}...")
        embeddings = np.load(str(emb_path))
        print(f"  Loaded embeddings shape: {embeddings.shape}")
        if embeddings.shape[0] != len(docs):
            print(f"ERROR: embeddings {embeddings.shape[0]} ≠ docs {len(docs)}. "
                  f"Re-embed or sync jsonl/npy.", file=sys.stderr)
            sys.exit(1)
    else:
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Embed
        model_path_obj = Path(args.model_path)
        model = load_model(model_path_obj, device)
        texts = [doc["searchable_text"] for doc in docs]
        embeddings = embed_docs(model, texts, batch_size=args.batch_size)

        # Save embeddings for VPS transfer
        emb_path = data_path / "embeddings.npy"
        np.save(str(emb_path), embeddings)
        print(f"  Saved embeddings to {emb_path} ({embeddings.shape})")

    # Upsert to Qdrant Native
    if not args.skip_qdrant:
        upsert_to_qdrant(docs, embeddings, args.qdrant_url, args.collection, args.recreate)

    print(f"\nIndex build complete!")
    print(f"  Docs indexed: {len(docs)}")
    print(f"  Qdrant: {args.qdrant_url} / {args.collection}")


if __name__ == "__main__":
    main()
