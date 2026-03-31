#!/usr/bin/env python3
"""
Build vector index: embed docs → upsert into Qdrant natively (dense + sparse).

Usage:
    python scripts/build_index.py [--batch-size 64] [--device cuda] [--recreate]
"""
import json
import sys
import click
import numpy as np
from pathlib import Path
from typing import Any
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.text import tokenize_ru


def load_all_docs(data_dir: Path) -> list[dict]:
    """Load all JSONL files from data directory."""
    docs = []
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    for f in jsonl_files:
        click.echo(f"  Loading {f.name}...")
        with open(f, encoding="utf-8") as fh:
            file_docs = [json.loads(line) for line in fh if line.strip()]
        click.echo(f"    -> {len(file_docs)} docs")
        docs.extend(file_docs)
    return docs


def load_model(model_path: Path, device: str):
    """Load BGE-M3 embedding model."""
    from sentence_transformers import SentenceTransformer
    click.echo(f"Loading embedding model from {model_path} on {device}...")

    if model_path.exists():
        model = SentenceTransformer(str(model_path), device=device)
    else:
        click.echo(f"  Model not found locally, downloading BAAI/bge-m3...")
        model_path.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(
            "BAAI/bge-m3",
            device=device,
            cache_folder=str(model_path.parent)
        )
    click.echo(f"  Model loaded (dim={model.get_sentence_embedding_dimension()})")
    return model


def embed_docs(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed texts using BGE-M3 (no prefix for documents)."""
    click.echo(f"Embedding {len(texts)} texts (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    click.echo(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


def upsert_to_qdrant(docs: list[dict], embeddings: np.ndarray, qdrant_url: str, collection: str, recreate: bool):
    """Upsert documents and embeddings to Qdrant (Native Sparse + Dense)."""
    import mmh3
    from collections import Counter
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType, SparseVectorParams, Modifier, SparseVector

    client = QdrantClient(url=qdrant_url, timeout=60)
    dim = embeddings.shape[1]

    click.echo(f"Connecting to Qdrant at {qdrant_url}...")
    try:
        info = client.get_collections()
        click.echo(f"  Connected. Collections: {[c.name for c in info.collections]}")
    except Exception as e:
        click.echo(f"  ERROR: Cannot connect to Qdrant: {e}", err=True)
        sys.exit(1)

    collections = [c.name for c in client.get_collections().collections]
    if collection in collections:
        if recreate:
            click.echo(f"  Deleting existing collection '{collection}'...")
            client.delete_collection(collection)
        else:
            click.echo(f"  Collection '{collection}' exists (use --recreate to rebuild)")

    if collection not in [c.name for c in client.get_collections().collections]:
        click.echo(f"  Creating collection '{collection}' (Hybrid: dense dim={dim}, lexical IDF)...")
        client.create_collection(
            collection_name=collection,
            vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={"lexical": SparseVectorParams(modifier=Modifier.IDF)}
        )

    click.echo("  Creating payload indexes...")
    for field in ["doc_type", "direction", "price_mode", "confidence_tier"]:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    BATCH_SIZE = 256
    click.echo(f"Upserting {len(docs)} points to Qdrant...")

    for start in tqdm(range(0, len(docs), BATCH_SIZE), desc="Upserting"):
        batch_docs = docs[start:start + BATCH_SIZE]
        batch_embeddings = embeddings[start:start + BATCH_SIZE]

        points = []
        for i, (doc, emb) in enumerate(zip(batch_docs, batch_embeddings)):
            payload = dict(doc.get("metadata", {}))
            payload["doc_type"] = doc["doc_type"]
            payload["doc_id"] = doc["doc_id"]
            
            full_text = doc.get("searchable_text", "")
            payload["searchable_text"] = full_text[:2000]

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
    click.echo(f"  Qdrant collection '{collection}': {count} points")


@click.command()
@click.option("--data-dir", default=str(PROJECT_ROOT / "data"), help="Directory with JSONL files")
@click.option("--model-path", default=str(PROJECT_ROOT / "models" / "embeddings" / "BAAI_bge-m3"), help="Embedding model path")
@click.option("--qdrant-url", default="http://localhost:6333", help="Qdrant server URL")
@click.option("--collection", default="labus_docs", help="Qdrant collection name")
@click.option("--batch-size", default=64, type=int, help="Embedding batch size")
@click.option("--device", default="auto", type=click.Choice(["auto", "cuda", "cpu"]), help="Device for embeddings")
@click.option("--recreate", is_flag=True, help="Recreate Qdrant collection if exists")
@click.option("--skip-qdrant", is_flag=True, help="Skip Qdrant upsert")
def main(data_dir, model_path, qdrant_url, collection, batch_size, device, recreate, skip_qdrant):
    """Build Qdrant vector index from canonical JSONL docs."""
    import torch

    data_path = Path(data_dir)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Using device: {device}")

    # Load docs
    click.echo(f"\nLoading docs from {data_path}...")
    docs = load_all_docs(data_path)
    if not docs:
        click.echo("ERROR: No docs found. Run `python scripts/ingest.py` first.", err=True)
        sys.exit(1)
    click.echo(f"  Total docs: {len(docs)}")

    # Embed
    model_path_obj = Path(model_path)
    model = load_model(model_path_obj, device)
    texts = [doc["searchable_text"] for doc in docs]
    embeddings = embed_docs(model, texts, batch_size=batch_size)

    # Save embeddings for VPS transfer
    emb_path = data_path / "embeddings.npy"
    np.save(str(emb_path), embeddings)
    click.echo(f"  Saved embeddings to {emb_path} ({embeddings.shape})")

    # Upsert to Qdrant Native
    if not skip_qdrant:
        upsert_to_qdrant(docs, embeddings, qdrant_url, collection, recreate)

    click.echo(f"\nIndex build complete!")
    click.echo(f"  Docs indexed: {len(docs)}")
    click.echo(f"  Qdrant: {qdrant_url} / {collection}")


if __name__ == "__main__":
    main()
