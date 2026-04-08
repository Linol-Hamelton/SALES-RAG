#!/usr/bin/env python3
"""
Embed smeta category `keywords_text` with BGE-M3 for semantic category lookup.

Reads:  RAG_ANALYTICS/output/smeta_templates.json
Writes: RAG_ANALYTICS/output/smeta_category_embeddings.npy
        (float32 matrix, shape (N_categories, 1024), row order matches JSON)

Run after buildSmetaTemplates.mjs during analytics pipeline. Called manually
for П7 bootstrap, integrated into runAll.mjs in a later iteration.

Usage:
    python scripts/embed_smeta_categories.py [--device cuda] [--batch-size 32]
"""
import json
import sys
import click
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ANALYTICS_OUTPUT = PROJECT_ROOT.parent / "RAG_ANALYTICS" / "output"
TEMPLATES_PATH = ANALYTICS_OUTPUT / "smeta_templates.json"
EMBEDDINGS_PATH = ANALYTICS_OUTPUT / "smeta_category_embeddings.npy"

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "embeddings" / "BAAI_bge-m3"


@click.command()
@click.option("--device", default="cuda", help="Device: cuda or cpu")
@click.option("--batch-size", default=32, help="Embedding batch size")
@click.option("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to BGE-M3 model")
def main(device: str, batch_size: int, model_path: str):
    if not TEMPLATES_PATH.exists():
        click.echo(f"ERROR: {TEMPLATES_PATH} not found. Run buildSmetaTemplates.mjs first.", err=True)
        sys.exit(1)

    click.echo(f"Loading {TEMPLATES_PATH}...")
    with open(TEMPLATES_PATH, encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    click.echo(f"  {len(categories)} categories loaded")

    if not categories:
        click.echo("ERROR: no categories to embed", err=True)
        sys.exit(1)

    # Build input texts: category_name + keywords + first canonical position names
    # This gives BGE-M3 more semantic anchors per category.
    texts = []
    for cat in categories:
        parts = [cat.get("category_name", "")]
        keywords = cat.get("keywords", [])
        if keywords:
            parts.append(", ".join(keywords))
        positions = cat.get("canonical_smeta", {}).get("positions", [])
        pos_names = [p.get("product_name", "") for p in positions[:5]]
        if pos_names:
            parts.append("компоненты: " + "; ".join(pos_names))
        texts.append(". ".join(p for p in parts if p))

    click.echo(f"Sample input texts:")
    for t in texts[:3]:
        click.echo(f"  {t[:120]}")

    click.echo(f"\nLoading BGE-M3 from {model_path} on {device}...")
    from sentence_transformers import SentenceTransformer
    model_p = Path(model_path)
    if model_p.exists():
        model = SentenceTransformer(str(model_p), device=device)
    else:
        click.echo(f"  Local path not found, downloading BAAI/bge-m3...")
        model = SentenceTransformer("BAAI/bge-m3", device=device)
    dim = model.get_sentence_embedding_dimension()
    click.echo(f"  Model loaded (dim={dim})")

    click.echo(f"\nEmbedding {len(texts)} category texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    click.echo(f"  Shape: {embeddings.shape}")

    np.save(EMBEDDINGS_PATH, embeddings)
    click.echo(f"\n✓ Saved → {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
