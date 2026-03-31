#!/usr/bin/env python3
"""
CLI for running queries locally without the FastAPI server.
Useful for debugging retrieval and generation.

Usage:
    python scripts/query_cli.py --query "световая вывеска кофейня"
    python scripts/query_cli.py --query "безнал 10%" --mode structured
    python scripts/query_cli.py --query "баннер 3х6" --show-context
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.utils.logging import setup_logging
from app.core.retriever import HybridRetriever
from app.core.reranker import CrossEncoderReranker
from app.core.generator import DeepseekGenerator
from app.core.pricing_resolver import PricingResolver

setup_logging("WARNING")  # Suppress info logs in CLI for cleaner output


@click.command()
@click.option("--query", "-q", required=True, help="Query text in Russian or English")
@click.option("--mode", "-m", type=click.Choice(["human", "structured"]), default="human",
              help="Response mode: human (text) or structured (JSON)")
@click.option("--top-k", "-k", default=8, type=int, help="Number of results to retrieve")
@click.option("--show-context", is_flag=True, help="Show retrieved context docs")
@click.option("--no-generate", is_flag=True, help="Skip LLM generation, show only retrieved docs")
def main(query, mode, top_k, show_context, no_generate):
    """Run a RAG query locally."""
    asyncio.run(_run(query, mode, top_k, show_context, no_generate))


async def _run(query, mode, top_k, show_context, no_generate):
    click.echo(f"\n{'='*60}")
    click.echo(f"Query: {query}")
    click.echo(f"Mode: {mode} | top_k: {top_k}")
    click.echo('='*60)

    # Initialize components
    retriever = HybridRetriever(settings)
    reranker = CrossEncoderReranker(settings)
    pricing = PricingResolver(settings)

    # Load
    click.echo("\nLoading retriever...")
    t0 = time.monotonic()
    retriever.load()
    reranker.load()
    click.echo(f"  Loaded in {(time.monotonic()-t0)*1000:.0f}ms")

    # Retrieve
    click.echo(f"\nRetrieving (top {top_k*2})...")
    t1 = time.monotonic()
    candidates = retriever.retrieve(query, top_k=top_k * 2)
    click.echo(f"  Retrieved {len(candidates)} candidates in {(time.monotonic()-t1)*1000:.0f}ms")

    if not candidates:
        click.echo("\n✗ No results found.")
        return

    # Rerank
    click.echo(f"\nReranking...")
    t2 = time.monotonic()
    reranked = reranker.rerank(query, candidates, top_n=top_k)
    click.echo(f"  Reranked to {len(reranked)} in {(time.monotonic()-t2)*1000:.0f}ms")

    # Pricing
    pr = pricing.resolve(reranked)
    click.echo(f"\nPricing: confidence={pr.confidence}, value={pr.estimated_value}")
    if pr.flags:
        for flag in pr.flags:
            click.echo(f"  ⚠ {flag}")

    # Show context
    if show_context:
        click.echo("\n--- Retrieved Context ---")
        for i, doc in enumerate(reranked):
            payload = doc.get("payload", {})
            score = doc.get("final_score", doc.get("rrf_score", 0))
            click.echo(f"\n[{i+1}] {payload.get('doc_type','?')} | score={score:.4f}")
            click.echo(f"    {payload.get('searchable_text','')[:200]}")
        click.echo("-" * 40)

    if no_generate:
        return

    # Generate
    if not settings.deepseek_api_key or settings.deepseek_api_key.startswith("sk-your"):
        click.echo("\n⚠ No valid DEEPSEEK_API_KEY. Skipping generation.")
        click.echo(f"\nTop result: {reranked[0]['payload'].get('product_name', reranked[0]['payload'].get('sample_title', '?'))}")
        return

    generator = DeepseekGenerator(settings)
    generator.load()

    click.echo(f"\nGenerating ({mode})...")
    t3 = time.monotonic()

    if mode == "human":
        response = await generator.generate(query, reranked, pr)
        gen_time = (time.monotonic() - t3) * 1000
        click.echo(f"\n{'='*60}")
        click.echo("RESPONSE:")
        click.echo('='*60)
        click.echo(response)
    else:
        raw = await generator.generate_structured(query, reranked, pr)
        gen_time = (time.monotonic() - t3) * 1000
        click.echo(f"\n{'='*60}")
        click.echo("STRUCTURED RESPONSE:")
        click.echo('='*60)
        click.echo(json.dumps(raw, ensure_ascii=False, indent=2))

    total_ms = int((time.monotonic() - t0) * 1000)
    click.echo(f"\nTotal: {total_ms}ms (retrieval+rerank: {int((t3-t0)*1000)}ms, generation: {gen_time:.0f}ms)")


if __name__ == "__main__":
    main()
