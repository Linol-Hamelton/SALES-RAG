"""
Hybrid retriever: Native Qdrant Dense + Sparse (BM25) search with native RRF fusion.
"""
import sys
from pathlib import Path
from typing import Any
import mmh3
from collections import Counter

from app.core.query_parser import ParsedQuery, parse_query
from app.utils.text import tokenize_ru
from app.utils.logging import get_logger

logger = get_logger(__name__)


def generate_sparse_vector(text: str):
    from qdrant_client.models import SparseVector
    tokens = tokenize_ru(text)
    if not tokens:
        return None
    counts = Counter(tokens)
    indices = []
    values = []
    for token, count in counts.items():
        indices.append(mmh3.hash(token, seed=42, signed=False))
        values.append(float(count))
    if not indices:
        return None
    return SparseVector(indices=indices, values=values)


class HybridRetriever:
    """
    Hybrid retriever using Qdrant native hybrid search (dense + sparse tf-idf) + Reciprocal Rank Fusion.
    """

    def __init__(self, settings):
        self.settings = settings
        self._model = None
        self._client = None
        self._loaded = False

    def load(self):
        """Load embedding model and Qdrant client."""
        if self._loaded:
            return

        import torch
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = Path(self.settings.embedding_model_path)
        if model_path.exists():
            logger.info("Loading embedding model", path=str(model_path), device=device)
            self._model = SentenceTransformer(str(model_path), device=device)
        else:
            logger.info("Downloading BAAI/bge-m3", device=device)
            self._model = SentenceTransformer("BAAI/bge-m3", device=device,
                                               cache_folder=str(model_path.parent))
        logger.info("Embedding model loaded", dim=self._model.get_sentence_embedding_dimension())

        logger.info("Connecting to Qdrant", url=self.settings.qdrant_url)
        self._client = QdrantClient(url=self.settings.qdrant_url, timeout=30)
        self._loaded = True
        logger.info("HybridRetriever ready (Native Qdrant Hybrid RRF)")

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._model is not None and self._client is not None

    def embed_query(self, query: str) -> list[float]:
        """Embed query with BGE-M3 prefix."""
        prefix = "Represent this sentence for searching relevant passages: "
        embedding = self._model.encode(
            [prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].tolist()

    def _direction_boost(self, payload: dict, detected_direction: str | None) -> float:
        """Apply soft direction boost to score (applied AFTER RRF internally in Qdrant limits, we do it in Python)."""
        if detected_direction is None:
            return 0.0
        doc_direction = payload.get("direction", "")
        if doc_direction == detected_direction:
            return 0.1
        return 0.0

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Retrieve top-k relevant docs using Qdrant Native RRF fusion.
        """
        if not self._loaded:
            self.load()

        from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, FusionQuery, Fusion

        top_k = top_k or self.settings.retrieval_top_k
        parsed = parse_query(query)

        logger.info("Retrieving", query=query[:100], direction=parsed.direction,
                    intent=parsed.intent, top_k=top_k)

        query_vec = self.embed_query(query)
        sparse_vec = generate_sparse_vector(query)

        query_filter = None
        if parsed.intent not in ("consulting", "general", "timeline") and \
           parsed.high_confidence_direction and parsed.direction != "Безнал":
            query_filter = Filter(must=[FieldCondition(key="direction", match=MatchValue(value=parsed.direction))])

        try:
            if sparse_vec is None:
                db_results = self._client.query_points(
                    collection_name=self.settings.qdrant_collection,
                    query=query_vec,
                    using="dense",
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True
                ).points
            else:
                prefetch = [
                    Prefetch(query=query_vec, using="dense", limit=top_k * 2, filter=query_filter),
                    Prefetch(query=sparse_vec, using="lexical", limit=top_k * 2, filter=query_filter)
                ]
                db_results = self._client.query_points(
                    collection_name=self.settings.qdrant_collection,
                    prefetch=prefetch,
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                    with_payload=True
                ).points
        except Exception as e:
            logger.error("Qdrant hybrid search failed", error=str(e))
            # Fallback (no filter if filter broke DB)
            try:
                db_results = self._client.query_points(
                    collection_name=self.settings.qdrant_collection,
                    query=query_vec,
                    using="dense",
                    limit=top_k,
                    with_payload=True
                ).points
            except Exception:
                return []

        candidates = []
        for p in db_results:
            payload = p.payload
            boost = self._direction_boost(payload, parsed.direction)
            candidates.append({
                "doc_id": payload.get("doc_id", f"id_{p.id}"),
                "payload": payload,
                "rrf_score": p.score + boost,
                "parsed_query": parsed,
                "qdrant_id": p.id,
                "dense_rank": None,
                "bm25_rank": None, 
                "bm25_score": 0.0,
            })

        # Soft boost sorting over returned top K candidates
        candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        return candidates[:top_k]

    def retrieve_for_component(self, sub_query: str, top_k: int = 8) -> list[dict]:
        """Targeted retrieval for a single component sub-query (Product only)."""
        if not self._loaded:
            self.load()

        from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, FusionQuery, Fusion
        query_vec = self.embed_query(sub_query)
        sparse_vec = generate_sparse_vector(sub_query)
        product_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="product"))])

        try:
            if sparse_vec is None:
                results = self._client.query_points(
                    collection_name=self.settings.qdrant_collection,
                    query=query_vec,
                    using="dense",
                    limit=top_k,
                    query_filter=product_filter,
                    with_payload=True
                ).points
            else:
                prefetch = [
                    Prefetch(query=query_vec, using="dense", limit=top_k * 2, filter=product_filter),
                    Prefetch(query=sparse_vec, using="lexical", limit=top_k * 2, filter=product_filter)
                ]
                results = self._client.query_points(
                    collection_name=self.settings.qdrant_collection,
                    prefetch=prefetch,
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                    with_payload=True
                ).points

            docs = []
            for r in results:
                docs.append({
                    "doc_id": r.payload.get("doc_id", f"id_{r.id}"),
                    "score": r.score,
                    "payload": r.payload,
                    "rrf_score": r.score,
                    "final_score": r.score,
                    "bm25_score": 0.0,
                })
            docs.sort(key=lambda x: x["final_score"], reverse=True)
            return docs

        except Exception as e:
            logger.error("Component retrieval failed", sub_query=sub_query[:60], error=str(e))
            return []

    def multi_retrieve(self, components: list) -> dict[str, list[dict]]:
        """Run separate targeted retrieval for each ComponentSpec."""
        pools: dict[str, list[dict]] = {}
        for comp in components:
            docs = self.retrieve_for_component(comp.sub_query, top_k=10)
            pools[comp.type] = docs
        return pools
