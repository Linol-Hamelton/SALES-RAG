"""
Hybrid retriever: Qdrant dense search + BM25 lexical search, fused with RRF.
"""
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

from app.core.query_parser import ParsedQuery, parse_query
from app.utils.text import tokenize_ru
from app.utils.logging import get_logger

logger = get_logger(__name__)


def rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    """
    Hybrid retriever combining Qdrant vector search with BM25 lexical search.
    Uses Reciprocal Rank Fusion (RRF) to merge rankings.
    """

    def __init__(self, settings):
        self.settings = settings
        self._model = None
        self._client = None
        self._bm25 = None
        self._bm25_doc_ids: list[str] = []
        self._bm25_docs: list[dict] = []  # full doc payloads for BM25 results
        self._loaded = False

    def load(self):
        """Load embedding model, Qdrant client, and BM25 state."""
        if self._loaded:
            return

        import torch
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient

        # Load embedding model
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

        # Connect to Qdrant
        logger.info("Connecting to Qdrant", url=self.settings.qdrant_url)
        self._client = QdrantClient(url=self.settings.qdrant_url, timeout=30)

        # Load BM25 state
        bm25_path = Path(self.settings.index_path) / "bm25_state.pkl"
        if bm25_path.exists():
            logger.info("Loading BM25 state", path=str(bm25_path))
            with open(bm25_path, "rb") as f:
                state = pickle.load(f)
            self._bm25 = state["bm25"]
            self._bm25_doc_ids = state["doc_ids"]
            logger.info("BM25 loaded", vocab_size=len(self._bm25_doc_ids))
        else:
            logger.warning("BM25 state not found, lexical search disabled", path=str(bm25_path))

        self._loaded = True
        logger.info("HybridRetriever ready")

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._model is not None and self._client is not None

    def embed_query(self, query: str) -> list[float]:
        """Embed query with BGE-M3 prefix. Public for reuse by FeedbackStore."""
        prefix = "Represent this sentence for searching relevant passages: "
        embedding = self._model.encode(
            [prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].tolist()

    def _qdrant_search(self, query_vec: list[float], parsed: ParsedQuery, top_k: int) -> list[dict]:
        """Search Qdrant with optional direction filter."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        # No direction filter for consulting/general queries — need broad retrieval
        # including faq and knowledge docs
        if parsed.intent not in ("consulting", "general", "timeline") and \
           parsed.high_confidence_direction and parsed.direction != "Безнал":
            # Hard filter only for very clear direction signals in product/bundle queries
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="direction",
                        match=MatchValue(value=parsed.direction)
                    )
                ]
            )

        try:
            results = self._client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=query_vec,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            return [
                {
                    "doc_id": r.payload.get("doc_id", f"id_{r.id}"),
                    "score": r.score,
                    "payload": r.payload,
                    "qdrant_id": r.id,
                }
                for r in results
            ]
        except Exception as e:
            logger.error("Qdrant search failed", error=str(e))
            # Retry without filter if filter caused the error
            if query_filter is not None:
                try:
                    results = self._client.search(
                        collection_name=self.settings.qdrant_collection,
                        query_vector=query_vec,
                        limit=top_k,
                        with_payload=True,
                    )
                    return [
                        {
                            "doc_id": r.payload.get("doc_id", f"id_{r.id}"),
                            "score": r.score,
                            "payload": r.payload,
                            "qdrant_id": r.id,
                        }
                        for r in results
                    ]
                except Exception:
                    pass
            return []

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 lexical search."""
        if self._bm25 is None:
            return []

        tokens = tokenize_ru(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "doc_id": self._bm25_doc_ids[i],
                "bm25_score": float(scores[i]),
                "bm25_rank": rank,
            }
            for rank, i in enumerate(top_indices)
            if scores[i] > 0
        ]

    def _direction_boost(self, payload: dict, detected_direction: str | None) -> float:
        """Apply soft direction boost to score."""
        if detected_direction is None:
            return 0.0
        doc_direction = payload.get("direction", "")
        if doc_direction == detected_direction:
            return 0.1
        return 0.0

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Retrieve top-k relevant docs using hybrid RRF fusion.

        Returns list of docs with merged scores and payloads.
        """
        if not self._loaded:
            self.load()

        top_k = top_k or self.settings.retrieval_top_k
        parsed = parse_query(query)

        logger.info("Retrieving", query=query[:100], direction=parsed.direction,
                    intent=parsed.intent, top_k=top_k)

        # Dense retrieval
        query_vec = self.embed_query(query)
        dense_results = self._qdrant_search(query_vec, parsed, top_k)

        # Lexical retrieval
        bm25_results = self._bm25_search(query, top_k)

        # Build doc_id → candidate dict
        candidates: dict[str, dict] = {}

        # Add dense results
        for rank, result in enumerate(dense_results):
            doc_id = result["doc_id"]
            candidates[doc_id] = {
                "doc_id": doc_id,
                "payload": result["payload"],
                "dense_rank": rank,
                "bm25_rank": None,
                "bm25_score": 0.0,
            }

        # Merge BM25 results — collect BM25-only doc_ids for payload fetch
        bm25_only_ids = []
        for result in bm25_results:
            doc_id = result["doc_id"]
            if doc_id in candidates:
                candidates[doc_id]["bm25_rank"] = result["bm25_rank"]
                candidates[doc_id]["bm25_score"] = result["bm25_score"]
            else:
                bm25_only_ids.append(doc_id)
                candidates[doc_id] = {
                    "doc_id": doc_id,
                    "payload": {},
                    "dense_rank": None,
                    "bm25_rank": result["bm25_rank"],
                    "bm25_score": result["bm25_score"],
                }

        # Fetch missing payloads for BM25-only results from Qdrant
        if bm25_only_ids:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchAny
                fetch_results = self._client.scroll(
                    collection_name=self.settings.qdrant_collection,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="doc_id", match=MatchAny(any=bm25_only_ids[:20]))]
                    ),
                    limit=len(bm25_only_ids[:20]),
                    with_payload=True,
                )
                for point in fetch_results[0]:
                    pid = point.payload.get("doc_id", "")
                    if pid in candidates:
                        candidates[pid]["payload"] = point.payload
            except Exception as e:
                logger.warning("Failed to fetch BM25-only payloads", error=str(e))

        # Compute RRF scores
        alpha = self.settings.rrf_alpha  # default 0.7 (dense-heavy)
        for doc_id, cand in candidates.items():
            dense_rrf = rrf_score(cand["dense_rank"]) if cand["dense_rank"] is not None else 0.0
            bm25_rrf = rrf_score(cand["bm25_rank"]) if cand["bm25_rank"] is not None else 0.0
            direction_boost = self._direction_boost(cand["payload"], parsed.direction)
            cand["rrf_score"] = alpha * dense_rrf + (1 - alpha) * bm25_rrf + direction_boost
            cand["parsed_query"] = parsed

        # Sort by RRF score descending
        ranked = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Filter out docs still missing payloads (shouldn't happen now)
        ranked = [c for c in ranked if c["payload"]]

        result = ranked[:top_k]
        logger.info("Retrieved", count=len(result), top_score=result[0]["rrf_score"] if result else 0)
        return result

    def retrieve_for_component(self, sub_query: str, top_k: int = 8) -> list[dict]:
        """
        Targeted retrieval for a single component sub-query.
        Filters to doc_type=product only — needed for parametric pricing.
        Used by multi-query pipeline for parametric pricing.
        """
        if not self._loaded:
            self.load()

        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_vec = self.embed_query(sub_query)
        product_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="product"))])

        try:
            results = self._client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=query_vec,
                limit=top_k,
                query_filter=product_filter,
                with_payload=True,
            )
            docs = [
                {
                    "doc_id": r.payload.get("doc_id", f"id_{r.id}"),
                    "score": r.score,
                    "payload": r.payload,
                    "rrf_score": r.score,
                    "final_score": r.score,
                }
                for r in results
            ]
            # Also run BM25 for lexical boost
            bm25_results = self._bm25_search(sub_query, top_k)
            bm25_map = {r["doc_id"]: r for r in bm25_results}

            for doc in docs:
                bm25 = bm25_map.get(doc["doc_id"], {})
                doc["bm25_score"] = bm25.get("bm25_score", 0.0)
                doc["final_score"] = 0.7 * doc["score"] + 0.3 * (doc["bm25_score"] / 10.0)

            docs.sort(key=lambda x: x["final_score"], reverse=True)
            return docs

        except Exception as e:
            logger.error("Component retrieval failed", sub_query=sub_query[:60], error=str(e))
            return []

    def multi_retrieve(self, components: list) -> dict[str, list[dict]]:
        """
        Run separate targeted retrieval for each ComponentSpec.
        Returns: {component_type: [docs]}
        """
        pools: dict[str, list[dict]] = {}
        for comp in components:
            docs = self.retrieve_for_component(comp.sub_query, top_k=10)
            pools[comp.type] = docs
            logger.info("Component retrieval done",
                        component=comp.type, docs=len(docs),
                        top_score=docs[0]["score"] if docs else 0)
        return pools
