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
        For consulting intent: tiered retrieval to guarantee knowledge/roadmap docs.
        """
        if not self._loaded:
            self.load()

        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Prefetch, FusionQuery, Fusion

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

        # Tiered retrieval for consulting: guaranteed knowledge/roadmap + open search
        if parsed.intent == "consulting":
            # P13.3 / T1: include service_page for consultation/discovery —
            # SEO-grade prose from xlsx closes semantic gaps (chat #96/300).
            knowledge_filter = Filter(must=[
                FieldCondition(key="doc_type", match=MatchAny(any=["knowledge", "faq", "roadmap", "service_page"]))
            ])
            half_k = max(top_k // 2, 5)
            try:
                # Tier 1: knowledge/faq/roadmap only
                if sparse_vec is not None:
                    prefetch_k = [
                        Prefetch(query=query_vec, using="dense", limit=half_k * 2, filter=knowledge_filter),
                        Prefetch(query=sparse_vec, using="lexical", limit=half_k * 2, filter=knowledge_filter),
                    ]
                    knowledge_results = self._client.query_points(
                        collection_name=self.settings.qdrant_collection,
                        prefetch=prefetch_k,
                        query=FusionQuery(fusion=Fusion.RRF),
                        limit=half_k,
                        with_payload=True,
                    ).points
                else:
                    knowledge_results = self._client.query_points(
                        collection_name=self.settings.qdrant_collection,
                        query=query_vec, using="dense",
                        limit=half_k, query_filter=knowledge_filter,
                        with_payload=True,
                    ).points

                # Tier 2: open search (all doc types)
                if sparse_vec is not None:
                    prefetch_o = [
                        Prefetch(query=query_vec, using="dense", limit=half_k * 2),
                        Prefetch(query=sparse_vec, using="lexical", limit=half_k * 2),
                    ]
                    open_results = self._client.query_points(
                        collection_name=self.settings.qdrant_collection,
                        prefetch=prefetch_o,
                        query=FusionQuery(fusion=Fusion.RRF),
                        limit=half_k,
                        with_payload=True,
                    ).points
                else:
                    open_results = self._client.query_points(
                        collection_name=self.settings.qdrant_collection,
                        query=query_vec, using="dense",
                        limit=half_k, with_payload=True,
                    ).points

                # Merge and deduplicate
                seen_ids = set()
                db_results = []
                for p in knowledge_results + open_results:
                    if p.id not in seen_ids:
                        seen_ids.add(p.id)
                        db_results.append(p)

                logger.info("Tiered consulting retrieval",
                            knowledge_hits=len(knowledge_results),
                            open_hits=len(open_results),
                            merged=len(db_results))

            except Exception as e:
                logger.warning("Tiered retrieval failed, falling back to standard", error=str(e))
                db_results = self._standard_search(query_vec, sparse_vec, query_filter, top_k)
        else:
            db_results = self._standard_search(query_vec, sparse_vec, query_filter, top_k)

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

    def _standard_search(self, query_vec, sparse_vec, query_filter, top_k):
        """Standard hybrid search (non-tiered)."""
        from qdrant_client.models import Prefetch, FusionQuery, Fusion
        try:
            if sparse_vec is None:
                return self._client.query_points(
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
                return self._client.query_points(
                    collection_name=self.settings.qdrant_collection,
                    prefetch=prefetch,
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                    with_payload=True
                ).points
        except Exception as e:
            logger.error("Qdrant hybrid search failed", error=str(e))
            try:
                return self._client.query_points(
                    collection_name=self.settings.qdrant_collection,
                    query=query_vec,
                    using="dense",
                    limit=top_k,
                    with_payload=True
                ).points
            except Exception:
                return []

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

    def retrieve_bundles(self, query: str, top_k: int = 5) -> list[dict]:
        """Dedicated retrieval filtered to doc_type=bundle for bundle-intent queries."""
        if not self._loaded:
            self.load()

        from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, FusionQuery, Fusion
        query_vec = self.embed_query(query)
        sparse_vec = generate_sparse_vector(query)
        bundle_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="bundle"))])

        try:
            if sparse_vec is None:
                results = self._client.query_points(
                    collection_name=self.settings.qdrant_collection,
                    query=query_vec,
                    using="dense",
                    limit=top_k,
                    query_filter=bundle_filter,
                    with_payload=True,
                ).points
            else:
                prefetch = [
                    Prefetch(query=query_vec, using="dense", limit=top_k * 2, filter=bundle_filter),
                    Prefetch(query=sparse_vec, using="lexical", limit=top_k * 2, filter=bundle_filter),
                ]
                results = self._client.query_points(
                    collection_name=self.settings.qdrant_collection,
                    prefetch=prefetch,
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
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
            logger.info("Bundle retrieval", query=query[:60], hits=len(docs))
            return docs

        except Exception as e:
            logger.error("Bundle retrieval failed", query=query[:60], error=str(e))
            return []

    def retrieve_roadmap(self, query: str, top_k: int = 3) -> list[dict]:
        """P11: Dedicated retrieval filtered to doc_type=roadmap for roadmap-trigger queries.

        Семантический поиск по общему индексу не достаёт roadmap-чанки в top-K
        из-за низкого embedding similarity (процессные описания vs. пользовательские запросы).
        Этот метод делает изолированный hybrid-запрос ТОЛЬКО в roadmap namespace,
        чтобы получить лучшие roadmap-совпадения независимо от конкурирующих doc_types.
        """
        if not self._loaded:
            self.load()

        from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, FusionQuery, Fusion
        query_vec = self.embed_query(query)
        sparse_vec = generate_sparse_vector(query)
        roadmap_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="roadmap"))])

        try:
            if sparse_vec is None:
                results = self._client.query_points(
                    collection_name=self.settings.qdrant_collection,
                    query=query_vec,
                    using="dense",
                    limit=top_k,
                    query_filter=roadmap_filter,
                    with_payload=True,
                ).points
            else:
                prefetch = [
                    Prefetch(query=query_vec, using="dense", limit=top_k * 2, filter=roadmap_filter),
                    Prefetch(query=sparse_vec, using="lexical", limit=top_k * 2, filter=roadmap_filter),
                ]
                results = self._client.query_points(
                    collection_name=self.settings.qdrant_collection,
                    prefetch=prefetch,
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
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
            logger.info("Roadmap retrieval", query=query[:60], hits=len(docs))
            return docs

        except Exception as e:
            logger.error("Roadmap retrieval failed", query=query[:60], error=str(e))
            return []

    def retrieve_by_intent(self, query: str, intent_name: str, hints: dict | None = None, top_k: int | None = None) -> list[dict]:
        """Intent-aware retrieval: select doc_type filter and top_k based on intent."""
        from qdrant_client.models import Filter, FieldCondition, MatchAny, Range

        top_k = top_k or self.settings.retrieval_top_k
        hints = hints or {}

        # P13.3 / T1: service_page added to consultation/describe/underspec/product_query.
        # Excluded from smeta_request (pricing only) and out_of_scope (no value-add).
        # P13.3 / T7: historical_deal added to pricing-grounded intents — closed deals
        # serve as "похожие сделки" anchors. Excluded from pure consultation (advisory).
        # P13.4: historical_request + referential strategies added after chat#97/M4 —
        # explicit user asks for past deals/examples must hit historical_deal bucket.
        INTENT_STRATEGIES = {
            "smeta_request":     (["bundle", "pricing_policy", "offer_profile", "product",
                                   "service_pricing_bridge", "offer_composition", "historical_deal"], 15),
            "consultation":      (["knowledge", "pricing_policy", "service_composition", "faq", "roadmap",
                                   "service_pricing_bridge", "service_page"], 10),
            "bundle_query":      (["bundle", "deal_profile", "offer_profile", "product", "pricing_policy",
                                   "service_pricing_bridge", "offer_composition", "historical_deal"], 12),
            "product_query":     (["product", "bundle", "pricing_policy", "offer_profile",
                                   "service_pricing_bridge", "service_page", "historical_deal"], 12),
            "underspec":         (["service_pricing_bridge", "pricing_policy", "knowledge", "faq",
                                   "product", "service_page", "historical_deal"], 8),
            "describe":          (["knowledge", "faq", "roadmap", "service_pricing_bridge", "service_page"], 8),
            "out_of_scope":      (["knowledge", "faq"], 5),
            # P13.4.4: tightened to deal-types only — when user explicitly asks for
            # past deals, including bundle/service_pricing_bridge lets BGE-M3 fill
            # the slate with bundles (semantically closer to «осмечивание»).
            "historical_request": (["historical_deal", "deal_profile", "offer_profile"], 12),
            "referential":       (["historical_deal", "deal_profile", "bundle",
                                   "offer_profile", "knowledge"], 10),
        }

        strategy = INTENT_STRATEGIES.get(intent_name)
        if strategy is None:
            return self.retrieve(query, top_k=top_k)

        doc_types, intent_top_k = strategy
        effective_k = min(intent_top_k, top_k) if top_k else intent_top_k

        query_vec = self.embed_query(query)
        sparse_vec = generate_sparse_vector(query)

        must_conditions = []
        if doc_types:
            must_conditions.append(FieldCondition(key="doc_type", match=MatchAny(any=doc_types)))

        height_cm = hints.get("height_cm")
        if height_cm and intent_name == "smeta_request":
            h = float(height_cm)
            must_conditions.append(FieldCondition(
                key="letter_height_cm",
                range=Range(gte=h * 0.7, lte=h * 1.3),
            ))

        query_filter = Filter(must=must_conditions) if must_conditions else None

        try:
            results = self._standard_search(query_vec, sparse_vec, query_filter, effective_k * 2)
        except Exception as e:
            logger.warning("Intent-aware retrieval failed, falling back to standard",
                           intent=intent_name, error=str(e))
            return self.retrieve(query, top_k=top_k)

        parsed = parse_query(query)

        # P10.6 D3 + P11 R1: roadmap boost parameters
        q_lower = query.lower()
        roadmap_trigger_words = (
            "регламент", "этап", "как делаете", "как делается",
            "процесс", "сроки", "схема работы", "порядок работ",
        )
        has_roadmap_trigger = any(w in q_lower for w in roadmap_trigger_words)
        hint_category_ids = set(hints.get("smeta_category_ids", []) or [])
        hint_product_ids = set(str(pid) for pid in (hints.get("product_ids", []) or []))

        candidates = []
        for p in results:
            payload = p.payload
            boost = self._direction_boost(payload, parsed.direction)

            # P10.6 D3 / P11 R1: бустим roadmap по двум сигналам
            if payload.get("doc_type") == "roadmap":
                if has_roadmap_trigger:
                    boost += 0.25
                # linked_smeta_category_ids ∩ hint_category_ids → +0.15
                linked_cats = set(payload.get("linked_smeta_category_ids") or [])
                if hint_category_ids and (linked_cats & hint_category_ids):
                    boost += 0.15
                # linked_product_ids ∩ hint_product_ids → +0.15
                linked_prods = set(str(x) for x in (payload.get("linked_product_ids") or []))
                if hint_product_ids and (linked_prods & hint_product_ids):
                    boost += 0.15

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

        candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        logger.info("Intent-aware retrieval", intent=intent_name,
                    doc_types=doc_types, hits=len(candidates[:effective_k]),
                    query=query[:60])
        return candidates[:effective_k]

    def multi_retrieve(self, components: list) -> dict[str, list[dict]]:
        """Run separate targeted retrieval for each ComponentSpec."""
        pools: dict[str, list[dict]] = {}
        for comp in components:
            docs = self.retrieve_for_component(comp.sub_query, top_k=10)
            pools[comp.type] = docs
        return pools
