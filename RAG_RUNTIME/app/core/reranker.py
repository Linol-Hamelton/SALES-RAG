"""
Cross-encoder reranker with heuristic score boosts.
Uses BAAI/bge-reranker-v2-m3 on CPU for production (VPS).
"""
from pathlib import Path
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Boost multipliers (also in settings.yaml, these are defaults)
DEFAULT_BOOSTS = {
    "auto_high": 1.3,       # price_mode=auto + confidence_tier=high
    "guided": 1.1,          # price_mode=guided
    "order_rows_high": 1.15, # order_rows >= 100
    "doc_type_match": 1.2,  # doc_type matches query intent
    "direction_match": 1.25, # direction matches query
    "financial_modifier": 0.3,  # penalize Безнал/discount products
}


class CrossEncoderReranker:
    """
    Reranks retrieved candidates using a cross-encoder model.
    Falls back to heuristic-only ranking if model unavailable.
    """

    def __init__(self, settings):
        self.settings = settings
        self._model = None
        self._loaded = False

    def load(self):
        """Load cross-encoder model."""
        if self._loaded:
            return

        model_path = Path(self.settings.reranker_model_path)
        try:
            from sentence_transformers import CrossEncoder
            if model_path.exists():
                logger.info("Loading reranker", path=str(model_path))
                self._model = CrossEncoder(str(model_path), device="cpu", max_length=512)
            else:
                logger.info("Downloading bge-reranker-v2-m3")
                self._model = CrossEncoder(
                    "BAAI/bge-reranker-v2-m3",
                    device="cpu",
                    max_length=512,
                    cache_folder=str(model_path.parent),
                )
            logger.info("Reranker loaded")
        except Exception as e:
            logger.warning("Reranker model failed to load, using heuristic only", error=str(e))
            self._model = None

        self._loaded = True

    def _get_boosts(self, settings) -> dict:
        """Get boost values from settings.yaml if available."""
        try:
            import yaml
            from pathlib import Path
            yaml_path = Path(settings.project_root) / "configs" / "settings.yaml"
            if yaml_path.exists():
                with open(yaml_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                boost_cfg = cfg.get("reranker", {}).get("boosts", {})
                # Map YAML keys to internal keys
                mapping = {
                    "auto_high_confidence": "auto_high",
                    "order_rows_high": "order_rows_high",
                    "doc_type_match": "doc_type_match",
                    "direction_match": "direction_match",
                    "financial_modifier": "financial_modifier",
                    "guided": "guided",
                }
                mapped = {mapping.get(k, k): v for k, v in boost_cfg.items()}
                return {**DEFAULT_BOOSTS, **mapped}
        except Exception:
            pass
        return DEFAULT_BOOSTS

    def _compute_heuristic_boost(self, payload: dict, parsed_query, boosts: dict) -> float:
        """Compute multiplicative boost based on doc metadata."""
        multiplier = 1.0

        price_mode = payload.get("price_mode", "")
        confidence_tier = payload.get("confidence_tier", "")
        order_rows = payload.get("order_rows", 0) or 0
        doc_type = payload.get("doc_type", "")
        direction = payload.get("direction", "")
        manual_review_reason = payload.get("manual_review_reason", "")

        # Auto + high confidence boost
        if price_mode == "auto" and confidence_tier == "high":
            multiplier *= boosts.get("auto_high", 1.3)
        elif price_mode == "guided":
            multiplier *= boosts.get("guided", 1.1)

        # Graduated order history boost (log-scaled)
        if order_rows >= 10:
            import math
            multiplier *= min(1.0 + math.log10(order_rows) * 0.07, 1.35)

        # Doc type matches query intent
        if parsed_query is not None:
            intent = parsed_query.intent
            if (intent == "bundle" and doc_type == "bundle") or \
               (intent == "product" and doc_type == "product") or \
               (intent == "policy" and doc_type == "pricing_policy"):
                multiplier *= boosts.get("doc_type_match", 1.2)

            # Direction match boost
            if parsed_query.direction and direction == parsed_query.direction:
                multiplier *= boosts.get("direction_match", 1.25)

        # Bundle completeness boost — full-service bundles surface higher
        if doc_type == "bundle":
            sp = (payload.get("sample_products", "") or "").lower()
            has_mounting = any(kw in sp for kw in [
                "монтаж вывески", "монтаж буквы", "монтаж конструкции",
            ])
            has_frame = any(kw in sp for kw in [
                "сварка каркаса", "каркас", "профильная труба",
            ])
            if has_mounting and has_frame:
                multiplier *= 1.15  # full-service bundles preferred
            elif not has_mounting and not has_frame:
                multiplier *= 0.85  # production-only bundles demoted

        # Boost FAQ, knowledge, and roadmap docs — they carry expert consulting context
        # Higher boost for consulting queries so knowledge surfaces above irrelevant products
        if doc_type in ("faq", "knowledge", "roadmap"):
            if parsed_query is not None and parsed_query.intent == "consulting":
                multiplier *= 2.0  # consulting docs are primary source
            else:
                multiplier *= 1.3

        # Demote deal_profile for pricing queries — line_total != per-unit price
        if doc_type == "deal_profile":
            if parsed_query is not None and parsed_query.intent in ("product", "bundle"):
                multiplier *= 0.60  # strong demotion to let product/bundle docs surface

                # Extra demotion for repair/demolition deals (no fabrication)
                comp_summary = (payload.get("component_summary", "") or "").lower()
                mat_text = (payload.get("materials", "") or "").lower()
                all_text = f"{comp_summary} {mat_text}"
                has_demolition = "демонтаж" in all_text
                has_fabrication = any(kw in all_text for kw in [
                    "сборка объема", "сборка объёма", "акрил", "пвх",
                ])
                if has_demolition and not has_fabrication:
                    multiplier *= 0.40  # total ≈ 0.24 for repair deals

        # Service composition: boost only for consulting, demote for product/bundle
        if doc_type == "service_composition":
            if parsed_query is not None and parsed_query.intent == "consulting":
                multiplier *= 1.1
            elif parsed_query is not None and parsed_query.intent in ("product", "bundle"):
                multiplier *= 0.9  # should not compete with product docs

        # Boost timeline_fact for timeline queries
        if doc_type == "timeline_fact":
            if parsed_query is not None and parsed_query.intent == "timeline":
                multiplier *= 1.3
            else:
                multiplier *= 1.05

        # Penalize financial modifiers
        if manual_review_reason == "financial_modifier" or \
           any(kw in (payload.get("product_name", "") or "").lower()
               for kw in ["безнал", "скидка", "надбавка"]):
            multiplier *= boosts.get("financial_modifier", 0.3)

        return multiplier

    def rerank(self, query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
        """
        Rerank candidates using cross-encoder + heuristic boosts.

        Args:
            query: Original user query
            candidates: List of candidate dicts from retriever (with payload)
            top_n: Number of results to return (default from settings)

        Returns:
            Reranked list of top_n candidates
        """
        if not self._loaded:
            self.load()

        if not candidates:
            return []

        top_n = top_n or self.settings.rerank_top_n
        boosts = self._get_boosts(self.settings)

        # Get parsed_query from first candidate if available
        parsed_query = candidates[0].get("parsed_query") if candidates else None

        # Cross-encoder scoring
        if self._model is not None:
            pairs = [(query, c["payload"].get("searchable_text", "")[:512]) for c in candidates]
            try:
                ce_scores = self._model.predict(pairs, batch_size=20, show_progress_bar=False)
                for i, cand in enumerate(candidates):
                    cand["cross_encoder_score"] = float(ce_scores[i])
            except Exception as e:
                logger.warning("Cross-encoder scoring failed", error=str(e))
                for cand in candidates:
                    cand["cross_encoder_score"] = cand.get("rrf_score", 0.0)
        else:
            # Use RRF score as fallback
            for cand in candidates:
                cand["cross_encoder_score"] = cand.get("rrf_score", 0.0)

        # Apply heuristic boosts
        for cand in candidates:
            boost = self._compute_heuristic_boost(cand["payload"], parsed_query, boosts)
            cand["final_score"] = cand["cross_encoder_score"] * boost
            cand["heuristic_boost"] = boost

        # Sort by final score
        reranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)

        # Deduplicate by content fingerprint (first 120 chars of searchable_text)
        seen_content: set[str] = set()
        deduped: list[dict] = []
        for cand in reranked:
            fingerprint = cand["payload"].get("searchable_text", "")[:120].strip()
            if fingerprint and fingerprint in seen_content:
                continue
            seen_content.add(fingerprint)
            deduped.append(cand)

        # Doc-type diversification: ensure knowledge/FAQ, bundle, and product docs
        # are represented in the final set
        result = self._diversify(deduped, top_n, parsed_query)

        logger.info("Reranked", input_count=len(candidates), output_count=len(result),
                    top_score=result[0]["final_score"] if result else 0)
        return result

    @staticmethod
    def _diversify(candidates: list[dict], top_n: int, parsed_query=None) -> list[dict]:
        """
        Ensure doc_type diversity: at least 1 knowledge/FAQ, 1 bundle,
        and 1 product (for pricing queries) in the final set.
        Hard cap: max 3 deal_profiles.
        """
        if len(candidates) <= top_n:
            return candidates

        result = candidates[:top_n]

        # --- Hard cap: max 3 deal_profile docs ---
        dp_indices = [
            i for i, c in enumerate(result)
            if c["payload"].get("doc_type") == "deal_profile"
        ]
        if len(dp_indices) > 2:
            overflow_non_dp = [
                c for c in candidates[top_n:]
                if c["payload"].get("doc_type") != "deal_profile"
            ]
            to_remove = dp_indices[3:]  # evict beyond 3rd deal_profile
            for idx, repl_idx in zip(
                sorted(to_remove, reverse=True),
                range(len(overflow_non_dp)),
            ):
                if repl_idx < len(overflow_non_dp):
                    result[idx] = overflow_non_dp[repl_idx]

        result_types = {c["payload"].get("doc_type") for c in result}

        # For product/bundle intent: guarantee at least 1 product doc
        intent = parsed_query.intent if parsed_query else None
        if intent in ("product", "bundle") and "product" not in result_types:
            for cand in candidates[top_n:]:
                if cand["payload"].get("doc_type") == "product":
                    result[-1] = cand  # replace lowest-scoring doc
                    break

        # Check if knowledge/FAQ is missing
        result_types = {c["payload"].get("doc_type") for c in result}
        if "knowledge" not in result_types and "faq" not in result_types:
            for cand in candidates[top_n:]:
                dt = cand["payload"].get("doc_type", "")
                if dt in ("knowledge", "faq"):
                    # Don't displace the product doc we just inserted
                    replace_idx = len(result) - 1
                    if result[replace_idx]["payload"].get("doc_type") == "product" and len(result) > 1:
                        replace_idx = len(result) - 2
                    result[replace_idx] = cand
                    break

        # Check if bundle is missing
        result_types = {c["payload"].get("doc_type") for c in result}
        if "bundle" not in result_types:
            for cand in candidates[top_n:]:
                if cand["payload"].get("doc_type") == "bundle":
                    # Find a slot that isn't product or knowledge/faq
                    replace_idx = len(result) - 1
                    for ri in range(len(result) - 1, -1, -1):
                        dt = result[ri]["payload"].get("doc_type", "")
                        if dt not in ("product", "knowledge", "faq"):
                            replace_idx = ri
                            break
                    result[replace_idx] = cand
                    break

        # Ensure at least 1 roadmap doc for consulting queries
        result_types = {c["payload"].get("doc_type") for c in result}
        if "roadmap" not in result_types and parsed_query and parsed_query.intent == "consulting":
            for cand in candidates[top_n:]:
                if cand["payload"].get("doc_type") == "roadmap":
                    replace_idx = len(result) - 1
                    for ri in range(len(result) - 1, -1, -1):
                        dt = result[ri]["payload"].get("doc_type", "")
                        if dt not in ("product", "knowledge", "faq", "bundle"):
                            replace_idx = ri
                            break
                    result[replace_idx] = cand
                    break

        return result
