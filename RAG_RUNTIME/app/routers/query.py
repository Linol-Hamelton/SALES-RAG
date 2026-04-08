"""Query endpoints: /query and /query_structured."""
import time
from fastapi import APIRouter, Request, HTTPException, Depends
from app.schemas.query import QueryRequest, HumanQueryResponse, StructuredResponse, ChatMessage
from app.schemas.pricing import (
    PriceBand, EstimatedPrice, BundleItem, Reference, SourceDistinction,
    ParametricBreakdown, ParametricLineItem, DealItem,
    SourceSegment, SegmentedReferences,
)
from app.core.query_decomposer import decompose, QueryDecomposition
from app.core.query_parser import parse_query
from app.core import parametric_calculator as param_calc
from app.core.feedback_store import build_feedback_context
from app.core.deal_lookup import DealLookup
from app.auth import get_optional_user
from app.routers.chats import save_message, get_chat_history
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Query"])


MIN_RELEVANT_SCORE = 0.005  # below this = cross-encoder deems irrelevant, don't use for pricing

_ESTIMATE_KEYWORDS = [
    "осмет", "смету", "смета", "сметы",
    "список товаров", "что добавить в сделку", "товары для сделки",
    "сформируй сделку", "составь сделку", "позиции для сделки",
    "заполни сделку", "создай сделку", "что включить в сделку",
    # Pricing intent (П7): любые «сколько стоит / под ключ / оцени / прайс»
    # — это запрос на estimate, требующий SmetaEngine, а не свободный summary.
    "под ключ", "сколько стоит", "сколько будет", "сколько обойдётся", "сколько обойдется",
    "оцени", "оцените", "оценить", "оценка", "прайс", "цена на", "цену на",
    "стоимость", "посчитай", "посчитайте", "рассчитай", "рассчитайте", "расчёт", "расчет",
]


def _is_deal_estimate_query(query: str) -> bool:
    """Detect if the user wants a deal estimate (list of products for Bitrix24)."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in _ESTIMATE_KEYWORDS)


def _parse_deal_items(raw_json: dict) -> list[DealItem]:
    """Extract and validate deal_items from LLM JSON output."""
    raw_items = raw_json.get("deal_items")
    if not raw_items or not isinstance(raw_items, list):
        return []
    result = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            qty = float(item.get("quantity", 1) or 1)
            unit_price = float(item.get("unit_price", 0) or 0)
            total = float(item.get("total", 0) or 0)
            if total == 0 and unit_price > 0:
                total = round(qty * unit_price, 2)
            result.append(DealItem(
                product_name=str(item.get("product_name", "")),
                quantity=qty,
                unit=str(item.get("unit", "шт")),
                unit_price=unit_price,
                total=total,
                b24_section=str(item.get("b24_section", "")),
                notes=str(item.get("notes", "")),
            ))
        except (ValueError, TypeError):
            continue
    return result


def _build_size_context(decomp: "QueryDecomposition") -> str:
    """Build size-aware pricing context when query has letter/height info."""
    if decomp.letter_count == 0:
        return ""
    lines = [
        f"ПАРАМЕТРЫ ЗАПРОСА: надпись '{decomp.letter_text}' "
        f"({decomp.letter_count} букв)",
    ]
    if decomp.technology:
        lines.append(f"Технология: {decomp.technology}")
    if decomp.height_cm > 0:
        lines.append(f"Высота: {decomp.height_cm:.0f} см")
        # Use pricing_resolver's rate tables for consistency
        from app.core.pricing_resolver import PricingResolver
        rate_min, rate_max = PricingResolver._get_letter_rate(decomp.height_cm, decomp.technology)
        low = rate_min * decomp.letter_count
        high = rate_max * decomp.letter_count
        lines.append(
            f"Рыночный ориентир по размеру: "
            f"{low:,} – {high:,} руб (только буквы, без монтажа/каркаса)"
        )
    else:
        lines.append("Высота НЕ указана — уточни у клиента для точного расчёта")
    if decomp.linear_meters > 0:
        lines.append(f"Расчётные погонные метры: {decomp.linear_meters:.1f} мп")
    return "\n".join(lines)


def _format_pricing_breakdown(pr) -> str:
    """Format pre-calculated pricing breakdown for LLM context."""
    if pr is None or pr.total_under_key_min is None:
        return ""
    lines = [
        f"РАСЧЁТ СИСТЕМЫ: итого под ключ {pr.total_under_key_min:,.0f} – {pr.total_under_key_max:,.0f} руб"
    ]
    if pr.price_breakdown:
        parts = []
        for name, (lo, hi) in pr.price_breakdown.items():
            parts.append(f"{name}: {lo:,.0f}–{hi:,.0f}")
        lines.append(f"({' | '.join(parts)})")
    lines.append("Используй эти цифры как основу ответа. НЕ пересчитывай.")
    return "\n".join(lines)


def _build_references(docs: list[dict]) -> list[Reference]:
    """Build Reference objects from retrieved docs. Only show relevant ones."""
    refs = []
    for doc in docs[:8]:
        score = doc.get("final_score", doc.get("rrf_score", 0.0))
        if score < MIN_RELEVANT_SCORE:
            continue  # skip noise docs
        payload = doc.get("payload", {})
        searchable_text = payload.get("searchable_text", "")
        refs.append(Reference(
            doc_id=payload.get("doc_id", ""),
            doc_type=payload.get("doc_type", ""),
            score=round(score, 4),
            snippet=searchable_text[:120],
        ))
    return refs


def _dominant_direction(docs: list[dict], top: int = 10) -> str | None:
    """Return direction appearing in >=60% of top docs, else None."""
    from collections import Counter
    dirs = [d.get("payload", {}).get("direction") for d in docs[:top] if d.get("payload", {}).get("direction")]
    if not dirs:
        return None
    counter = Counter(dirs)
    top_dir, freq = counter.most_common(1)[0]
    return top_dir if freq / len(dirs) >= 0.6 else None


def _seg_from_doc(doc: dict, kind: str, photo_enrich: dict | None = None) -> SourceSegment:
    p = doc.get("payload", {})
    score = round(doc.get("final_score", doc.get("rrf_score", 0.0)), 4)
    text = p.get("searchable_text", "")
    deal_id = str(p.get("deal_id") or "") or None

    title = (p.get("title") or p.get("sample_title") or p.get("deal_title")
             or p.get("sample_order_title") or "").strip() or (text[:60] if text else "")
    direction = p.get("direction") or None
    total = p.get("line_total") or p.get("median_deal_value") or p.get("deal_total")
    duration = p.get("deal_duration_days") or p.get("median_duration_days")
    image_urls = list(p.get("image_urls") or [])

    product_type = p.get("product_type") or None
    application = p.get("application") or None
    roi_hint = None
    dimensions = p.get("dimensions") or None

    # Enrich from photo_index for deal/offer segments when doc payload lacks visuals
    if photo_enrich:
        if not image_urls and photo_enrich.get("image_urls"):
            image_urls = list(photo_enrich["image_urls"])
        product_type = product_type or photo_enrich.get("product_type") or None
        application = application or photo_enrich.get("application") or None
        dimensions = dimensions or photo_enrich.get("dimensions") or None
        roi = photo_enrich.get("roi_romi_avg")
        if roi:
            roi_hint = f"ROMI ~{int(roi)}%"

    subtitle_parts = [x for x in [product_type, dimensions] if x]
    subtitle = " · ".join(subtitle_parts) if subtitle_parts else None

    return SourceSegment(
        kind=kind,
        deal_id=deal_id,
        title=title[:140],
        subtitle=subtitle,
        direction=direction,
        total=float(total) if isinstance(total, (int, float)) else None,
        duration_days=int(duration) if isinstance(duration, (int, float)) else None,
        snippet=text[:180],
        score=score,
        image_urls=image_urls[:4],
        product_type=product_type,
        application=application,
        roi_hint=roi_hint,
    )


def _build_segmented_references(docs: list[dict], photo_index=None) -> SegmentedReferences:
    """Partition retrieved docs into three thematic segments.

    - similar_orders: doc_type == deal_profile (top 7), enriched via photo_index.
    - similar_offers: doc_type == offer_profile, fallback to bundle[dataset=offers] (top 5).
    - product_links: doc_type == photo_analysis with image_urls (top 6), deduped by deal_id.
    """
    def _dir_hard_filter(doc, dom):
        if not dom:
            return True
        d = doc.get("payload", {}).get("direction")
        return (not d) or d == dom

    dom_dir = _dominant_direction(docs)
    relevant = [d for d in docs
                if d.get("final_score", d.get("rrf_score", 0.0)) >= MIN_RELEVANT_SCORE]

    seen_deals: set[str] = set()
    similar_orders: list[SourceSegment] = []
    similar_offers: list[SourceSegment] = []
    product_links: list[SourceSegment] = []
    seen_pt_dir: set[tuple] = set()

    # 1) similar_orders — deal_profile
    for doc in relevant:
        if len(similar_orders) >= 7:
            break
        p = doc.get("payload", {})
        if p.get("doc_type") != "deal_profile":
            continue
        if not _dir_hard_filter(doc, dom_dir):
            continue
        deal_id = str(p.get("deal_id") or "")
        if deal_id and deal_id in seen_deals:
            continue
        enrich = photo_index.get(deal_id) if photo_index and deal_id else None
        similar_orders.append(_seg_from_doc(doc, "order", enrich))
        if deal_id:
            seen_deals.add(deal_id)

    # 2) similar_offers — prefer offer_profile, fallback to bundle[offers]
    offer_profiles = [d for d in relevant
                      if d.get("payload", {}).get("doc_type") == "offer_profile"]
    if offer_profiles:
        for doc in offer_profiles:
            if len(similar_offers) >= 5:
                break
            if not _dir_hard_filter(doc, dom_dir):
                continue
            deal_id = str(doc.get("payload", {}).get("deal_id") or "")
            if deal_id and deal_id in seen_deals:
                continue
            enrich = photo_index.get(deal_id) if photo_index and deal_id else None
            similar_offers.append(_seg_from_doc(doc, "offer", enrich))
            if deal_id:
                seen_deals.add(deal_id)
    else:
        for doc in relevant:
            if len(similar_offers) >= 5:
                break
            p = doc.get("payload", {})
            if p.get("doc_type") != "bundle":
                continue
            if p.get("dataset_type") != "offers":
                continue
            if not _dir_hard_filter(doc, dom_dir):
                continue
            similar_offers.append(_seg_from_doc(doc, "offer"))

    # 3) product_links — photo_analysis with image_urls, dedup by (deal_id) and (product_type, direction)
    for doc in relevant:
        if len(product_links) >= 6:
            break
        p = doc.get("payload", {})
        if p.get("doc_type") != "photo_analysis":
            continue
        urls = p.get("image_urls") or []
        if not urls:
            continue
        if not _dir_hard_filter(doc, dom_dir):
            continue
        deal_id = str(p.get("deal_id") or "")
        if deal_id and deal_id in seen_deals:
            continue
        pt_key = (p.get("product_type") or "", p.get("direction") or "")
        if pt_key in seen_pt_dir and pt_key != ("", ""):
            continue
        product_links.append(_seg_from_doc(doc, "product_visual"))
        if deal_id:
            seen_deals.add(deal_id)
        seen_pt_dir.add(pt_key)

    return SegmentedReferences(
        similar_orders=similar_orders,
        similar_offers=similar_offers,
        product_links=product_links,
    )


def _filter_relevant_docs(docs: list[dict]) -> list[dict]:
    """Filter out docs with near-zero relevance scores to prevent noise pricing."""
    relevant = [d for d in docs if d.get("final_score", d.get("rrf_score", 0.0)) >= MIN_RELEVANT_SCORE]
    return relevant if relevant else docs[:3]  # always return at least 3 for LLM context


def _build_suggested_bundle(docs: list[dict]) -> list[BundleItem]:
    """Build BundleItem list from top bundle docs."""
    items = []
    for doc in docs[:5]:
        payload = doc.get("payload", {})
        if payload.get("doc_type") != "bundle":
            continue
        sample_products_str = payload.get("sample_products", "")
        bundle_key = payload.get("bundle_key", "")
        product_keys = bundle_key.split("|") if bundle_key else []
        for key in product_keys[:6]:
            if key:
                items.append(BundleItem(
                    product_key=key.strip(),
                    product_name=f"Продукт {key.strip()}",
                    direction=payload.get("direction", ""),
                ))
        break  # only top bundle doc
    return items


def _detect_source_distinction(docs: list[dict]) -> SourceDistinction:
    """Determine which data sources contributed to the response."""
    has_orders = False
    has_offers = False
    for doc in docs:
        payload = doc.get("payload", {})
        dataset = payload.get("dataset_type", "")
        if dataset == "orders" or payload.get("order_rows", 0) > 0:
            has_orders = True
        if dataset == "offers" or payload.get("offer_rows", 0) > 0:
            has_offers = True

    if has_orders and has_offers:
        dataset_type = "both"
    elif has_orders:
        dataset_type = "orders"
    elif has_offers:
        dataset_type = "offers"
    else:
        dataset_type = "catalog_only"

    return SourceDistinction(
        has_order_data=has_orders,
        has_offer_data=has_offers,
        dataset_type=dataset_type,
        data_freshness_note="Данные по заказам до 2026-03",
    )


def _build_parametric_breakdown(
    estimate: "param_calc.ParametricEstimate",
    decomp: QueryDecomposition,
) -> ParametricBreakdown:
    """Convert ParametricEstimate + QueryDecomposition to Pydantic schema."""
    items = [
        ParametricLineItem(
            component=li.component,
            label=li.label,
            product_name=li.product_name,
            unit_price=li.unit_price,
            quantity=li.quantity,
            unit=li.unit,
            total=li.total,
            confidence_tier=li.confidence_tier,
            quantity_basis=li.quantity_basis,
        )
        for li in estimate.line_items
    ]
    return ParametricBreakdown(
        line_items=items,
        total_estimate=estimate.total_estimate,
        total_min=estimate.total_min,
        total_max=estimate.total_max,
        missing_components=estimate.missing_components,
        letter_text=decomp.letter_text,
        letter_count=decomp.letter_count,
        height_cm=decomp.height_cm,
        linear_meters=decomp.linear_meters,
    )


async def _handle_complex_query(
    query: str,
    decomp: QueryDecomposition,
    retriever,
    reranker,
    generator,
    top_k: int,
) -> dict:
    """
    Pipeline for complex multi-component queries:
    1. Multi-retrieve: separate search per component (products, for parametric)
    2. Parametric calculate: internal production costs (reference only, NOT retail price)
    3. Standard retrieve: bundles + context with real deal prices
    4. Generate LLM summary using real bundle prices + market knowledge
    """
    # 1. Multi-retrieve per component (products only — for parametric breakdown)
    doc_pools = retriever.multi_retrieve(decomp.components)

    # 2. Parametric estimate — internal production norms, NOT retail prices
    estimate = param_calc.calculate(decomp.components, doc_pools)
    breakdown_text = param_calc.format_breakdown(estimate)

    # 3. Standard retrieval for bundles with real deal prices (primary price reference)
    standard_candidates = retriever.retrieve(query, top_k=top_k * 2)
    reranked = reranker.rerank(query, standard_candidates, top_n=top_k)

    # 4. Build context for LLM
    parametric_context = (
        f"Запрос: {query}\n\n"
        f"Параметры: буквы='{decomp.letter_text}' ({decomp.letter_count} букв), "
        f"высота={decomp.height_cm:.0f} см, линейные метры≈{decomp.linear_meters:.1f} мп\n\n"
        f"ВНИМАНИЕ: следующие данные — внутренние нормы производства (себестоимость операций), "
        f"НЕ розничные цены. Для розничной цены используй данные [Набор] из контекста.\n"
        f"{breakdown_text}"
    )

    return {
        "doc_pools": doc_pools,
        "estimate": estimate,
        "reranked": reranked,
        "parametric_context": parametric_context,
    }


@router.post("/query", response_model=HumanQueryResponse)
async def query_human(req: QueryRequest, request: Request,
                      user: dict | None = Depends(get_optional_user)) -> HumanQueryResponse:
    """
    Query the RAG system and get a human-readable Russian response.
    """
    t0 = time.monotonic()

    retriever = request.app.state.retriever
    reranker = request.app.state.reranker
    generator = request.app.state.generator
    pricing = request.app.state.pricing
    vision = getattr(request.app.state, "vision", None)

    if not retriever or not retriever.is_ready:
        raise HTTPException(503, "Retriever not ready. Index may not be built yet.")

    # Load chat history from DB when chat_id is provided
    if req.chat_id and user:
        db_history = get_chat_history(req.chat_id, user["id"], limit=12)
        if db_history and not req.history:
            req.history = [ChatMessage(role=m["role"], content=m["content"]) for m in db_history]

    try:
        # Vision analysis (if image provided)
        vision_context = ""
        if req.image_base64 and vision and vision.is_available:
            vision_context = await vision.analyze(req.image_base64, req.image_mime_type)
            if vision_context:
                logger.info("Vision analysis prepended to query context")

        extra_ctx = f"АНАЛИЗ ИЗОБРАЖЕНИЯ:\n{vision_context}\n\n" if vision_context else ""

        # --- Parse query for clarification needs ---
        parsed = parse_query(req.query)
        if parsed.needs_clarification:
            extra_ctx += "ВАЖНО: Запрос слишком общий — не указан тип изделия, размеры или направление. Задай 1-2 уточняющих вопроса клиенту ВМЕСТО угадывания цены. Дай примерные диапазоны по типам.\n\n"

        # --- Decompose query for size-aware pricing ---
        decomp = decompose(req.query)

        if decomp.is_complex and decomp.components:
            # Full parametric pipeline (same as query_structured)
            logger.info("Complex query in /query — using parametric pipeline",
                        letter_text=decomp.letter_text,
                        letter_count=decomp.letter_count,
                        height_cm=decomp.height_cm)

            result = await _handle_complex_query(
                req.query, decomp, retriever, reranker, generator, req.top_k
            )
            reranked = result["reranked"]
            extra_ctx += result["parametric_context"]
            pricing_resolution = pricing.resolve(reranked, decomp=decomp)

            # Inject pre-calculated breakdown
            breakdown_ctx = _format_pricing_breakdown(pricing_resolution)
            if breakdown_ctx:
                extra_ctx += "\n\n" + breakdown_ctx
        else:
            # Standard retrieval with optional size context
            candidates = retriever.retrieve(req.query, top_k=req.top_k * 2)
            if not candidates:
                return HumanQueryResponse(
                    summary="По вашему запросу не найдено подходящих товаров/услуг в базе данных.",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )

            reranked = reranker.rerank(req.query, candidates, top_n=req.top_k)
            pricing_resolution = pricing.resolve(reranked, decomp=decomp)

            # Inject size context for queries with letter info
            size_ctx = _build_size_context(decomp)
            if size_ctx:
                extra_ctx += size_ctx + "\n\n"

            # Inject pre-calculated breakdown so LLM doesn't do arithmetic
            breakdown_ctx = _format_pricing_breakdown(pricing_resolution)
            if breakdown_ctx:
                extra_ctx += breakdown_ctx + "\n\n"

        # Inject feedback-based learning context (RLHF)
        feedback_store = getattr(request.app.state, "feedback_store", None)
        if feedback_store and retriever.is_ready:
            try:
                q_vec = retriever.embed_query(req.query)
                direction = getattr(decomp, "direction", "") or ""
                fb_ctx = build_feedback_context(feedback_store, q_vec, direction)
                if fb_ctx:
                    extra_ctx = fb_ctx + "\n\n" + extra_ctx
                    logger.info("Feedback context injected", lessons_matched=fb_ctx.count("["))
            except Exception as e:
                logger.warning("Feedback context failed", error=str(e))

        # Generate
        summary = await generator.generate(
            req.query, reranked, pricing_resolution,
            history=req.history, extra_context=extra_ctx,
        )

        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info("Query completed", latency_ms=latency_ms, sources=len(reranked))

        # Persist messages to chat if authenticated
        if req.chat_id and user:
            try:
                save_message(req.chat_id, user["id"], "user", req.query, mode="simple")
                save_message(req.chat_id, user["id"], "assistant", summary,
                             mode="simple", latency_ms=latency_ms)
            except Exception as e:
                logger.warning("Failed to save chat message", error=str(e))

        return HumanQueryResponse(
            summary=summary,
            latency_ms=latency_ms,
            sources_count=len(reranked),
        )

    except Exception as e:
        logger.error("Query failed", error=str(e), query=req.query[:100])
        raise HTTPException(500, f"Query failed: {str(e)}")


@router.post("/query_structured", response_model=StructuredResponse)
async def query_structured(req: QueryRequest, request: Request,
                           user: dict | None = Depends(get_optional_user)) -> StructuredResponse:
    """
    Query the RAG system and get a structured JSON response with pricing, bundle, and evidence.
    """
    t0 = time.monotonic()

    retriever = request.app.state.retriever
    reranker = request.app.state.reranker
    generator = request.app.state.generator
    pricing = request.app.state.pricing
    vision = getattr(request.app.state, "vision", None)

    if not retriever or not retriever.is_ready:
        raise HTTPException(503, "Retriever not ready.")

    # Load chat history from DB when chat_id is provided
    if req.chat_id and user:
        db_history = get_chat_history(req.chat_id, user["id"], limit=12)
        if db_history and not req.history:
            req.history = [ChatMessage(role=m["role"], content=m["content"]) for m in db_history]

    try:
        # Vision analysis (if image provided)
        vision_context = ""
        if req.image_base64 and vision and vision.is_available:
            vision_context = await vision.analyze(req.image_base64, req.image_mime_type)
            if vision_context:
                logger.info("Vision analysis prepended to structured query context")

        # --- Parse query for clarification needs ---
        parsed = parse_query(req.query)
        clarification_prefix = ""
        if parsed.needs_clarification:
            clarification_prefix = "ВАЖНО: Запрос слишком общий — не указан тип изделия, размеры или направление. Задай 1-2 уточняющих вопроса клиенту ВМЕСТО угадывания цены. Дай примерные диапазоны по типам.\n\n"

        # --- Decompose query: detect complex multi-component requests ---
        decomp = decompose(req.query)
        parametric_breakdown_schema: ParametricBreakdown | None = None
        estimate = None
        is_estimate = _is_deal_estimate_query(req.query)

        # Feedback-based learning context (RLHF)
        feedback_prefix = ""
        feedback_store = getattr(request.app.state, "feedback_store", None)
        if feedback_store and retriever.is_ready:
            try:
                q_vec = retriever.embed_query(req.query)
                direction = getattr(decomp, "direction", "") or ""
                feedback_prefix = build_feedback_context(feedback_store, q_vec, direction)
                if feedback_prefix:
                    logger.info("Feedback context injected (structured)", lessons_matched=feedback_prefix.count("["))
            except Exception as e:
                logger.warning("Feedback context failed", error=str(e))

        if decomp.is_complex and decomp.components:
            logger.info("Complex query detected — using parametric pipeline",
                        letter_text=decomp.letter_text,
                        letter_count=decomp.letter_count,
                        height_cm=decomp.height_cm,
                        linear_meters=decomp.linear_meters,
                        components=[c.type for c in decomp.components])

            result = await _handle_complex_query(
                req.query, decomp, retriever, reranker, generator, req.top_k
            )
            estimate = result["estimate"]
            reranked = result["reranked"]
            parametric_context = result["parametric_context"]

            if estimate.is_parametric:
                parametric_breakdown_schema = _build_parametric_breakdown(estimate, decomp)

            # Generate with parametric context (+ optional vision analysis + feedback)
            full_extra = parametric_context
            if clarification_prefix:
                full_extra = clarification_prefix + full_extra
            if feedback_prefix:
                full_extra = feedback_prefix + "\n\n" + full_extra
            if vision_context:
                full_extra = f"АНАЛИЗ ИЗОБРАЖЕНИЯ:\n{vision_context}\n\n---\n{full_extra}"
            pr_obj = type("PR", (), {
                "confidence": estimate.confidence,
                "estimated_value": estimate.total_estimate,
                "estimated_basis": "параметрический расчёт по компонентам",
                "price_band_min": estimate.total_min,
                "price_band_max": estimate.total_max,
                "flags": (
                    [f"Не найдено в базе: {', '.join(estimate.missing_components)}"]
                    if estimate.missing_components else []
                ),
                "risks": [],
                "is_financial_modifier": False,
            })()
            if is_estimate:
                # Inject real product names for deal_items.
                # Force catalog retrieval: general reranked may return only
                # offer/deal profiles with no product docs (fb#22/#24 root cause).
                real_names_set = {
                    doc["payload"].get("product_name")
                    for doc in reranked if doc["payload"].get("product_name")
                }
                try:
                    catalog_docs = retriever.retrieve_for_component(req.query, top_k=12)
                    for d in catalog_docs:
                        name = d["payload"].get("product_name")
                        if name:
                            real_names_set.add(name)
                except Exception as e:
                    logger.warning("Catalog retrieval for deal_items failed", error=str(e))
                real_names = sorted(n for n in real_names_set if n)
                if real_names:
                    full_extra += "\nНАЗВАНИЯ ТОВАРОВ (используй ТОЛЬКО эти названия в deal_items):\n"
                    full_extra += "\n".join(f"- {n}" for n in real_names[:50]) + "\n\n"
                raw_json = await generator.generate_deal_estimate(
                    req.query, reranked, pr_obj,
                    extra_context=full_extra, history=req.history,
                )
            else:
                raw_json = await generator.generate_structured(
                    req.query, reranked, pr_obj,
                    extra_context=full_extra, history=req.history,
                )

        else:
            # --- Standard pipeline for simple queries ---
            candidates = retriever.retrieve(req.query, top_k=req.top_k * 2)
            if not candidates:
                return StructuredResponse(
                    summary="По вашему запросу не найдено подходящих товаров/услуг.",
                    confidence="manual",
                    flags=["Нет данных в базе для данного запроса."],
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )

            reranked = reranker.rerank(req.query, candidates, top_n=req.top_k)
            pr = pricing.resolve(reranked, decomp=decomp)
            # Filter noise: only pass relevant docs to generator for pricing context
            reranked_relevant = _filter_relevant_docs(reranked)
            vision_extra = ""
            if clarification_prefix:
                vision_extra += clarification_prefix
            if feedback_prefix:
                vision_extra += feedback_prefix + "\n\n"
            if vision_context:
                vision_extra += f"АНАЛИЗ ИЗОБРАЖЕНИЯ:\n{vision_context}\n\n"
            # Add size context for structured queries with letter info
            size_ctx = _build_size_context(decomp)
            if size_ctx:
                vision_extra += size_ctx + "\n\n"
            # Inject pre-calculated breakdown
            breakdown_ctx = _format_pricing_breakdown(pr)
            if breakdown_ctx:
                vision_extra += breakdown_ctx + "\n\n"
            if is_estimate:
                # Inject real product names — force catalog retrieval when
                # general reranked misses product docs.
                real_names_set = {
                    doc["payload"].get("product_name")
                    for doc in reranked if doc["payload"].get("product_name")
                }
                try:
                    catalog_docs = retriever.retrieve_for_component(req.query, top_k=12)
                    for d in catalog_docs:
                        name = d["payload"].get("product_name")
                        if name:
                            real_names_set.add(name)
                except Exception as e:
                    logger.warning("Catalog retrieval for deal_items failed", error=str(e))
                real_names = sorted(n for n in real_names_set if n)
                if real_names:
                    vision_extra += "НАЗВАНИЯ ТОВАРОВ (используй ТОЛЬКО эти названия в deal_items):\n"
                    vision_extra += "\n".join(f"- {n}" for n in real_names[:50]) + "\n\n"
                raw_json = await generator.generate_deal_estimate(
                    req.query, reranked_relevant, pr,
                    extra_context=vision_extra, history=req.history,
                )
            else:
                raw_json = await generator.generate_structured(
                    req.query, reranked_relevant, pr,
                    extra_context=vision_extra, history=req.history,
                )

        # --- Build unified response ---
        top_payload = reranked[0]["payload"] if reranked else {}
        summary = raw_json.get("summary", "") or top_payload.get("searchable_text", req.query)[:120]
        reasoning = raw_json.get("reasoning", "")
        llm_flags = raw_json.get("flags", []) if isinstance(raw_json.get("flags"), list) else []
        llm_risks = raw_json.get("risks", []) if isinstance(raw_json.get("risks"), list) else []

        if decomp.is_complex and estimate is not None:
            # Complex path: compute under-key totals from pricing resolver
            complex_pr = pricing.resolve(reranked, decomp=decomp)

            # Prefer pre-calculated under-key totals when available
            if complex_pr.total_under_key_min is not None:
                total_mid = round((complex_pr.total_under_key_min + complex_pr.total_under_key_max) / 2)
                estimated_price = EstimatedPrice(
                    value=total_mid,
                    currency="RUB",
                    basis=complex_pr.estimated_basis + " (итого под ключ)" if complex_pr.estimated_basis else "рыночный ориентир (итого под ключ)",
                )
                price_band = PriceBand(
                    min=complex_pr.total_under_key_min,
                    max=complex_pr.total_under_key_max,
                    currency="RUB",
                )
            else:
                # Fallback to LLM's synthesized price
                llm_price = raw_json.get("estimated_price")
                llm_band = raw_json.get("price_band")

                if llm_price and isinstance(llm_price, dict) and llm_price.get("value"):
                    estimated_price = EstimatedPrice(
                        value=float(llm_price["value"]),
                        currency="RUB",
                        basis=llm_price.get("basis", "оценка по аналогичным сделкам и рынку"),
                    )
                else:
                    estimated_price = None

                if llm_band and isinstance(llm_band, dict) and llm_band.get("min") and llm_band.get("max"):
                    price_band = PriceBand(
                        min=float(llm_band["min"]),
                        max=float(llm_band["max"]),
                        currency="RUB",
                    )
                else:
                    price_band = PriceBand()

            llm_confidence = raw_json.get("confidence")
            confidence_out = llm_confidence if llm_confidence in ("auto", "guided", "manual") else (complex_pr.confidence or "manual")
            pr_flags = [f"Не найдено в базе: {', '.join(estimate.missing_components)}"] if estimate.missing_components else []
            pr_risks = []
        else:
            # Standard path: prefer under-key totals > LLM price > resolver estimate
            llm_price = raw_json.get("estimated_price")
            llm_band = raw_json.get("price_band")
            llm_confidence = raw_json.get("confidence")

            # Under-key totals available — use as authoritative price band
            if pr.total_under_key_min is not None and not pr.is_financial_modifier:
                total_mid = round((pr.total_under_key_min + pr.total_under_key_max) / 2)
                estimated_price = EstimatedPrice(
                    value=total_mid,
                    currency="RUB",
                    basis=pr.estimated_basis + " (итого под ключ)",
                )
                price_band = PriceBand(
                    min=pr.total_under_key_min,
                    max=pr.total_under_key_max,
                    currency="RUB",
                )
            # Use LLM price only when resolver can't determine a confident value
            elif llm_price and isinstance(llm_price, dict) and llm_price.get("value") and \
               pr.confidence == "manual" and not pr.is_financial_modifier:
                estimated_price = EstimatedPrice(
                    value=float(llm_price["value"]),
                    currency="RUB",
                    basis=llm_price.get("basis", "оценка на основе контекста"),
                )
                # Price band from LLM
                if llm_band and isinstance(llm_band, dict) and \
                   llm_band.get("min") and llm_band.get("max"):
                    price_band = PriceBand(
                        min=float(llm_band["min"]),
                        max=float(llm_band["max"]),
                        currency="RUB",
                    )
                else:
                    price_band = PriceBand(
                        min=pr.price_band_min,
                        max=pr.price_band_max,
                        currency="RUB",
                    )
            elif not pr.is_financial_modifier and pr.estimated_value is not None:
                estimated_price = EstimatedPrice(
                    value=pr.estimated_value,
                    currency="RUB",
                    basis=pr.estimated_basis,
                )
                price_band = PriceBand(
                    min=pr.price_band_min,
                    max=pr.price_band_max,
                    currency="RUB",
                )
            else:
                estimated_price = None
                # Price band: prefer LLM's when resolver has no data
                if llm_band and isinstance(llm_band, dict) and \
                   llm_band.get("min") and llm_band.get("max"):
                    price_band = PriceBand(
                        min=float(llm_band["min"]),
                        max=float(llm_band["max"]),
                        currency="RUB",
                    )
                else:
                    price_band = PriceBand(
                        min=pr.price_band_min,
                        max=pr.price_band_max,
                        currency="RUB",
                    )

            confidence_out = llm_confidence if llm_confidence in ("auto", "guided", "manual") else pr.confidence
            pr_flags = pr.flags
            pr_risks = pr.risks

        all_flags = list(dict.fromkeys(pr_flags + llm_flags))
        all_risks = list(dict.fromkeys(pr_risks + llm_risks))

        deal_items = []
        smeta_result = None
        if is_estimate:
            # PRIMARY (П7): SmetaEngine — deterministic template with price statistics.
            smeta_engine = getattr(request.app.state, "smeta_engine", None)
            if smeta_engine is not None and smeta_engine.is_ready:
                try:
                    q_vec_smeta = retriever.embed_query(req.query)
                    # Pass parametric decomp so SmetaEngine can scale per-letter positions
                    _decomp_for_smeta = None
                    try:
                        _decomp_local = locals().get("decomp")
                        if _decomp_local is not None:
                            _decomp_for_smeta = {
                                "letter_count": getattr(_decomp_local, "letter_count", 0) or 0,
                                "letter_text": getattr(_decomp_local, "letter_text", "") or "",
                                "height_cm": getattr(_decomp_local, "height_cm", 0) or 0,
                                "linear_meters": getattr(_decomp_local, "linear_meters", 0) or 0,
                            }
                    except Exception:
                        _decomp_for_smeta = None
                    smeta_result = smeta_engine.build_smeta(
                        req.query, q_vec_smeta, decomp=_decomp_for_smeta,
                    )
                    if smeta_result.is_usable:
                        deal_items = smeta_result.deal_items
                        logger.info("Deal estimate from SmetaEngine",
                                    category=smeta_result.category_name,
                                    quality=smeta_result.match_quality,
                                    sim=round(smeta_result.match_similarity, 3),
                                    items=len(deal_items),
                                    total=smeta_result.total)
                        # Authoritative price override from template statistics
                        estimated_price = EstimatedPrice(
                            value=smeta_result.total,
                            currency="RUB",
                            basis=f"шаблон категории «{smeta_result.category_name}» ({smeta_result.deals_in_category} сделок)",
                        )
                        price_band = PriceBand(
                            min=smeta_result.price_band_min,
                            max=smeta_result.price_band_max,
                            currency="RUB",
                        )
                        # Map smeta confidence to StructuredResponse schema
                        _smeta_conf_map = {"high": "auto", "medium": "guided", "low": "manual"}
                        confidence_out = _smeta_conf_map.get(smeta_result.confidence, "manual")
                        all_flags = [smeta_result.match_reason] + smeta_result.flags + all_flags

                        # Fix E: rewrite summary from authoritative smeta data so
                        # the textual answer cannot disagree with estimated_price.
                        try:
                            _fmt_total = f"{int(round(smeta_result.total)):,}".replace(",", " ")
                            _fmt_min = f"{int(round(smeta_result.price_band_min)):,}".replace(",", " ")
                            _fmt_max = f"{int(round(smeta_result.price_band_max)):,}".replace(",", " ")
                            _conf_ru = {"high": "высокая", "medium": "средняя", "low": "низкая"}.get(
                                smeta_result.confidence, "средняя"
                            )
                            _lines = [
                                f"Оценка по шаблону категории «{smeta_result.category_name}» "
                                f"(база: {smeta_result.deals_in_category} аналогичных сделок).",
                                f"Итого: {_fmt_total} ₽ (диапазон {_fmt_min}–{_fmt_max} ₽), "
                                f"уверенность {_conf_ru}.",
                            ]
                            if deal_items:
                                _lines.append(f"Состав сметы: {len(deal_items)} позиций.")
                            if smeta_result.flags:
                                _lines.append("⚠ " + "; ".join(smeta_result.flags[:2]))
                            summary = " ".join(_lines)
                            reasoning = (
                                f"Шаблон выбран по семантической близости запроса к категории "
                                f"«{smeta_result.category_name}» (cosine {smeta_result.match_similarity:.2f}, "
                                f"качество совпадения: {smeta_result.match_quality}). "
                                f"Цены — статистическое среднее (mean+median+weighted+trimmed) "
                                f"по {smeta_result.deals_in_category} сделкам категории."
                            )
                        except Exception as _e_sum:
                            logger.warning("Smeta summary rewrite failed", error=str(_e_sum))
                    else:
                        logger.info("SmetaEngine no match",
                                    reason=smeta_result.match_reason)
                except Exception as e:
                    logger.warning("SmetaEngine failed", error=str(e))

            # FALLBACK 1: existing DealLookup (legacy real-deal matching)
            if not deal_items:
                deal_lookup: DealLookup | None = getattr(request.app.state, "deal_lookup", None)
                matched_title = ""
                if deal_lookup and deal_lookup._loaded:
                    deal_items, matched_title = deal_lookup.find_best_deal_items(reranked)
                    if deal_items:
                        logger.info("Deal estimate from DealLookup fallback",
                                    items=len(deal_items), source_deal=matched_title[:60])
                        if matched_title:
                            all_flags = [f"Состав по аналогу: «{matched_title[:80]}»"] + all_flags

            # FALLBACK 2: LLM-generated items if no template and no real deal matched
            if not deal_items:
                deal_items = _parse_deal_items(raw_json)
                if deal_items:
                    all_flags = ["Состав сформирован моделью — нет точного аналога в базе"] + all_flags
                logger.info("Deal estimate from LLM fallback", item_count=len(deal_items))

            # --- PRICE RECONCILIATION (SINGLE SOURCE OF TRUTH) ---
            # When deal_items exist, their sum becomes authoritative: estimated_price,
            # price_band and summary must agree. This prevents the 4-number disagreement
            # (summary / deal_items / estimated_price / parametric_breakdown).
            if deal_items:
                items_total = sum((di.total or 0) for di in deal_items)
                if items_total > 0:
                    current_val = estimated_price.value if estimated_price else 0
                    drift = abs(items_total - current_val) / max(items_total, 1)
                    if current_val == 0 or drift > 0.05:
                        logger.info("Price reconciliation applied",
                                    old=current_val, new=items_total,
                                    items=len(deal_items), drift=round(drift, 2))
                        estimated_price = EstimatedPrice(
                            value=round(items_total),
                            currency="RUB",
                            basis="сумма позиций сметы (deal_items)",
                        )
                        price_band = PriceBand(
                            min=round(items_total * 0.85),
                            max=round(items_total * 1.15),
                            currency="RUB",
                        )
                        # Parametric breakdown was computed independently — stale now
                        if parametric_breakdown_schema is not None:
                            all_flags = ["Параметрический расчёт заменён сметой"] + all_flags
                            parametric_breakdown_schema = None

        response = StructuredResponse(
            summary=summary,
            suggested_bundle=_build_suggested_bundle(reranked),
            estimated_price=estimated_price,
            price_band=price_band,
            confidence=confidence_out,
            reasoning=reasoning,
            flags=all_flags,
            risks=all_risks,
            references=_build_references(reranked),
            segmented_references=_build_segmented_references(
                reranked,
                photo_index=getattr(request.app.state, "photo_index", None),
            ),
            source_distinction=_detect_source_distinction(reranked),
            parametric_breakdown=parametric_breakdown_schema,
            deal_items=deal_items,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

        logger.info("Structured query completed",
                    latency_ms=response.latency_ms,
                    confidence=response.confidence,
                    sources=len(reranked))

        # Persist messages to chat if authenticated
        if req.chat_id and user:
            try:
                save_message(req.chat_id, user["id"], "user", req.query, mode="structured")
                structured_data = response.model_dump(exclude={"references"})
                save_message(req.chat_id, user["id"], "assistant", response.summary,
                             mode="structured", structured_data=structured_data,
                             latency_ms=response.latency_ms)
            except Exception as e:
                logger.warning("Failed to save structured chat message", error=str(e))

        return response

    except Exception as e:
        logger.error("Structured query failed", error=str(e), query=req.query[:100])
        raise HTTPException(500, f"Structured query failed: {str(e)}")
