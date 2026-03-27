"""Query endpoints: /query and /query_structured."""
import time
from fastapi import APIRouter, Request, HTTPException, Depends
from app.schemas.query import QueryRequest, HumanQueryResponse, StructuredResponse, ChatMessage
from app.schemas.pricing import (
    PriceBand, EstimatedPrice, BundleItem, Reference, SourceDistinction,
    ParametricBreakdown, ParametricLineItem, DealItem,
)
from app.core.query_decomposer import decompose, QueryDecomposition
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
        if is_estimate:
            # Primary: look up real line items from matching deal in CSV
            deal_lookup: DealLookup | None = getattr(request.app.state, "deal_lookup", None)
            matched_title = ""
            if deal_lookup and deal_lookup._loaded:
                deal_items, matched_title = deal_lookup.find_best_deal_items(reranked)
                if deal_items:
                    logger.info("Deal estimate from real data",
                                items=len(deal_items), source_deal=matched_title[:60])
                    if matched_title:
                        all_flags = [f"Состав по аналогу: «{matched_title[:80]}»"] + all_flags

            # Fallback: use LLM-generated items if no real deal found
            if not deal_items:
                deal_items = _parse_deal_items(raw_json)
                if deal_items:
                    all_flags = ["Состав сформирован моделью — нет точного аналога в базе"] + all_flags
                logger.info("Deal estimate from LLM fallback", item_count=len(deal_items))

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
