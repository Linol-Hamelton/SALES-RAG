"""Query request and response Pydantic schemas."""
from typing import Any, Literal
from pydantic import BaseModel, Field
from app.schemas.pricing import (
    PriceBand, EstimatedPrice, BundleItem, Reference, SourceDistinction,
    ParametricBreakdown, DealItem, SegmentedReferences, HistoricalDealRef,
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=10000)


class QueryRequest(BaseModel):
    # P9: расширено до 10000 символов — чтобы клиентские письма влезали целиком
    query: str = Field(..., min_length=1, max_length=10000, description="User query in Russian or English")
    top_k: int = Field(default=8, ge=1, le=20, description="Number of results to retrieve")
    language: Literal["ru", "en", "auto"] = "auto"
    history: list[ChatMessage] = Field(default_factory=list, description="Previous chat turns (last 6 max)")
    image_base64: str | None = Field(default=None, description="Base64-encoded image for visual analysis")
    image_mime_type: str = Field(default="image/jpeg", description="MIME type of the uploaded image")
    chat_id: int | None = Field(default=None, description="Chat ID for persisting messages (requires auth)")


class HumanQueryResponse(BaseModel):
    summary: str
    latency_ms: int = 0
    sources_count: int = 0


class StructuredResponse(BaseModel):
    summary: str = ""
    suggested_bundle: list[BundleItem] = []
    estimated_price: EstimatedPrice | None = None
    price_band: PriceBand = Field(default_factory=PriceBand)
    confidence: Literal["auto", "guided", "manual"] = "manual"
    reasoning: str = ""
    flags: list[str] = []
    risks: list[str] = []
    references: list[Reference] = []
    segmented_references: SegmentedReferences = Field(default_factory=SegmentedReferences)
    source_distinction: SourceDistinction = Field(default_factory=SourceDistinction)
    parametric_breakdown: ParametricBreakdown | None = None
    deal_items: list[DealItem] = []   # populated when query is a deal-estimate request
    historical_deals: list[HistoricalDealRef] = []   # P13.3 / T7: similar closed deals from orders.csv
    estimated_lead_time: str | None = None   # P14.3.3: «3-5 рабочих дней», «1-2 недели», «до завтра»
    latency_ms: int = 0


class NoRagRequest(BaseModel):
    """P14: pure-LLM bypass — no retrieval, no SmetaEngine, no dialog_state.
    Used by SOFT_TUNE_DATA evaluator to measure RAG-uplift quantitatively."""
    query: str = Field(..., min_length=1, max_length=10000)
    mode: Literal["human", "structured"] = "structured"
    system_prompt_mode: Literal["full", "minimal", "custom"] = "full"
    """`full` = тот же system prompt что у /query_structured (изолирует вклад retrieval).
    `minimal` = generic Russian assistant prompt (показывает baseline без любого SALES_RAG приминга).
    `custom` = используй custom_system_prompt (для evaluator-сценариев типа генерации вопросов)."""
    custom_system_prompt: str | None = Field(default=None, max_length=20000)
    history: list[ChatMessage] = Field(default_factory=list)
    model_override: str | None = Field(default=None, max_length=128,
        description="Использовать другую DeepSeek-модель (например 'deepseek-reasoner' для evaluator-фаз).")
    response_format_override: Literal["json", "text"] | None = Field(default=None,
        description="Принудительно задать response_format. По умолчанию: structured=json, human=text.")
    temperature: float = Field(default=0.15, ge=0.0, le=2.0)
    max_tokens_override: int | None = Field(default=None, ge=1, le=32000,
        description="Если задан — переопределяет max_tokens из settings (для длинных генераций).")


class NoRagResponse(BaseModel):
    summary: str
    raw_response: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    system_prompt_mode: str = "full"


class RebuildIndexRequest(BaseModel):
    doc_types: list[Literal["product", "bundle", "policy", "support"]] | None = None
    recreate_collection: bool = False


class RebuildIndexResponse(BaseModel):
    status: str
    message: str
    doc_types: list[str]
