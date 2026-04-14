"""Query request and response Pydantic schemas."""
from typing import Any, Literal
from pydantic import BaseModel, Field
from app.schemas.pricing import (
    PriceBand, EstimatedPrice, BundleItem, Reference, SourceDistinction,
    ParametricBreakdown, DealItem, SegmentedReferences,
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
    latency_ms: int = 0


class RebuildIndexRequest(BaseModel):
    doc_types: list[Literal["product", "bundle", "policy", "support"]] | None = None
    recreate_collection: bool = False


class RebuildIndexResponse(BaseModel):
    status: str
    message: str
    doc_types: list[str]
