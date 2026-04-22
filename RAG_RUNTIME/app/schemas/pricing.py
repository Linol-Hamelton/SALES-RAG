"""Pricing-related Pydantic schemas."""
from typing import Literal
from pydantic import BaseModel, Field


class ParametricLineItem(BaseModel):
    component: str = ""
    label: str = ""
    product_name: str = ""
    unit_price: float = 0.0
    quantity: float = 0.0
    unit: str = ""
    total: float = 0.0
    confidence_tier: str = ""
    quantity_basis: str = ""


class ParametricBreakdown(BaseModel):
    line_items: list[ParametricLineItem] = []
    total_estimate: float = 0.0
    total_min: float = 0.0
    total_max: float = 0.0
    missing_components: list[str] = []
    letter_text: str = ""
    letter_count: int = 0
    height_cm: float = 0.0
    linear_meters: float = 0.0


class PriceBand(BaseModel):
    min: float | None = None
    max: float | None = None
    currency: Literal["RUB"] = "RUB"


class EstimatedPrice(BaseModel):
    value: float | None = None
    currency: Literal["RUB"] = "RUB"
    basis: str = ""   # "P50 по заказам", "медиана набора", "базовый каталог", "аналог: ..."


class BundleItem(BaseModel):
    product_key: str = ""
    product_name: str = ""
    unit_price: float | None = None
    direction: str = ""


class Reference(BaseModel):
    doc_id: str = ""
    doc_type: str = ""
    score: float = 0.0
    snippet: str = ""   # first 120 chars of searchable_text
    article_id: str | None = None
    product_name: str | None = None
    direction: str | None = None
    bitrix_url: str | None = None   # Clickable link to the deal/offer in Bitrix24


class SourceSegment(BaseModel):
    """A single enriched source card shown in segmented references UI."""
    kind: Literal["order", "offer", "product_visual"] = "order"
    deal_id: str | None = None
    deal_url: str | None = None           # Clickable link to the deal in Bitrix24
    title: str = ""
    subtitle: str | None = None           # product_type + dimensions
    direction: str | None = None
    total: float | None = None
    duration_days: int | None = None
    snippet: str = ""
    score: float = 0.0
    image_urls: list[str] = Field(default_factory=list)
    product_type: str | None = None
    application: str | None = None
    roi_hint: str | None = None


class SegmentedReferences(BaseModel):
    """Three thematic lists rendered under every assistant response."""
    similar_orders: list[SourceSegment] = Field(default_factory=list)
    similar_offers: list[SourceSegment] = Field(default_factory=list)
    product_links: list[SourceSegment] = Field(default_factory=list)


class SourceDistinction(BaseModel):
    has_order_data: bool = False
    has_offer_data: bool = False
    dataset_type: Literal["orders", "offers", "both", "catalog_only", "unknown"] = "unknown"
    data_freshness_note: str = ""


class DealItem(BaseModel):
    """A single line item for a Bitrix24 deal (сделка)."""
    product_name: str = ""
    quantity: float = 1.0
    unit: str = "шт"
    unit_price: float = 0.0
    total: float = 0.0
    b24_section: str = ""   # Цех | Дизайн | Печатная | РИК | Мерч | Сольвент
    notes: str = ""         # hint for the manager
    source_deal_id: str | None = None   # Bitrix deal where this line item came from
    source_deal_url: str | None = None  # Clickable link to source deal in Bitrix24


class HistoricalDealRef(BaseModel):
    """P13.3 / T7: anonymized closed-deal reference from orders.csv.

    Surfaced under "Похожие закрытые сделки" for pricing/sizing intents
    so the manager can cite real precedents. PII (company/contact names)
    is NEVER populated.
    """
    deal_id: str = ""
    deal_title: str = ""             # internal Bitrix label, no PII
    year: int = 0
    total_price: float = 0.0
    price_bucket: str = ""           # "5–25К", "25–100К", etc.
    sections: list[str] = Field(default_factory=list)   # signature_sections
    directions: list[str] = Field(default_factory=list)
    items_preview: list[str] = Field(default_factory=list)   # top 3-5 product names
    items_count: int = 0
    score: float = 0.0
    bitrix_url: str | None = None    # /crm/deal/details/{id}/
