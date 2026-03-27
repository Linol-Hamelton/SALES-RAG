"""FastAPI dependency injection helpers."""
from fastapi import Request
from app.core.retriever import HybridRetriever
from app.core.reranker import CrossEncoderReranker
from app.core.generator import DeepseekGenerator
from app.core.pricing_resolver import PricingResolver


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever


def get_reranker(request: Request) -> CrossEncoderReranker:
    return request.app.state.reranker


def get_generator(request: Request) -> DeepseekGenerator:
    return request.app.state.generator


def get_pricing(request: Request) -> PricingResolver:
    return request.app.state.pricing
