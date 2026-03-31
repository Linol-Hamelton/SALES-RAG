"""Health check endpoint."""
import torch
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    embedding_model_loaded: bool
    reranker_loaded: bool
    cuda_available: bool
    device: str
    gpu_name: str | None
    collection: str | None


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health(request: Request) -> HealthResponse:
    """System health check."""
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else None
    device = "cuda" if cuda else "cpu"

    # Check Qdrant connection
    qdrant_ok = False
    collection_name = None
    try:
        retriever = getattr(request.app.state, "retriever", None)
        if retriever and retriever._client:
            info = retriever._client.get_collections()
            qdrant_ok = True
            collection_name = getattr(retriever.settings, "qdrant_collection", None)
    except Exception:
        pass

    retriever_loaded = getattr(request.app.state, "retriever", None) is not None
    reranker_loaded = getattr(request.app.state, "reranker", None) is not None

    return HealthResponse(
        status="ok",
        qdrant_connected=qdrant_ok,
        embedding_model_loaded=retriever_loaded,
        reranker_loaded=reranker_loaded,
        cuda_available=cuda,
        device=device,
        gpu_name=gpu_name,
        collection=collection_name,
    )
