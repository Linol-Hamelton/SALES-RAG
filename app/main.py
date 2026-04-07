"""
FastAPI application factory for Labus Sales RAG system.

Start with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

IMPORTANT: workers=1 is mandatory. BGE-M3 is loaded into process memory.
Multiple workers would each load the model separately, exhausting RAM/VRAM.
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path when running with uvicorn
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.utils.logging import setup_logging, get_logger
from app.core.retriever import HybridRetriever
from app.core.reranker import CrossEncoderReranker
from app.core.generator import DeepseekGenerator
from app.core.pricing_resolver import PricingResolver
from app.core.vision import VisionAnalyzer
from app.core.feedback_store import FeedbackStore
from app.core.deal_lookup import DealLookup
from app.core.photo_index import PhotoIndex
from app.database import init_db
from app.routers import health, query, admin, eval as eval_router
from app.routers import auth_router, chats

setup_logging(settings.log_level)
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup, clean up on shutdown."""
    logger.info("Starting Labus Sales RAG", version="1.1.0")

    # Initialize SQLite database
    init_db()
    logger.info("Database initialized")

    # Initialize core components
    retriever = HybridRetriever(settings)
    reranker = CrossEncoderReranker(settings)
    generator = DeepseekGenerator(settings)
    pricing = PricingResolver(settings)
    vision = VisionAnalyzer(settings)

    # Load models (lazy — will load on first request, or eagerly here)
    try:
        logger.info("Loading retriever (embedding model + Qdrant + BM25)...")
        retriever.load()
        logger.info("Loading reranker...")
        reranker.load()
        logger.info("Initializing generator (Deepseek API)...")
        generator.load()
        logger.info("Initializing vision analyzer...")
        vision.load()
    except Exception as e:
        logger.error("Component load failed", error=str(e))
        # Don't crash — allow health check to report unhealthy state

    # Initialize feedback store (RLHF learning from user feedback)
    feedback_store = FeedbackStore()
    try:
        feedback_store.load()
        logger.info("Feedback store loaded")
    except Exception as e:
        logger.error("Feedback store load failed", error=str(e))

    # Initialize deal lookup (real line items from normalized CSV)
    deal_lookup = DealLookup(settings.analytics_output_path)
    try:
        deal_lookup.load()
        logger.info("Deal lookup loaded")
    except Exception as e:
        logger.error("Deal lookup load failed", error=str(e))
        deal_lookup = None

    # Store in app state
    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.generator = generator
    app.state.pricing = pricing
    app.state.vision = vision
    app.state.feedback_store = feedback_store
    app.state.deal_lookup = deal_lookup

    # Initialize photo index (visual enrichment for segmented references)
    try:
        photo_index = PhotoIndex(Path(settings.data_path) / "photo_analysis_raw.jsonl").load()
        logger.info("Photo index loaded", deals=len(photo_index))
    except Exception as e:
        logger.error("Photo index load failed", error=str(e))
        photo_index = None
    app.state.photo_index = photo_index

    logger.info("Application ready")
    yield

    # Cleanup
    logger.info("Shutting down...")
    # Release embedding model memory
    if hasattr(retriever, "_model") and retriever._model is not None:
        del retriever._model


def create_app() -> FastAPI:
    app = FastAPI(
        title="Labus Sales RAG",
        description="Local RAG system for product/bundle lookup, pricing, and sales quoting.",
        version="1.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://ai.labus.pro",
            "http://localhost:8000",    # local dev
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router)
    app.include_router(auth_router.router)
    app.include_router(chats.router)
    app.include_router(query.router)
    app.include_router(admin.router)
    app.include_router(eval_router.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/chat")

    static_dir = PROJECT_ROOT / "app" / "static"
    app.mount("/chat", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1, reload=False)
