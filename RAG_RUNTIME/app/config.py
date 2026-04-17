"""Application configuration using Pydantic Settings."""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="configs/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Deepseek API
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    # Vision API (for image analysis — OpenAI-compatible)
    vision_api_key: str = Field(default="", alias="VISION_API_KEY")
    vision_base_url: str = Field(default="https://api.artemox.com/v1", alias="VISION_BASE_URL")
    vision_model: str = Field(default="gemini-3-flash", alias="VISION_MODEL")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="labus_docs", alias="QDRANT_COLLECTION")

    # Model paths (relative to RAG_RUNTIME root)
    embedding_model_path: str = Field(default="models/embeddings/BAAI_bge-m3", alias="EMBEDDING_MODEL_PATH")
    reranker_model_path: str = Field(default="models/reranker/BAAI_bge-reranker-v2-m3", alias="RERANKER_MODEL_PATH")

    # Data dirs
    data_dir: str = Field(default="data", alias="DATA_DIR")
    index_dir: str = Field(default="indexes", alias="INDEX_DIR")
    log_dir: str = Field(default="logs", alias="LOG_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    # Analytics output dir (relative to RAG_RUNTIME parent, i.e. SALES_RAG root)
    analytics_output_dir: str = Field(
        default="../RAG_ANALYTICS/output",
        alias="ANALYTICS_OUTPUT_DIR",
    )

    # Feature flags
    use_intent_classifier: bool = Field(default=True, alias="USE_INTENT_CLASSIFIER")

    # Bitrix24 — URL template used to link deals/offers back to the CRM.
    # {id} is the offers.csv/goods.csv ID (== Bitrix deal ID).
    bitrix_deal_url_template: str = Field(
        default="https://labus.bitrix24.ru/crm/deal/details/{id}/",
        alias="BITRIX_DEAL_URL_TEMPLATE",
    )

    # Retrieval
    retrieval_top_k: int = Field(default=20, alias="RETRIEVAL_TOP_K")
    rerank_top_n: int = Field(default=8, alias="RERANK_TOP_N")
    rrf_alpha: float = Field(default=0.7, alias="RRF_ALPHA")

    @property
    def project_root(self) -> Path:
        """RAG_RUNTIME root directory."""
        return Path(__file__).parent.parent

    @property
    def analytics_output_path(self) -> Path:
        return (self.project_root / self.analytics_output_dir).resolve()

    @property
    def data_path(self) -> Path:
        return self.project_root / self.data_dir

    @property
    def index_path(self) -> Path:
        return self.project_root / self.index_dir

    @property
    def embedding_model_full_path(self) -> Path:
        return self.project_root / self.embedding_model_path

    @property
    def reranker_model_full_path(self) -> Path:
        return self.project_root / self.reranker_model_path


settings = Settings()
