"""
Core configuration for the Wellbore AI Agent application.

This module centralizes all application settings using Pydantic for type safety
and validation. Settings are loaded from environment variables.
"""
import os

from langchain_community.document_loaders.notiondb import DATABASE_URL
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # ... (existing settings)

    # ==================== Database Configuration (SQLite) ====================
    DATABASE_URL: str = f"sqlite:///'data' / 'wellbore_agent.db'"

    # ==================== Pydantic Settings ====================
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set default for DATABASE_URL if not provided
        if 'DATABASE_URL' not in kwargs:
            self.DATABASE_URL = self.DATABASE_URL = f"sqlite:///{self.DATA_DIR / 'wellbore_agent.db'}"
        self._create_directories()

    # ... (the rest of the existing file)
    # ...
    # Make sure to copy the rest of the file content here
    # ...
    APP_NAME: str = "Wellbore AI Agent"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    # ==================== Development Defaults ====================
    # Safe defaults for local development
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # CORS - Development default
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # ==================== File Storage ====================
    # Path structure - can be customized via .env
    DATA_DIR: Path = Path("data")

    @property
    def UPLOAD_DIR(self) -> Path:
        return self.DATA_DIR / "uploads"

    @property
    def PROCESSED_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def RAW_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def FAILED_DIR(self) -> Path:
        return self.DATA_DIR / "failed"

    @property
    def VECTOR_DB_DIR(self) -> Path:
        return self.DATA_DIR / "vector_db"

    # File upload constraints
    MAX_UPLOAD_SIZE: int = 52428800  # 50MB in bytes
    ALLOWED_EXTENSIONS: set[str] = {".pdf"}

    # ==================== LLM Configuration (Ollama) ====================
    # These MUST be set in .env - no defaults
    LLM_PROVIDER: str = "ollama"
    # LLM_API_KEY: str = "<KEY>"
    HF_TOKEN: str
    OLLAMA_BASE_URL: str  # e.g., "http://localhost:11434"
    OLLAMA_MODEL: str  # e.g., "phi3:mini"

    # LLM parameters - can be tuned via .env
    OLLAMA_TIMEOUT: int = 120
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048
    LLM_TOP_P: float = 0.9

    # Huggingface
    HF_TOKEN: str
    # Gemini
    GEMINI_API_KEY: str
    # Groq
    GROQ_API_KEY: str

    # ==================== Embedding Configuration ====================
    # Embedding model - must be specified
    EMBEDDING_MODEL: str  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str  # e.g., "sentence-transformers/all-MiniLM-L6-v2"

    # Model metadata (match your chosen model)
    EMBEDDING_DIMENSION: int = 384  # for all-MiniLM-L6-v2
    EMBEDDING_DEVICE: str = "cpu"

    # ==================== Vector Store (ChromaDB) ====================
    CHROMA_COLLECTION_NAME: str = "wellbore_documents"
    CHROMA_DISTANCE_METRIC: str = "cosine"

    # Retrieval parameters
    RETRIEVAL_TOP_K: int = 10
    RETRIEVAL_SCORE_THRESHOLD: float = 0.4

    # ==================== Document Processing ====================
    # Text chunking parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # PDF processing flags
    PDF_EXTRACT_IMAGES: bool = True
    PDF_EXTRACT_TABLES: bool = True
    OCR_CONFIDENCE_THRESHOLD: float = 0.6
    # ==================== Agent Configuration ====================
    MAX_AGENT_ITERATIONS: int = 5
    AGENT_STREAMING: bool = True

    # ==================== Performance & Resource Limits ====================
    MAX_CONCURRENT_REQUESTS: int = 3
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MESSAGE_QUEUE_SIZE: int = 100

    # ==================== Logging ====================
    LOG_FORMAT: str = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"

    def _create_directories(self) -> None:
        """Create necessary directories on initialization."""
        directories = [
            self.DATA_DIR,
            self.UPLOAD_DIR,
            self.PROCESSED_DIR,
            self.RAW_DIR,
            self.FAILED_DIR,
            self.VECTOR_DB_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate_required_settings(self) -> list[str]:
        """
        Validate that all required external dependencies are configured.

        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []

        # Check Ollama config
        if self.LLM_PROVIDER == "ollama":
            if not self.OLLAMA_BASE_URL:
                errors.append("OLLAMA_BASE_URL is required")
            if not self.OLLAMA_MODEL:
                errors.append("OLLAMA_MODEL is required")

        elif self.LLM_PROVIDER == "huggingface":
            if not self.HF_TOKEN:
                errors.append("HF_TOKEN is required")

        elif self.LLM_PROVIDER == "gemini":
            if not self.GEMINI_API_KEY:
                errors.append("GEMINI_API_KEY is required")

        elif self.LLM_PROVIDER == "groq":
            if not self.GROQ_API_KEY:
                errors.append("GROQ_API_KEY is required")

            # Embedding config (common requirement)
        if not self.EMBEDDING_MODEL:
            errors.append("EMBEDDING_MODEL is required")

        # Reranker config (common requirement)
        if not self.RERANKER_MODEL:
            errors.append("EMBEDDING_MODEL is required")

        return errors


# ==================== Global Settings Instance ====================
settings = Settings()
from functools import lru_cache


# ==================== Helper Functions ====================
@lru_cache()
def get_settings() -> Settings:
    """
    Dependency function for FastAPI routes.

    Usage:
        @app.get("/config")
        def get_config(settings: Settings = Depends(get_settings)):
            return {"model": settings.OLLAMA_MODEL}
    """
    return settings