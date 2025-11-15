"""
Core configuration for the Wellbore AI Agent application.

This module centralizes all application settings using Pydantic for type safety
and validation. Settings are loaded from environment variables.

Philosophy:
- Only hardcode truly universal constants (app name, version)
- Development defaults for convenience (localhost, debug mode)
- All model/API configs MUST come from .env
"""
import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Required settings (no defaults) will raise ValidationError if not in .env
    Optional settings have sensible defaults for local development.
    """

    # ==================== Application Constants ====================
    # These rarely/never change - safe to hardcode
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
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # ==================== File Storage ====================
    # Path structure - can be customized via .env
    DATA_DIR: Path = Path("../data")

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

    # ==================== Embedding Configuration ====================
    # Embedding model - must be specified
    EMBEDDING_MODEL: str  # e.g., "sentence-transformers/all-MiniLM-L6-v2"

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

    # ==================== Agent Configuration ====================
    MAX_AGENT_ITERATIONS: int = 5
    AGENT_STREAMING: bool = True

    # ==================== Nodal Analysis API ====================
    # Must be configured in .env
    NODAL_API_URL: str  # e.g., "http://localhost:8001/api/nodal-analysis"
    NODAL_API_TIMEOUT: int = 30
    NODAL_API_MOCK: bool = True  # Switch to False when real API is ready

    # ==================== Performance & Resource Limits ====================
    MAX_CONCURRENT_REQUESTS: int = 3
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MESSAGE_QUEUE_SIZE: int = 100

    # ==================== Logging ====================
    LOG_FORMAT: str = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"

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
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories on initialization."""
        directories = [
            self.DATA_DIR,
            self.UPLOAD_DIR,
            self.PROCESSED_DIR,
            self.RAW_DIR,
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
        if not self.OLLAMA_BASE_URL:
            errors.append("OLLAMA_BASE_URL is required")
        if not self.OLLAMA_MODEL:
            errors.append("OLLAMA_MODEL is required")
        if not self.HF_TOKEN:
            errors.append("HF_TOKEN is required")
        # Check embedding config
        if not self.EMBEDDING_MODEL:
            errors.append("EMBEDDING_MODEL is required")
        # Check nodal API config
        if not self.NODAL_API_URL:
            errors.append("NODAL_API_URL is required")

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


def validate_ollama_connection() -> bool:
    """
    Check if Ollama is running and accessible.

    Returns:
        bool: True if Ollama is accessible, False otherwise
    """
    import httpx

    try:
        response = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def validate_ollama_model() -> bool:
    """
    Check if the configured model is available in Ollama.

    Returns:
        bool: True if model exists, False otherwise
    """
    import httpx

    try:
        response = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            return settings.OLLAMA_MODEL in model_names
        return False
    except Exception:
        return False


if __name__ == "__main__":
    """Test configuration loading."""
    print("=" * 60)
    print("WELLBORE AI AGENT - Configuration Test")
    print("=" * 60)

    # Check for validation errors
    errors = settings.validate_required_settings()
    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        print("\n💡 Please check your .env file")
        exit(1)

    # Display key settings
    print(f"\n📱 Application:")
    print(f"   Name: {settings.APP_NAME}")
    print(f"   Version: {settings.APP_VERSION}")
    print(f"   Debug: {settings.DEBUG}")

    print(f"\n🤖 LLM Settings:")
    print(f"   Ollama URL: {settings.OLLAMA_BASE_URL}")
    print(f"   HF Token: {settings.HF_TOKEN[:5] + ("*" * 10)}")
    print(f"   Model: {settings.OLLAMA_MODEL}")
    print(f"   Temperature: {settings.LLM_TEMPERATURE}")

    print(f"\n🔤 Embedding Settings:")
    print(f"   Model: {settings.EMBEDDING_MODEL}")
    print(f"   Dimension: {settings.EMBEDDING_DIMENSION}")

    print(f"\n📁 Storage Paths:")
    print(f"   Upload Dir: {settings.UPLOAD_DIR}")
    print(f"   Vector DB: {settings.VECTOR_DB_DIR}")

    print(f"\n🔍 RAG Settings:")
    print(f"   Chunk Size: {settings.CHUNK_SIZE}")
    print(f"   Top K: {settings.RETRIEVAL_TOP_K}")

    print(f"\n🔌 Validating Ollama Connection...")
    if validate_ollama_connection():
        print("   ✅ Ollama is running")

        if validate_ollama_model():
            print(f"   ✅ Model '{settings.OLLAMA_MODEL}' is available")
        else:
            print(f"   ⚠️  Model '{settings.OLLAMA_MODEL}' not found")
            print(f"   💡 Run: ollama pull {settings.OLLAMA_MODEL}")
    else:
        print("   ❌ Ollama is not accessible")
        print("   💡 Run: ollama serve")

    print("\n" + "=" * 60)