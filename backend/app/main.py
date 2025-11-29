"""
Main FastAPI Application.

This is the entry point for the backend server.
It configures and runs the complete API.
"""
from fastapi import FastAPI
import dotenv

from app.api.middleware.cors import setup_cors
from app.api.routes import api_router, health
from app.api.middleware.error_handler import ErrorMiddleware, middleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.db.database import create_db_and_tables
from app.utils import get_logger

dotenv.load_dotenv()

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")

    # Initialize database
    create_db_and_tables()

    # Validate services
    logger.info("Validating services...")

    # Check Ollama connection
    from app.services.llm_service import get_llm_service
    try:
        logger.debug("Validating llm service...")
        llm = get_llm_service()
        if llm.validate_connection():
            logger.info(f"✅ LLM service ready (model: {settings.OLLAMA_MODEL})")
        else:
            logger.warning("⚠️  LLM service not responding - check Ollama")
    except Exception as e:
        logger.error(f"❌ LLM service error: {e}")

    # Check vector store
    from app.rag.vector_store_manager import get_vector_store
    try:
        logger.debug("Validating vector store...")
        store = get_vector_store()
        stats = store.get_stats()
        logger.info(f"✅ Vector store ready ({stats['total_chunks']} chunks indexed)")
    except Exception as e:
        logger.error(f"❌ Vector store error: {e}")

    logger.info(f"🚀 Server ready at http://{settings.HOST}:{settings.PORT}")

    yield  # Server runs here

    # Shutdown
    logger.info("Shutting down gracefully...")


# Create FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title = settings.APP_NAME,
    description = "AI-powered wellbore analysis system using RAG",
    version = settings.APP_VERSION,
    debug=settings.DEBUG,
    middleware=middleware,
)

# Error middleware
app.add_middleware(ErrorMiddleware)

# CORS middleware
setup_cors(app)

# Routes
api_router.include_router(health.router, tags=["health"], prefix="")
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

@app.get("/")
async def root():
    """Root endpoint - API information."""
    logger.info(f"Root responding...")
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,  # Auto-reload in dev mode
            log_level=settings.LOG_LEVEL.lower()
        )
    except KeyboardInterrupt:
        logger.warning("Shutdown requested")
        logger.info("Goodbye")