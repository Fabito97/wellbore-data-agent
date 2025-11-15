from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.services.llm_service import get_llm_service
    from app.rag.vector_store import get_vector_store

    logger.info("Checking system health...")

    health_status = {
        "status": "healthy",
        "services": {}
    }

    # Check LLM
    try:
        logger.debug(f"Checking llm service connection...")
        llm = get_llm_service()
        llm_healthy = llm.validate_connection()
        if llm_healthy:
            logger.error("LLM service connection successfully established")
            health_status["services"]["llm"] = "healthy"
        else: logger.warning("Failed to connect to LLM service")
    except:
        logger.error("Failed to connect to llm service")
        health_status["services"]["llm"] = "unhealthy"
        health_status["status"] = "degraded"

    # Check Vector Store
    try:
        logger.debug(f"Checking vector store connection...")
        store = get_vector_store()
        store.get_stats()
        health_status["services"]["vector_store"] = "healthy"
        logger.info(f"Vector store connection successfully established")
    except:
        logger.error("Failed to connect to vector store...")
        health_status["services"]["vector_store"] = "unhealthy"
        health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] == "healthy" else 503
    logger.info(f"Health check endpoint received: {status_code}")
    return JSONResponse(content=health_status, status_code=status_code)