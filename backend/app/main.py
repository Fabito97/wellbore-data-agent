from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware.cors import setup_middleware
from app.api.routes import api_router
import requests
import time
from app.api.middleware.error_handler import middleware
from contextlib import asynccontextmanager
from app.core.config import settings, get_settings
from app.utils import get_logger
from app.telemetry.setup import setup_telemetry

setup_telemetry("wellbore-agent")

logger = get_logger(__name__)
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        try:
            response = requests.get("http://localhost:11434")
            if response.status_code == 200:
                print("Ollama is running")

        except requests.exceptions.ConnectionError:
            print("Ollama did not start in time.")
        yield
        return
    except Exception as e:
        print(f"Startup error: {e}")

    yield


# # Create FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title = settings.APP_NAME,
    description = "API for wellbore data analysis using AI agents",
    version = settings.APP_VERSION,
    middleware = middleware
)

# CORS middleware
setup_middleware(app)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)

@app.get("/")
async def root():
    logger.info(f"Starting {settings.APP_NAME}")
    return {
        "message": "Wellbore Data Agent API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "uptime": time.time()}

