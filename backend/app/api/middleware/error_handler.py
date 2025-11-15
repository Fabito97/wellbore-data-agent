from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.middleware import Middleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.utils import get_logger

logger = get_logger("error-handler")

class ErrorMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.exception("Unhandled exception occurred")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Sever Error",
                    "detail": str(e) if settings.DEBUG else "An unexpected error occurred"
                }
            )

middleware = [Middleware(ErrorMiddleware)]