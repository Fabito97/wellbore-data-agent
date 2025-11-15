from fastapi import APIRouter
from app.api.routes import documents, chat, health, websocket

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["Agent"])
api_router.include_router(documents.router,prefix="/documents", tags=["documents"])
api_router.include_router(websocket.router, prefix="/ws", tags=["websocket"])


__all__ = ["api_router", "health"]