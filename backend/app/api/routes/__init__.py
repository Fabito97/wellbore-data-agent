from fastapi import APIRouter
from app.api.routes import documents, chat, health, chat_ws

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["Agent"])
api_router.include_router(documents.router,prefix="/documents", tags=["documents"])
api_router.include_router(chat_ws.router, prefix="/ws", tags=["websocket"])


__all__ = ["api_router", "health"]