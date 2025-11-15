import time
from typing import Dict, Any

import httpx
from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse

from app.agents.simple_agent import SimpleAgent, get_simple_agent
from ollama import Client

from app.services.llm_service import get_llm_service, LLMProvider
from app.utils.logger import get_logger
# client = Client()

router = APIRouter()
logger = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"

@router.post("/")
async def handle_chat(query: str = Form(...)):
    logger.info(f"Chat request received: {query}")
    try:
        start = time.time()

        service = get_llm_service()

        messages = [
            {"role": "user", "content": query}
        ]
        response = service.generate(prompt=query)

        result = response.content
        end = time.time()

        logger.info(f"Returning response...")
        logger.info(f"Ollama response time: {end - start:.2f}s")

        return {
            "status": "success",
            "message": "Chat generated successfully",
            "data": result,
        }

    except Exception as e:
        logger.exception("Unhandled exception in chat endpoint")
        return {
            "status": "error",
            "message": "Internal server error",
            "detail": str(e)
        }


@router.post("/ask")
async def ask_question(
        question: str,
        include_sources: bool = True,
        agent: SimpleAgent = Depends(get_simple_agent)
):
    """
    Simple REST endpoint for Q&A (alternative to WebSocket).

    Args:
        question: Question to ask
        include_sources: Include source citations
        agent: Agent (injected)

    Returns:
        Answer with optional sources
    """
    try:
        response = agent.answer(question, include_sources=include_sources)

        result: Dict[str, Any] = {
            "answer": response.answer,
            "confidence": response.confidence
        }

        if include_sources:
            result["sources"] = [
                {
                    "citation": s.citation,
                    "similarity_score": s.similarity_score
                }
                for s in response.sources
            ]

        return {
            "status": "success",
            "content": result
        }

    except Exception as e:
        logger.error(f"Failed to answer question: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.post("/stream")
async def handle_chat(query: str = Form(...)):
    logger.info(f"Chat request received: {query}")
    start = time.time()

    payload = {
        "model": "phi3:mini",
        "messages": [{"role": "user", "content": query}],
        "stream": True
    }

    async def stream_response():
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", OLLAMA_URL, json=payload, timeout=120) as response:
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            yield chunk
        except Exception as e:
            logger.exception("Streaming failed")
            yield f"\n[Error]: {str(e)}"

    logger.info("Streaming response...")
    end = time.time()
    logger.info(f"Ollama stream setup time: {end - start:.2f}s")

    return StreamingResponse(stream_response(), media_type="text/plain")
