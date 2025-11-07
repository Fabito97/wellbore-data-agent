import time
import httpx
from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse
from ollama import Client
from app.utils.logger import get_logger
# client = Client()

router = APIRouter(prefix="/chat", tags=["Agent"])
logger = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"

@router.post("/")
async def handle_chat(query: str = Form(...)):
    logger.info(f"Chat request received: {query}")
    try:
        start = time.time()

        payload = {
            "model": "phi3:mini",
            "messages": [{"role": "user", "content": query}],
            "stream": False  # Set to True if you want to stream
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(OLLAMA_URL, json=payload, timeout=120)

        if response.status_code != 200:
            logger.error(f"Ollama error: {response.text}")
            return {
                "status": "error",
                "message": "Failed to generate response from model",
                "detail": response.text
            }

        result = response.json()["message"]["content"]
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
