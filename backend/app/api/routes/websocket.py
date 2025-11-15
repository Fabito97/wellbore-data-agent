"""
Chat Routes - WebSocket for real-time Q&A.

Provides interactive chat interface using WebSocket for:
- Real-time question answering
- Streaming responses
- Multi-turn conversations
"""

from app.utils.logger import get_logger
import json
from typing import Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse

from app.agents.simple_agent import SimpleAgent, get_simple_agent
from app.core.config import settings

logger = get_logger(__name__)

router = APIRouter()


# ==================== WebSocket Chat ====================

@router.websocket("/")
async def websocket_chat(
        websocket: WebSocket,
        agent: SimpleAgent = Depends(get_simple_agent)
):
    """
    WebSocket endpoint for real-time chat.

    Args:
        websocket: WebSocket connection
        agent: Simple agent (injected)
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")

    # Send welcome message
    await websocket.send_json({
        "type": "connected",
        "message": "Connected to Wellbore AI Agent",
        "model": settings.OLLAMA_MODEL
    })

    try:
        # Main message loop
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            message_type = data.get("type")

            if message_type == "question":
                await handle_question(websocket, data, agent)

            elif message_type == "summarize":
                await handle_summarize(websocket, data, agent)

            elif message_type == "extract_tables":
                await handle_extract_tables(websocket, data, agent)

            elif message_type == "ping":
                # Heartbeat to keep connection alive
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket.client}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass  # Connection might be closed


async def handle_question(
        websocket: WebSocket,
        data: Dict[str, Any],
        agent: SimpleAgent
):
    """
    Handle question message.

   Flow:
    1. Send "processing" status
    2. Retrieve relevant chunks
    3. Send "generating" status
    4. Generate answer
    5. Send "complete" with answer
    """
    question = data.get("content", "")
    options = data.get("options", {})

    if not question:
        await websocket.send_json({
            "type": "error",
            "message": "Question content is required"
        })
        return

    logger.info(f"Processing question: {question[:100]}...")

    try:
        # Send processing status
        await websocket.send_json({
            "type": "status",
            "message": "Retrieving relevant information..."
        })

        # Get options
        include_sources = options.get("include_sources", True)
        top_k = options.get("top_k", settings.RETRIEVAL_TOP_K)
        filters = options.get("filters")

        # Process question
        response = agent.answer(
            question=question,
            filters=filters,
            include_sources=include_sources
        )

        # Prepare response
        response_data: Dict[str, Any] = {
            "type": "answer",
            "content": response.answer,
            "confidence": response.confidence,
            "tokens_used": response.tokens_used
        }

        # Add sources if requested
        if include_sources and response.sources:
            response_data["sources"] = [
                {
                    "chunk_id": s.chunk_id,
                    "citation": s.citation,
                    "similarity_score": s.similarity_score,
                    "content_preview": s.content[:200] + "..." if len(s.content) > 200 else s.content
                }
                for s in response.sources
            ]

        # Send answer
        await websocket.send_json(response_data)

        logger.info(f"Question answered (confidence: {response.confidence})")

    except Exception as e:
        logger.error(f"Failed to answer question: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to process question: {str(e)}"
        })


async def handle_summarize(
        websocket: WebSocket,
        data: Dict[str, Any],
        agent: SimpleAgent
):
    """
    Handle document summarization request.

    Message format:
    ```json
    {
        "type": "summarize",
        "document_id": "abc-123",
        "max_words": 200
    }
    ```
    """
    document_id = data.get("document_id")
    max_words = data.get("max_words", 200)

    if not document_id:
        await websocket.send_json({
            "type": "error",
            "message": "document_id is required"
        })
        return

    logger.info(f"Summarizing document: {document_id}")

    try:
        await websocket.send_json({
            "type": "status",
            "message": "Generating summary..."
        })

        response = agent.summarize_document(
            document_id=document_id,
            max_words=max_words
        )

        await websocket.send_json({
            "type": "summary",
            "content": response.answer,
            "document_id": document_id,
            "word_count": len(response.answer.split())
        })

        logger.info(f"Summary generated for {document_id}")

    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": f"Summarization failed: {str(e)}"
        })


async def handle_extract_tables(
        websocket: WebSocket,
        data: Dict[str, Any],
        agent: SimpleAgent
):
    """
    Handle table extraction request.

    Message format:
    ```json
    {
        "type": "extract_tables",
        "query": "completion parameters",
        "top_k": 3
    }
    ```
    """
    query = data.get("query", "")
    top_k = data.get("top_k", 5)

    if not query:
        await websocket.send_json({
            "type": "error",
            "message": "query is required"
        })
        return

    logger.info(f"Extracting tables for: {query}")

    try:
        await websocket.send_json({
            "type": "status",
            "message": "Searching for tables..."
        })

        response = agent.extract_tables(query=query, top_k=top_k)

        # Format tables for response
        tables = []
        for source in response.sources:
            tables.append({
                "page_number": source.page_number,
                "content": source.content,
                "similarity_score": source.similarity_score,
                "metadata": source.metadata
            })

        await websocket.send_json({
            "type": "tables",
            "query": query,
            "tables": tables,
            "count": len(tables)
        })

        logger.info(f"Found {len(tables)} tables")

    except Exception as e:
        logger.error(f"Table extraction failed: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": f"Table extraction failed: {str(e)}"
        })
