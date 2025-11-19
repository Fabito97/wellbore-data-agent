"""
Chat API - Endpoints for all chat and conversation management.
"""

import logging
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uuid

from app.agents.simple_agent import get_simple_agent, SimpleAgent, AgentResponse as AgentResponseModel
from app.services.conversation_service import get_conversation_service, ConversationService
from app.models.message import Message as MessageModel, Message

logger = logging.getLogger(__name__)
router = APIRouter()

# ==================== Pydantic Models for API ====================

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's question or message.")
    conversation_id: Optional[str] = Field(None, description="The ID of the conversation for context.")

class SimpleChatResponse(BaseModel):
    status: str
    message: str
    data: Message

class ConversationInfo(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class MessageInfo(BaseModel):
    id: str
    sender: str
    content: str
    timestamp: datetime
    conversation_id: str

    class Config:
        orm_mode = True

# ==================== Core Chat Endpoints ====================

@router.post("/agent", response_model=AgentResponseModel)
async def chat_with_agent(
    request: ChatRequest,
    agent: SimpleAgent = Depends(get_simple_agent)
):
    """
    Handles a conversation turn with the RAG agent using a persistent database.
    """
    logger.info(f"Agent request for conversation_id: {request.conversation_id}")
    try:
        response = agent.answer(
            question=request.question,
            conversation_id=request.conversation_id
        )
        return response
    except Exception as e:
        logger.error(f"Error in agent chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask", response_model=SimpleChatResponse)
async def simple_chat(
    request: ChatRequest,
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """

    A simple chat endpoint that talks directly to the LLM, bypassing RAG,
    and persists the conversation.
    """
    logger.info(f"Simple chat request for conversation_id: {request.conversation_id}")
    
    try:
        conversation = conversation_service.get_or_create_conversation(request.conversation_id)
        conversation_service.add_message(conversation.id, "user", request.query)
        
        history = conversation_service.get_history(conversation.id)
        
        llm_response = conversation_service.llm.generate(messages=history)
        
        llm_message = conversation_service.add_message(conversation.id, "assistant", llm_response.content)
        
        conversation_service.generate_title_if_needed(conversation)
        
        return SimpleChatResponse(
            status="success",
            message="Response generated successfully",
            data=llm_message
        )
    except Exception as e:
        logger.error(f"Error in simple chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """
    Streams a response from the LLM token by token and persists the full conversation.
    """
    logger.info(f"Streaming chat request for conversation_id: {request.conversation_id}")
    
    try:
        conversation = conversation_service.get_or_create_conversation(request.conversation_id)
        conversation_service.add_message(conversation.id, "user", request.question)
        
        history = conversation_service.get_history(conversation.id)

        async def stream_generator():
            full_response = ""
            try:
                yield f'{{"conversation_id": "{conversation.id}"}}\n'

                for chunk in conversation_service.llm.stream_generate(messages=history):
                    yield chunk
                    full_response += chunk
                
                conversation_service.add_message(conversation.id, "assistant", full_response)
                conversation_service.generate_title_if_needed(conversation)
                logger.info(f"Streaming complete for conversation: {conversation.id}")

            except Exception as e:
                logger.error(f"Error during streaming: {e}", exc_info=True)
                yield f'{{"error": "{str(e)}"}}'

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
    except Exception as e:
        logger.error(f"Error setting up stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Conversation Management Endpoints ====================

@router.get("", response_model=List[ConversationInfo])
async def get_conversations(
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """
    Retrieve a list of all conversations, sorted by the most recently updated.
    """
    try:
        return conversation_service.get_all_conversations()
    except Exception as e:
        logger.error(f"Error getting conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve conversations.")

@router.get("/{conversation_id}/messages", response_model=List[MessageInfo])
async def get_messages(
    conversation_id: str,
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """

    Retrieve all messages for a specific conversation.
    """
    try:
        # get_history returns Pydantic models which are compatible with MessageInfo
        return conversation_service.get_history(conversation_id)
    except Exception as e:
        logger.error(f"Error getting messages for conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve messages.")

@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    """
    Delete a conversation and all of its associated messages.
    """
    try:
        success = conversation_service.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}", exc_info=True)
        # Re-raise if it's an HTTPException, otherwise wrap it
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="Could not delete conversation.")
