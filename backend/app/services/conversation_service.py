"""
Conversation Service - Manages conversation history in the database.
"""
from datetime import datetime

from app.utils.logger import get_logger
import uuid
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from fastapi import Depends, HTTPException

from app.core.database import Conversation, Message, get_db
from app.services.llm_service import LLMService, get_llm_service
from app.models.message import Message as PydanticMessage

logger = get_logger(__name__)

class ConversationService:
    """
    Manages the lifecycle of conversations in a persistent database with robust
    logging and error handling.
    """
    def __init__(self, db_session: Session, llm_service: LLMService):
        self.db = db_session
        self.llm = llm_service
        logger.debug("ConversationService initialized with database session.")

    def get_or_create_conversation(self, conversation_id: Optional[str] = None) -> Conversation:
        """
        Retrieves an existing conversation or creates a new one.
        """
        try:
            if conversation_id:
                stmt = select(Conversation).where(Conversation.id == conversation_id)
                conversation = self.db.execute(stmt).scalar_one_or_none()
                if conversation:
                    logger.debug(f"Found existing conversation: {conversation_id}")
                    return conversation
            
            new_id = conversation_id or str(uuid.uuid4())
            new_conversation = Conversation(id=new_id, title="New Conversation")
            self.db.add(new_conversation)
            self.db.commit()
            self.db.refresh(new_conversation)
            logger.info(f"Created new conversation: {new_id}")
            return new_conversation
        except Exception as e:
            logger.error(f"Database error in get_or_create_conversation: {e}", exc_info=True)
            self.db.rollback()
            raise HTTPException(status_code=500, detail="Database operation failed.")

    def add_message(self, conversation_id: str, sender: str, content: str) -> Message:
        """
        Adds a new message and updates the parent conversation's timestamp.
        """
        try:
            conversation = self.db.get(Conversation, conversation_id)
            if conversation:
                conversation.updated_at = func.now()
            
            db_message = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                sender=sender,
                content=content
            )
            self.db.add(db_message)
            self.db.commit()
            self.db.refresh(db_message)
            logger.debug(f"Added message from '{sender}' to conversation {conversation_id}")
            return db_message
        except Exception as e:
            logger.error(f"Database error in add_message for conversation {conversation_id}: {e}", exc_info=True)
            self.db.rollback()
            raise HTTPException(status_code=500, detail="Failed to save message.")

    def get_history(self, conversation_id: str) -> List[PydanticMessage]:
        """
        Retrieves the message history for a conversation from the database.
        """
        try:
            stmt = (select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.timestamp))
            db_messages = self.db.execute(stmt).scalars().all()
            logger.debug(f"Retrieved {len(db_messages)} messages for conversation {conversation_id}")
            return [PydanticMessage.from_orm(msg) for msg in db_messages]
        except Exception as e:
            logger.error(f"Database error in get_history for conversation {conversation_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve message history.")

    def generate_title_if_needed(self, conversation: Conversation):
        """
        Generates a title for a conversation if it's new, with error handling.
        """
        try:
            self.db.refresh(conversation)
            if conversation.title == "New Conversation" and len(conversation.messages) >= 2:
                logger.info(f"Attempting to generate title for conversation: {conversation.id}")
                
                user_message = conversation.messages[0].content
                assistant_message = conversation.messages[1].content
                
                prompt = (f"Based on this exchange, create a short title (4-6 words):"
                          f"\nUser: \"{user_message}\"\nAssistant: \"{assistant_message}\"\nTitle:")
                
                title_messages = [PydanticMessage(
                    sender="user", content=prompt,
                    conversation_id=conversation.id,
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now()
                )]
                response = self.llm.generate(messages=title_messages, system_prompt="You are a title generator.")
                
                new_title = response.content.strip().strip('"')
                conversation.title = new_title
                self.db.commit()
                logger.info(f"Successfully generated title for conversation {conversation.id}: '{new_title}'")
        except Exception as e:
            # Log the error but don't crash the request. Title generation is non-critical.
            logger.error(f"LLM or DB error during title generation for conversation {conversation.id}: {e}", exc_info=True)
            self.db.rollback()

    def get_all_conversations(self) -> List[Conversation]:
        """Returns all conversations, sorted by the most recently updated."""
        try:
            stmt = select(Conversation).order_by(Conversation.updated_at.desc())
            conversations = self.db.execute(stmt).scalars().all()
            logger.debug(f"Retrieved {len(conversations)} conversations.")
            return list(conversations)
        except Exception as e:
            logger.error(f"Database error in get_all_conversations: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve conversations.")

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Deletes a conversation and all its messages.
        """
        try:
            conversation = self.db.get(Conversation, conversation_id)
            if conversation:
                self.db.delete(conversation)
                self.db.commit()
                logger.info(f"Deleted conversation {conversation_id}")
                return True
            logger.warning(f"Attempted to delete non-existent conversation {conversation_id}")
            return False
        except Exception as e:
            logger.error(f"Database error in delete_conversation for conversation {conversation_id}: {e}", exc_info=True)
            self.db.rollback()
            raise HTTPException(status_code=500, detail="Failed to delete conversation.")

# ==================== FastAPI Dependency ====================

def get_conversation_service(
        db: Session = Depends(get_db),
        llm_service: LLMService = Depends(get_llm_service)
) -> ConversationService:
    """
    FastAPI dependency to get an instance of the ConversationService.
    """
    return ConversationService(db_session=db, llm_service=llm_service)
