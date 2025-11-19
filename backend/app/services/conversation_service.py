
"""
Conversation Service - Manages conversation history.

This service is responsible for storing and retrieving the history of messages
for each conversation. It acts as the single source of truth for what has
been said in a conversation.

This implementation uses a simple in-memory dictionary, which is suitable for
development and testing. For production, this should be replaced with a
persistent storage backend like Redis, PostgreSQL, or MongoDB.
"""

import logging
from typing import List, Dict, Optional
from collections import defaultdict
import uuid

from app.models.message import Message
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationService:
    """
    Manages the lifecycle of conversations.
    """
    def __init__(self):
        # In-memory storage for conversations.
        # The key is the conversation_id (str), and the value is a list of Messages.
        # A defaultdict simplifies the logic for creating new conversations.
        self._histories: Dict[str, List[Message]] = defaultdict(list)
        logger.info("ConversationService initialized with in-memory storage.")

    def get_history(self, conversation_id: str) -> List[Message]:
        """
        Retrieve the message history for a given conversation.

        Args:
            conversation_id: The unique identifier for the conversation.

        Returns:
            A list of Message objects, or an empty list if the conversation is new.
        """
        logger.debug(f"Retrieving history for conversation_id: {conversation_id}")
        return self._histories[conversation_id]

    def add_message(self, conversation_id: str, sender: str, content: str) -> Message:
        """
        Add a new message to a conversation's history.

        Args:
            conversation_id: The ID of the conversation.
            sender: The role of the message sender (e.g., 'user', 'assistant').
            content: The text content of the message.

        Returns:
            The newly created Message object.
        """
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty.")

        new_message = Message(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            sender=sender,
            content=content,
            timestamp=datetime.now()
        )
        
        self._histories[conversation_id].append(new_message)
        logger.debug(f"Added message from '{sender}' to conversation_id: {conversation_id}")
        return new_message

    def clear_history(self, conversation_id: str):
        """
        Clear the history of a specific conversation.

        Args:
            conversation_id: The ID of the conversation to clear.
        """
        if conversation_id in self._histories:
            del self._histories[conversation_id]
            logger.info(f"Cleared history for conversation_id: {conversation_id}")

# ==================== Module-level instance ====================

# Create a single, module-level instance of the service.
# This singleton pattern ensures that all parts of the application
# share the same conversation history data (in this in-memory case).
_conversation_service_instance: Optional[ConversationService] = None

def get_conversation_service() -> ConversationService:
    """
    Get the singleton instance of the ConversationService.
    """
    global _conversation_service_instance
    if _conversation_service_instance is None:
        _conversation_service_instance = ConversationService()
    return _conversation_service_instance
