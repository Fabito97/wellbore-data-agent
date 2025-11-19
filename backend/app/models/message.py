"""
Pydantic model for a Message.
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class Message(BaseModel):
    """
    Represents a single message in a conversation.
    """
    id: str
    conversation_id: str
    sender: str
    content: str
    timestamp: datetime

    class Config:
        """
        Pydantic configuration.
        
        from_attributes (formerly orm_mode) allows the model to be created
        from arbitrary class instances (like SQLAlchemy models) by reading
        their attributes, not just from dictionaries.
        """
        from_attributes = True
