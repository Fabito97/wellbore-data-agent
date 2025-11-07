from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class Message(BaseModel):
    message_id: str
    session_id: str
    sender: str                  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    source: Optional[str] = None # e.g. 'chat', 'tool', 'api'