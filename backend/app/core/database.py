"""
Database setup and session management for the application.
"""

import logging
from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func
from app.core.config import settings

logger = logging.getLogger(__name__)

# ==================== SQLAlchemy Setup ====================
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=settings.DEBUG
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==================== Database Models ====================

class Conversation(Base):
    """
    Represents a single conversation thread.
    """
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=True, default="New Conversation")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    """
    Represents a single message within a conversation.
    """
    __tablename__ = "messages"

    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    sender = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")

# ==================== Database Initialization ====================

def create_db_and_tables():
    """
    Create the database and all defined tables if they don't already exist.
    """
    try:
        logger.info("Initializing database and creating tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database and tables created successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}", exc_info=True)
        raise

# ==================== FastAPI Dependency ====================

def get_db():
    """
    FastAPI dependency to get a database session for a single request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
