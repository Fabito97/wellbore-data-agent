"""
Database setup and session management for the application.

This module initializes the database connection using SQLAlchemy and defines
the table structures for conversations and messages. It provides a dependency
for getting a database session in FastAPI routes.
"""

import logging
from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func
from app.core.config import settings

logger = logging.getLogger(__name__)

# ==================== SQLAlchemy Setup ====================

# The engine is the entry point to the database.
# `connect_args` is specific to SQLite to allow multi-threaded access.
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=settings.DEBUG  # Log SQL queries in debug mode
)

# The SessionLocal class is a factory for creating new Session objects.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our declarative models.
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
    
    # This relationship links a conversation to its messages.
    # `back_populates` creates a two-way link.
    # `cascade` ensures that when a conversation is deleted, its messages are also deleted.
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    """
    Represents a single message within a conversation.
    """
    __tablename__ = "messages"

    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    sender = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # This relationship links a message back to its parent conversation.
    conversation = relationship("Conversation", back_populates="messages")

# ==================== Database Initialization ====================

def create_db_and_tables():
    """
    Create the database and all defined tables if they don't already exist.
    This function should be called once on application startup.
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
    
    This pattern ensures that the database session is created at the beginning
    of a request and closed at the end, even if an error occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
