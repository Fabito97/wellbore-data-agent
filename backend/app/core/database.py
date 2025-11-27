"""
Enhanced Database with Documents and Wells persistence.

New tables:
- wells: Store well metadata
- documents: Store document metadata (chunks in vector DB)
"""
import logging
from sqlalchemy import (
    create_engine, Column, String, DateTime, ForeignKey,
    Text, Integer, Float, Boolean, JSON
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func
from app.core.config import settings

logger = logging.getLogger(__name__)

# ==================== SQLAlchemy Setup ====================
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
    # echo=settings.DEBUG
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ==================== Database Models ====================

class Well(Base):
    """
    Represents a well with associated documents.
    """
    __tablename__ = "wells"

    id = Column(String, primary_key=True, index=True)  # well-{uuid}
    name = Column(String, unique=True, index=True, nullable=False)  # well-4

    # Optional metadata
    field_name = Column(String, nullable=True)
    operator = Column(String, nullable=True)
    location = Column(JSON, nullable=True)  # {lat, lon, etc.}

    # Counts
    document_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    documents = relationship("Document", back_populates="well", cascade="all, delete-orphan")


class Document(Base):
    """
    Represents a document (PDF) with metadata.
    Chunks are stored in vector DB, metadata here.
    """
    __tablename__ = "documents"

    # Identity
    id = Column(String, primary_key=True, index=True)  # UUID
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False) # stored file_path

    # Well association
    well_id = Column(String, ForeignKey("wells.id"), nullable=True, index=True)
    well_name = Column(String, index=True, nullable=True)

    # Document metadata
    document_type = Column(String, nullable=True)  # WELL_REPORT, PVT, etc.
    file_format = Column(String, default="pdf")
    original_folder_path = Column(String, nullable=True)

    # Processing status
    status = Column(String, default="pending")  # pending, processed, indexed, failed

    # Counts
    page_count = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    table_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)

    # Processing details
    processing_time_seconds = Column(Float, nullable=True)
    extraction_method = Column(String, default="pdfplumber")
    ocr_enabled = Column(Boolean, default=False)

    # Timestamps
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    well = relationship("Well", back_populates="documents")


class Conversation(Base):
    """Represents a single conversation thread."""
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=True, default="New Conversation")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """Represents a single message within a conversation."""
    __tablename__ = "messages"

    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    sender = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")


# ==================== Database Initialization ====================

def create_db_and_tables():
    """Create all database tables."""
    try:
        logger.info("Initializing database and creating tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created: wells, documents, conversations, messages")
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}", exc_info=True)
        raise

def reset_database():
    """
    Drop all tables and recreate them using SQLAlchemy metadata.
    """
    try:
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        logging.info("Database tables dropped")

        # Recreate all tables
        Base.metadata.create_all(bind=engine)
        logging.info("Database tables recreated")
    except Exception as e:
        logging.error(f"Failed to reset database: {e}")
        raise



# ==================== FastAPI Dependency ====================

def get_db():
    """FastAPI dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()