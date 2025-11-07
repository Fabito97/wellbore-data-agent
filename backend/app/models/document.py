"""
Pydantic schemas for document processing and API responses.

These models define the structure of data flowing through our system.
Using Pydantic gives us:
- Type safety (catches bugs early)
- Automatic validation (ensures data integrity)
- JSON serialization (easy API responses)
- Documentation (self-documenting code)
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from enum import Enum

class DocumentStatus(str, Enum):
    """
       Status of document processing pipeline.
       Using Enum ensures only valid statuses are used (type safety).
       """
    UPLOADED = "uploaded"  # File saved to disk
    PROCESSING = "processing"  # Currently being processed
    PROCESSED = "processed"  # Successfully processed
    FAILED = "failed"  # Processing failed
    INDEXED = "indexed"  # Embedded and stored in vector DB


class TableExtractionMethod(str, Enum):
    """
    Methods for extracting tables from PDFs.
    Different methods work better for different PDF types:
    - PDFPLUMBER: Good for most PDFs, fast
    - CAMELOT: Better for complex tables, slower
    """
    PDFPLUMBER = "pdfplumber"
    CAMELOT = "camelot"

# ==================== Core Document Models ====================
class TableData(BaseModel):
    """
    Represents a single table extracted from a PDF.

    Why separate model?
    - Tables are structured data (rows/columns)
    - Need different handling than plain text
    - Can be used for direct parameter extraction
    """
    page_number: int = Field(..., description= "PDF page where table was found")
    table_index: int = Field(..., description= "Index of table on the (0-based")
    headers: List[str] = Field(default_factory=list, description= "Column headers")
    rows: List[List[str]] = Field(..., description= "Table data as list of rows")
    bbox: Optional[Dict[str, float]] = Field(default=None, description= "Bounding box {x0, y0, x1, y1}")


    @property
    def row_count(self):
        """Number of data rows (excluding headers)."""
        return len(self.rows)

    @property
    def column_count(self):
        """Number of Columns"""
        return len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)


    def to_markdown(self):
        """
        Convert table to markdown format for better LLM understanding.

        Why markdown?
        - LLMs are trained on markdown tables
        - Human-readable and machine-parseable
        - Preserves structure in text format
        """
        if not self.rows:
            return ""

        lines = []

        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["----"] * len(self.headers)) + " |")

        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(lines)


class PageContent(BaseModel):
    """
    Content extracted from a single page.

    Why per-page?
    - Preserves document structure
    - Enables page-level citations
    - Easier to debug extraction issues
    """

    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Raw text extracted from page")
    tables: List[TableData] = Field(..., description="Tables extracted from page")
    # images: List[bytes] = Field(default_factory=list, description="Raw image bytes from page")
    word_count: int = Field(0, description="Number of words extracted from page")
    char_count: int = Field(0, description="Number of characters extracted from page")
    has_images: bool = Field(False, description="Whether page contains images")

    @field_validator("text")
    @classmethod
    def clean_text(cls, v: str) -> str:
        """
        Clean extracted text.

        Common PDF issues:
        - Multiple spaces
        - Weird line breaks
        - Special characters
        """
        # Remove multiple spaces
        v = " ".join(v.split())
        return v.strip()

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-calculate counts if not provided
        if self.word_count == 0:
            self.word_count = len(self.text.split())
        if self.char_count == 0:
            self.char_count = len(self.text)


class DocumentContent(BaseModel):
    """
    Complete processed document content.

    This is the main output of our document processor.
    Contains everything needed for RAG pipeline.
    """
    # Identification
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_path: Path = Field(..., description="Path to stored PDF file")

    # Content
    pages: List[PageContent] = Field(..., description="Content from each page extracted from the document")
    full_text: str = Field("", description="Full text extracted from all pages in the document")

    # Metadata
    page_count: Optional[int] = Field(..., description="Total number of pages in the document")
    total_word_count: Optional[int] = Field(0, description="Total number of words extracted from the document")
    total_char_count: Optional[int] = Field(0, description="Total number of characters extracted from the document")
    table_count: Optional[int] = Field(0, description="Total number of tables extracted from the document")
    chunk_count: Optional[int] = Field(0, description="Number of chunks created for RAG")

    # Processing info
    status: DocumentStatus = Field(DocumentStatus.PROCESSED, description="Processing status")
    uploaded_at: Optional[datetime] = Field(default_factory=datetime.now, description="When file was uploaded")
    processed_at: Optional[datetime] = Field(None, description="When processing complete")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken to process")

    # Extraction settings used
    extraction_method: TableExtractionMethod = Field(
        TableExtractionMethod.PDFPLUMBER,
        description="Method used for table extraction",
    )


    def __init__(self, **data):
        super().__init__(**data)
        # Auto-calculate aggregates if not provided
        if not self.full_text:
            self.full_text = "\n\n".join(page.text for page in self.pages)
        if self.total_word_count == 0:
            self.total_word_count = sum(page.word_count for page in self.pages)
        if self.total_char_count == 0:
            self.total_char_count = sum(page.char_count for page in self.pages)
        if self.table_count == 0:
            self.table_count = sum(len(page.tables) for page in self.pages)

    @property
    def all_tables(self) -> List[TableData]:
        """Get all tables from all pages."""
        tables: List [TableData] = []
        for page in self.pages:
            tables.extend(page.tables)
        return tables

    @property
    def summary(self):
        """Quick summary of document for logging/debugging."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "pages": self.page_count,
            "words": self.total_word_count,
            "tables": self.table_count,
            "chunks": self.chunk_count,
            "status": self.status,
            "uploaded": self.uploaded_at.isoformat(),
            "processed": self.processed_at.isoformat() if self.processed_at else "pending",
            "processing_time": f"{self.processing_time_seconds:.2f}s" if self.processing_time_seconds else "N/A"
        }


# ==================== API Request/Response Models ====================
class DocumentUploadResponse(BaseModel):
    """
    Response sent to frontend after document upload.

    Separating API models from internal models is good practice:
    - API can change without affecting internal logic
    - Can expose only what frontend needs
    """
    document_id: str
    filename: str
    status: DocumentStatus
    page_count: int
    word_count: int
    table_count: int
    chunk_count: int
    uploaded_at: str  # ISO format timestamp
    message: str = "Document uploaded and processed successfully"


class DocumentProcessingError(BaseModel):
    """
    Error response when processing fails.
    """
    document_id: Optional[str] = None
    filename: str
    error: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ==================== Chunk Models (for RAG) ====================
class DocumentChunk(BaseModel):
    """
    A chunk of text for embedding and retrieval.

    Why chunk?
    - Embedding models have token limits (~512 tokens)
    - Smaller chunks = more precise retrieval
    - But too small = lose context

    Typical chunk size: 500-1000 characters with overlap
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")

    # Position tracking
    page_number: Optional[int] = Field(None, description="Source page number")
    chunk_index: int = Field(..., description="Index within document (0-based)")

    # Metadata for retrieval
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (filename, section, etc.)"
    )
    # Embedding (populated after embedding generation)
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def word_count(self) -> int:
        return len(self.content.split())