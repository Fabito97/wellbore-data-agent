"""
Pydantic schemas for document processing and API responses - defines the doc data structures for our system.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from enum import Enum


class DocumentType(str, Enum):
    """Common wellbore report categories."""
    WELL_REPORT = "report"
    PVT = "pvt"
    PRODUCTION = "production"
    WELL_TEST = "test"
    LOGS = "logs"
    OTHER = "other"

class DocumentFormat(str, Enum):
    """Types of documents - NEW but optional."""
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    CSV = "csv"
    EXCEL = "excel"
    DOCX = "docx"
    OTHER = "other"


class DocumentStatus(str, Enum):
    """
    Status of document processing pipeline - ensures only valid statuses are used (type safety).
    """
    UPLOADED = "uploaded"  # File saved to disk
    PROCESSING = "processing"  # Currently being processed
    PROCESSED = "processed"  # Successfully processed
    FAILED = "failed"  # Processing failed
    INDEXED = "indexed"  # Embedded and stored in vector DB


class TableExtractionMethod(str, Enum):
    """
    Different for extracting tables from PDFs like
    - PDFPLUMBER: Good for most PDFs, fast
    - CAMELOT: Better for complex tables, slower
    """
    PDFPLUMBER = "pdfplumber"
    CAMELOT = "camelot"

# ==================== Core Document Models ====================
class ImageData(BaseModel):
    """
    Represents a single image extracted from a PDF page.
    """

    page_number: int = Field(..., description="PDF page where image was found")
    image_index: int = Field(..., description="Index of image on the page (0-based)")

    # Storage
    file_path: Optional[str] = Field(None, description="Path to stored image file (e.g., PNG/JPEG)")
    bbox: Optional[Dict[str, float]] = Field(
        default=None,
        description="Bounding box {x0, y0, x1, y1} in PDF coordinates"
    )

    # Optional metadata
    detected_by: Optional[str] = Field(None, description="Library/tool used, e.g., pdfplumber, fitz")
    extraction_method: Optional[str] = Field(None, description="Method used, e.g., raster, vector, embedded")
    ocr_text: Optional[str] = Field(None, description="Text extracted via OCR, if available")
    ocr_confidence: Optional[float] = Field(None, description="Confidence score of OCR extraction")

    # Flags
    is_scanned: bool = Field(False, description="Whether image looks like a scanned page")
    is_inline: bool = Field(False, description="Whether image is inline (diagram/figure) vs full-page scan")


class TableData(BaseModel):
    """
    Represents a single table extracted from a PDF - Need different handling than plain text
    """
    page_number: int = Field(..., description= "PDF page where table was found")
    table_index: int = Field(..., description= "Index of table on the (0-based")
    headers: List[str] = Field(default_factory=list, description= "Column headers")
    rows: List[List[str]] = Field(..., description= "Table data as list of rows")
    bbox: Optional[Dict[str, float]] = Field(default=None, description= "Bounding box {x0, y0, x1, y1}")
    raw_text: Optional[str] = Field(default=None, description= "Raw text extracted from PDF")

    # optional fields (won't break existing code)
    extraction_method: Optional[TableExtractionMethod] = None
    confidence: Optional[float] = None

    @property
    def row_count(self):
        """Number of data rows (excluding headers)."""
        return len(self.rows)

    @property
    def column_count(self) -> int:
        """Number of columns."""
        if self.headers:
            return len(self.headers)
        elif self.rows and len(self.rows[0]) > 0:
            return len(self.rows[0])
        return 0


    def to_markdown(self):
        """
        Convert table to markdown format for better LLM understanding - Preserves structure in text format
        """
        if not self.rows:
            return ""

        lines = []

        # Add seperator lines
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["----"] * len(self.headers)) + " |")
        else:
            # still add seperator lines for consistency
            lines.append("| " + " | ".join(["----"] * len(self.rows[0])) + " |")

        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "table_index": self.table_index,
            "headers": self.headers,
            "rows": self.rows,
            "bbox": self.bbox,
            "extraction_method": str(self.extraction_method) if self.extraction_method else None,
            "confidence": self.confidence,
        }


class PageContent(BaseModel):
    """
    Content extracted from a single page - Enables page-level citations and debug.
    Auto-calculates word and character counts without overriding __init__.
    """
    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Raw text extracted from page")
    tables: List['TableData'] = Field(default_factory=list, description="Tables extracted from page")
    has_images: bool = Field(False, description="Whether page contains images")
    images: Optional[List[ImageData]] = Field(default_factory=list, description="Metadata for images on page")

    # optional fields
    is_scanned: bool = False
    ocr_confidence: Optional[float] = None

    @field_validator("text")
    @classmethod
    def clean_text(cls, v: str) -> str:
        """
        Clean extracted text: remove multiple spaces and weird line breaks.
        """
        return " ".join(v.split()).strip()

    @property
    def word_count(self) -> int:
        """Number of words in the cleaned text."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Number of characters in the cleaned text."""
        return len(self.text)

    @property
    def table_count(self) -> int:
        """Number of tables on this page."""
        return len(self.tables)

    @property
    def image_count(self) -> int:
        return len(self.images)


class Well(BaseModel):
    """
    This is the well models for tracking each well documents.
    """
    well_id: str = Field(..., description="Unique ID")
    well_name: str = Field(..., description="Well name")
    document_count: int = Field(0, description="Count of documents")
    uploaded_at: datetime = Field(..., description="Date and time of uploaded") # for tracking the upload date for filtering


class DocumentContent(BaseModel):
    """
    Main output of document processing.

    Stores content and metadata from a processed document,
    suitable for RAG pipelines.
    """

    # Identification
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_path: Path = Field(..., description="Path to stored PDF file")

    # Well/folder metadata
    well_id: Optional[str] = Field(None, description="Unique well identifier")
    well_name: Optional[str] = Field(None, description="Well identifier derived from folder structure")
    document_type: Optional[str] = Field(None, description="Category/type of document, e.g., PVT, production")
    original_folder_path: Optional[str] = Field(None, description="Relative path inside uploaded folder/zip")
    file_format: Optional[str] = Field(None, description="PDF, CSV, TXT, Excel, etc.")

    # Content
    pages: List['PageContent'] = Field(default_factory=list, description="Content from each page")
    # full_text: Optional[str] = Field(None, description="Full text extracted from all pages")

    # Metadata / aggregates
    page_count: Optional[int] = Field(None, description="Total number of pages")
    # total_word_count: Optional[int] = Field(None, description="Total word count")
    # total_char_count: Optional[int] = Field(None, description="Total character count")
    # table_count: Optional[int] = Field(None, description="Total number of tables")
    chunk_count: int = Field(0, description="Number of chunks created for RAG")

    # Processing info
    status: 'DocumentStatus' = Field(..., description="Processing status")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = Field(None)
    processing_time_seconds: Optional[float] = Field(None)
    extraction_method: 'TableExtractionMethod' = Field(..., description="Method used for table extraction")

    # Optional flags
    ocr_enabled: bool = False

    # --- Computed properties ---
    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def total_word_count(self) -> int:
        return sum(page.word_count for page in self.pages)

    @property
    def total_char_count(self) -> int:
        return sum(page.char_count for page in self.pages)

    @property
    def table_count(self) -> int:
        return sum(page.table_count for page in self.pages)

    @property
    def image_count(self) -> int:
        return sum(page.image_count for page in self.pages)

    @property
    def all_tables(self) -> List[TableData]:
        """Get all tables from all pages."""
        tables: List [TableData] = []
        for page in self.pages:
            tables.extend(page.tables)
        return tables

    @property
    def all_images(self) -> List[ImageData]:
        images = []
        for page in self.pages:
            images.extend(page.images)
        return images

    @property
    def image_pages(self) -> List[int]:
        return [page.page_number for page in self.pages if page.has_images]

    @property
    def summary(self):
        """Quick summary of document for logging/debugging."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "well_id": self.well_id,
            "pages": self.page_count,
            "words": self.total_word_count,
            "tables": self.table_count,
            "chunks": self.chunk_count,
            "image_pages": self.image_pages,
            "status": self.status,
            "well_name": self.well_name,
            "document_type": self.document_type,
            "uploaded": self.uploaded_at.isoformat(),
            "processed": self.processed_at.isoformat() if self.processed_at else "pending",
            "processing_time": f"{self.processing_time_seconds:.2f}s" if self.processing_time_seconds else "N/A"
        }

    class Config:
        arbitrary_types_allowed = True

# ==================== Chunk Models (for RAG) ====================
class DocumentChunk(BaseModel):
    """
    A chunk of text for embedding and retrieval.
    Typical chunk size: 500-1000 characters with overlap
    """
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")

    # Well/folder metadata
    well_id: str = Field(..., description="Well Id")
    well_name: str = Field(..., description="Well name")
    document_type: str = Field(..., description="Category/type of document, e.g., PVT, production")

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


# ==================== API Request/Response Models ====================
class DocumentUploadResponse(BaseModel):
    """
    API response sent to frontend after document upload.
    """
    document_id: str
    filename: str
    well_id: str
    well_name: str
    document_type: str
    format: str
    status: DocumentStatus
    page_count: int
    word_count: int
    table_count: int
    chunk_count: int
    uploaded_at: str  # ISO format timestamp
    elapsed_time: float  # Time taken to process in seconds
    message: str = "Document uploaded and processed successfully"



# ==================== NEW: Batch Upload Response (Optional to use) ====================

class BatchUploadResponse(BaseModel):
    """NEW: For batch processing. Only use if you need it."""
    total_documents: int
    successful: int
    failed: int
    documents: List[DocumentUploadResponse]
    errors: List[Dict[str, str]] = []
    total_time: float
    message: str


class DocumentProcessingError(BaseModel):
    """
    Error response when processing fails.
    """
    document_id: Optional[str] = None
    filename: str
    error: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
