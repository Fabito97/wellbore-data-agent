"""
Document Processor - Main Orchestrator

This is the simplified main class that delegates to specialized utilities.
Responsibilities:
1. Determine document type
2. Delegate to appropriate handler
3. Orchestrate the processing flow

Actual extraction logic lives in utils/extractors/
"""
import time
from pathlib import Path
from typing import Optional
import uuid

from app.models.document import DocumentContent, DocumentStatus, DocumentFormat
from app.utils.extractors.pdf_extractor import PDFExtractor
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== Main Document Processor ====================

class DocumentProcessor:
    """
    Main document processor - orchestrates document handling.

    This class:
    1. Validates file
    2. Determines type
    3. Delegates to appropriate extractor
    """

    def __init__(
            self,
            extract_tables: bool = True,
            enable_ocr: bool = False
    ):
        """
        Initialize document processor.

        Args:
            extract_tables: Whether to extract tables from documents
            enable_ocr: Whether to enable OCR for scanned documents
        """
        self.extract_tables = extract_tables
        self.enable_ocr = enable_ocr

        # Initialize extractors (lazy loading for optional dependencies)
        self._pdf_extractor = None

        logger.info(
            f'DocumentProcessor initialized: '
            f'tables={extract_tables}, ocr={enable_ocr}'
        )

    @property
    def pdf_extractor(self) -> PDFExtractor:
        """Lazy load PDF extractor."""
        if self._pdf_extractor is None:
            self._pdf_extractor = PDFExtractor(
                extract_tables=self.extract_tables,
                enable_ocr=self.enable_ocr
            )
        return self._pdf_extractor

    def process_document(
            self,
            file_path: Path,
            well_id: str,
            document_id: Optional[str] = None,
            well_name: Optional[str] = None,
            document_type: Optional[str] = None,
    ) -> Optional[DocumentContent]:
        """
        Process a document file.

        Args:
            file_path: Path to document file
            document_id: Optional custom ID
            well_name: Well this document belongs to
            well_id: Well ID
            document_type: Document type (e.g., "WELL_REPORT")

        Returns:
            DocumentContent or None if file type not supported

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        start_time = time.time()

        # Validation
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate ID if needed
        if document_id is None:
            document_id = str(uuid.uuid4())

        # Determine document type and delegate
        file_format = self._detect_format(file_path)

        if file_format == DocumentFormat.PDF:
            logger.info(f"Processing PDF: {file_path.name}")

            return self._process_pdf(
                file_path=file_path,
                document_id= document_id,
                well_name= well_name,
                well_id= well_id,
                document_type=document_type
            )

        elif file_format == DocumentFormat.DOCX:
            logger.warning(f"DOCX processing not implemented: {file_path.name}")
            return None

        elif file_format == DocumentFormat.IMAGE:
            logger.warning(f"Image processing not implemented: {file_path.name}")
            return None

        else:
            logger.warning(f"Unsupported file type: {file_path.name}")
            return None

    def _detect_format(self, file_path: Path) -> DocumentFormat:
        """
        Detect document format from file extension.

        Args:
            file_path: Path to file

        Returns:
            DocumentFormat enum
        """
        suffix = file_path.suffix.lower()

        if suffix == '.pdf':
            return DocumentFormat.PDF
        elif suffix in ['.docx', '.doc']:
            return DocumentFormat.DOCX
        elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return DocumentFormat.IMAGE
        else:
            return DocumentFormat.OTHER

    def _process_pdf(
            self,
            file_path: Path,
            document_id: str,
            well_name: Optional[str],
            well_id: Optional[str],
            document_type: Optional[str]
    ) -> DocumentContent:
        """
        Process PDF document.

        This method just orchestrates - actual extraction is in PDFExtractor.
        """
        try:
            return self.pdf_extractor.extract(
                file_path=file_path,
                document_id=document_id,
                well_name=well_name,
                well_id=well_id,
                document_type=document_type
            )
        except Exception as e:
            logger.error(f"PDF processing failed: {e}", exc_info=True)
            raise

    def get_page_count(self, file_path: Path) -> int:
        """
        Quick page count without full extraction.

        Args:
            file_path: Path to document

        Returns:
            Number of pages (0 if error or not applicable)
        """
        file_format = self._detect_format(file_path)

        if file_format == DocumentFormat.PDF:
            return self.pdf_extractor.get_page_count(file_path)

        return 0


# ==================== Convenience Functions ====================

def process_pdf(
        file_path: Path,
        well_id: str,
        document_id: Optional[str] = None,
        extract_tables: bool = True
) -> DocumentContent:
    """
    Convenience function for quick PDF processing.

    Args:
        file_path: Path to PDF file
        well_id: Well ID
        document_id: Optional document ID
        extract_tables: Whether to extract tables

    Returns:
        DocumentContent
    """
    processor = DocumentProcessor(extract_tables=extract_tables)
    return processor.process_document(file_path=file_path, well_id=well_id, document_id=document_id)


def get_page_count(file_path: Path) -> int:
    """
    Quick page count.

    Args:
        file_path: Path to document

    Returns:
        Page count
    """
    processor = DocumentProcessor()
    return processor.get_page_count(file_path)