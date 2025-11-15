"""
Document Processor - Extracts content from PDF files.

This module handles the complex task of extracting structured data from PDFs:
- Text extraction with layout preservation
- Table detection and extraction
- Page-by-page processing for better granularity

Notes:
- Some PDFs are actually scanned images (need OCR - skipped for CPU constraints)
- PDFs store text as positioned glyphs, not flowing text
- Table extraction uses visual/whitespace analysis
"""

import time
import pdfplumber
from pdfplumber.page import Page as PDFPage
from app.utils.logger import get_logger
from app.core.config import settings
from app.models.document import DocumentChunk, PageContent, TableData, DocumentContent, TableExtractionMethod, \
    DocumentStatus
from pathlib import Path
from typing import List, Optional
import fitz
import uuid

# Setup logging
logger = get_logger(__name__)

class DocumentProcessor:
    """
    Processes PDF documents to extract text and tables.

    Args:
        extract_tables: Whether to extract tables (slower but more accurate)
        table_method: Which library to use for table extraction
    """
    def __init__(
            self,
            extract_tables: bool = True,
            table_method: TableExtractionMethod =
            TableExtractionMethod.PDFPLUMBER
    ):
        # Initialize document processor
        self.extract_tables = extract_tables
        self.table_method = table_method

        logger.info(
            f"DocumentProcessor initialized: "
            f"extract_tables={extract_tables}, method={table_method.value}"
        )


    def process_document(
            self,
            file_path: Path,
            document_id: Optional[str] = None
    ) -> DocumentContent:
        """
        Main entry point: Call to process a PDF file completely

        Args:
            file_path: Path to PDF file
            document_id: Optional custom ID (generates UUID if not provided

        Returns:
            DocumentContent with all extracted data

        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If file is not a PDF
            Exception: For PDF processing errors
        """
        start_time = time.time()

        # Validation
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF, got: {file_path.suffix}")

        # Generate ID if not provided
        if document_id is None:
            document_id = str(uuid.uuid4())

        logger.info(f"Processing document: {file_path.name} (ID: {document_id})")

        try:
            # Extract content page by page
            pages = self._extract_pages(file_path)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Build DocumentContent object
            document = DocumentContent(
                document_id=document_id,
                filename=file_path.name,
                file_path=file_path,
                pages=pages,
                page_count=len(pages),
                status=DocumentStatus.PROCESSED,
                processing_time_seconds=processing_time,
                extraction_method=self.table_method
            )

            logger.info(f"Document processed successfully: {document.summary}")

            return document

        except Exception as e:
            logger.error(f"Failed to process document {file_path.name}: {e}", exc_info=True)
            raise


    def _extract_pages(self, file_path: Path) -> List[PageContent]:
        """
        Extract content from all pages - Not meant to be called externally
        """
        pages = []

        # Open PDF with pdfplumber (easier API for tables)
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            logger.debug(f"Extracting {total_pages} pages from {file_path.name}")

            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_content = self._extract_page_content(page, page_num)
                    pages.append(page_content)

                except Exception as e:
                    # Don't fail entire document for one bad page
                    logger.warning(f"Failed to extract page {page_num}: {e}")
                    # Add empty page content to maintain page numbering
                    pages.append(PageContent(
                        page_number=page_num,
                        text=f"[Error extracting page {page_num}]",
                        tables=[]
                    ))

        return pages


    def _extract_page_content(self, page: PDFPage, page_number: int) -> PageContent:
        """
        Extract text and tables from a single page

        Args:
            page: pdfplumber Page object
            page_number: 1-indexed page numbe

        Returns:
            PageContent with text and table
        """
        # Extract text
        text = page.extract_text() or ""

        # Extract tables if enabled
        tables = []
        if self.extract_tables:
            tables = self._extract_tables_from_page(page, page_number)

        # Check for images (simplified - just checks if page has images)
        has_images = len(page.images) > 0 if hasattr(page, 'images') else False

        return PageContent(
            page_number=page_number,
            text=text,
            tables=tables,
            has_images=has_images
        )


    def _extract_tables_from_page(
            self,
            page: pdfplumber.page.Page,
            page_number: int
    ) -> List[TableData]:
        """
        Extract all tables from a page.

        Teaching: Table Extraction Algorithm - Use pdfplumber library defaults (they're pretty good)
        """
        tables = []

        try:
            # pdfplumber.extract_tables() returns list of 2D arrays
            # Each table is: [[cel, cell, cell], [cell, cell, cell], ...]
            raw_tables = page.extract_tables()

            if not raw_tables:
                return tables

            for table_idx, raw_table in enumerate(raw_tables):
                if not raw_table or len(raw_table) < 2:
                    # Skip empty of single_row tables
                    continue

                headers = [str(cell or "").strip() for cell in raw_table[0]]

                # Remaining rows are data
                rows = []
                for row in raw_table[1:]:
                    # Clean cells: convert to string, strip whitespace
                    cleaned_row = [str(cell or "").strip() for cell in row]
                    rows.append(cleaned_row)

                # Create TableData object
                table = TableData(
                    page_number=page_number,
                    table_index=table_idx,
                    headers=headers,
                    rows=rows,
                    bbox=None # pdfplumber doesn't provide bbox easily
                )

                tables.append(table)

                logger.debug(
                    f"Extracted table {table_idx} from page {page_number}: {table.headers} "
                    f"{table.row_count} rows * {table.column_count} columns"
                )

        except Exception as e:
            logger.warning(f"Failed to extract tables from page {page_number}: {e}")

        return tables


    def extract_text_only(self, file_path: Path) -> str:
        """
        Quick text extraction without tables or structure.
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)

                return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise


    def get_page_count(self, file_path: Path) -> int:
        """
        Quick page count without full extraction - Useful for progress bars, validation
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                return len(pdf.pages)
        except Exception as e:
            logger.error(f"Failed to get page count form {file_path}: {e}")
            return 0


# ==================== Module-level convenience functions ====================
def process_pdf(
        file_path: Path,
        document_id: Optional[str] = None,
        extract_tables: bool = True
) -> DocumentContent:
    """
    Convenience function for quick PDF processing.
    """
    processor = DocumentProcessor(extract_tables=extract_tables)
    return processor.process_document(file_path, document_id)