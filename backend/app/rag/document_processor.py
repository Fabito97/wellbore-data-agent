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
    DocumentStatus, DocumentFormat
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
            TableExtractionMethod.PDFPLUMBER,
            enable_ocr: bool = False
    ):
        # Initialize document processor
        self.extract_tables = extract_tables
        self.table_method = table_method
        self.enable_ocr = enable_ocr

        # Only import OCR libs if needed (optional dependencies)
        self.ocr_available = False
        self.camelot_available = False

        logger.info(
            f"DocumentProcessor initialized: "
            f"extract_tables={extract_tables}, method={table_method.value}"
        )

        if enable_ocr:
            try:
                import pytesseract
                from pdf2image import convert_from_path
                self.pytesseract = pytesseract
                self.convert_from_path = convert_from_path
                self.ocr_available = True
                logger.info("OCR enabled")
            except ImportError:
                logger.warning("OCR requested but pytesseract/pdf2image not installed")

        if table_method == TableExtractionMethod.CAMELOT:
            try:
                import camelot
                self.camelot = camelot
                self.camelot_available = True
                logger.info("Camelot table extraction enabled")
            except ImportError:
                logger.warning("Camelot requested but not installed, falling back to pdfplumber")
                self.table_method = TableExtractionMethod.PDFPLUMBER

        logger.info(
            f"DocumentProcessor initialized: "
            f"tables={extract_tables}, method={table_method.value}, ocr={enable_ocr}"
        )

    def process_document(
            self,
            file_path: Path,
            document_id: Optional[str] = None,
            well_name: Optional[str] = None,
            well_id: Optional[str] = None,
            document_type: Optional[str] = None
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
                extraction_method=self.table_method,
                file_format=DocumentFormat.PDF,
                well_name=well_name,
                well_id=well_id,
                document_type=document_type,
                ocr_enabled=self.enable_ocr
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


    def _extract_page_content(self, page: PDFPage, page_number: int, file_path: Optional[Path] = None) -> PageContent:
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

        # Check if page might be scanned (optional)
        word_count = len(text.split())
        is_scanned = word_count < 30  # Heuristic
        ocr_confidence = None

        # OCR fallback if enabled and page looks scanned
        if is_scanned and self.enable_ocr and self.ocr_available and file_path:
            logger.debug(f"Page {page_number} appears scanned, applying OCR")
            try:
                ocr_text, ocr_confidence = self._ocr_page(file_path, page_number)
                if ocr_confidence and ocr_confidence > 0.6:
                    text = ocr_text
            except Exception as e:
                logger.warning(f"OCR failed for page {page_number}: {e}")

        # Extract tables if enabled
        tables = []
        if self.extract_tables:
            if self.table_method == TableExtractionMethod.CAMELOT and self.camelot_available and file_path:
                tables = self._extract_tables_camelot(file_path, page_number)
            else:
                tables = self._extract_tables_from_page(page, page_number)

        # Check for images (simplified - just checks if page has images)
        has_images = len(page.images) > 0 if hasattr(page, 'images') else False

        return PageContent(
            page_number=page_number,
            text=text,
            tables=tables,
            has_images=has_images,
            is_scanned=is_scanned,  # NEW
            ocr_confidence=ocr_confidence  # NEW
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
                # Determine the maximum number of columns in the table
                max_cols = max(len(row) for row in raw_table)

                # Generate headers
                headers = [
                    (str(cell).strip() if cell else "") for i, cell in
                    enumerate(raw_table[0] + [None] * max_cols)
                ][:max_cols]  # pad with None to match max_cols

                # Remaining rows are data
                rows = [
                    [str(cell or "").strip() for cell in row] # Clean cells: convert to string, strip whitespace
                    for row in raw_table[1:]
                ]

                # Create TableData object
                table = TableData(
                    page_number=page_number,
                    table_index=table_idx + 1,
                    headers=headers,
                    rows=rows,
                    bbox=None, # pdfplumber doesn't provide bbox easily
                    extraction_method = TableExtractionMethod.PDFPLUMBER
                )

                tables.append(table)
                logger.debug(
                    f"Extracted table {table_idx + 1} from page {page_number}: {table.headers} "
                    f"{table.row_count} rows * {table.column_count} columns"
                )

        except Exception as e:
            logger.warning(f"Failed to extract tables from page {page_number}: {e}")

        return tables

    def _extract_tables_camelot(
            self,
            file_path: Path,
            page_number: int
    ) -> List[TableData]:
        """NEW: Camelot extraction (better for complex tables)."""
        tables = []

        try:
            # Try lattice (bordered tables) first
            camelot_tables = self.camelot.read_pdf(
                str(file_path),
                pages=str(page_number),
                flavor='lattice',
                suppress_stdout=True
            )

            # Fallback to stream (borderless) if no tables found
            if len(camelot_tables) == 0:
                camelot_tables = self.camelot.read_pdf(
                    str(file_path),
                    pages=str(page_number),
                    flavor='stream',
                    suppress_stdout=True
                )

            for idx, table in enumerate(camelot_tables):
                df = table.df
                headers = df.iloc[0].tolist()
                rows = df.iloc[1:].values.tolist()

                table_data = TableData(
                    page_number=page_number,
                    table_index=idx,
                    headers=[str(h).strip() for h in headers],
                    rows=[[str(cell).strip() for cell in row] for row in rows],
                    extraction_method=TableExtractionMethod.CAMELOT,
                    confidence=table.accuracy if hasattr(table, 'accuracy') else None
                )

                tables.append(table_data)
                logger.debug(
                    f"Camelot extracted table {idx} from page {page_number}: "
                    f"{table_data.row_count}x{table_data.column_count}"
                )

        except Exception as e:
            logger.warning(f"Camelot extraction failed page {page_number}: {e}")

        return tables

    def _ocr_page(self, file_path: Path, page_num: int) -> tuple[str, float]:
        """NEW: OCR a single page."""
        try:
            # Convert PDF page to image
            images = self.convert_from_path(
                file_path,
                first_page=page_num,
                last_page=page_num,
                dpi=300
            )

            if not images:
                return "", 0.0

            image = images[0]

            # OCR with confidence
            ocr_data = self.pytesseract.image_to_data(
                image,
                output_type=self.pytesseract.Output.DICT
            )

            text = self.pytesseract.image_to_string(image)

            # Calculate average confidence
            confidences = [int(c) for c in ocr_data['conf'] if c != '-1']
            avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0

            logger.debug(f"OCR page {page_num}: confidence={avg_confidence:.2f}")
            return text, avg_confidence

        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
            return "", 0.0

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