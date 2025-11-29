

# ==================== table_extractor.py ====================

"""
Table Extractor - Handles table extraction from PDFs.

Single responsibility: Extract tables from PDF pages.
"""
from pathlib import Path
from typing import List, Optional
import pdfplumber
import pdfplumber.page as pdf_page

from app.models.document import TableData, TableExtractionMethod
from app.utils.extractors.ocr_extractor import OCRExtractor
from app.utils.logger import get_logger

logger = get_logger(__name__)

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

class TableExtractor:
    """Extracts tables from PDF pages."""

    def __init__(self, method: TableExtractionMethod = TableExtractionMethod.CAMELOT):
        """
        Initialize table extractor.

        Args:
            method: Extraction method to use
        """
        self.method = method
        self.camelot_available = False
        self.ocr_extractor = OCRExtractor()  # add OCR support


        if method == TableExtractionMethod.CAMELOT:
            try:
                import camelot
                self.camelot = camelot
                self.camelot_available = True
                logger.info("Camelot table extraction enabled")
            except ImportError:
                logger.warning("Camelot not installed, falling back to pdfplumber")
                self.method = TableExtractionMethod.PDFPLUMBER

    def extract_tables(
            self,
            file_path: Path,
            page_number: int,
            pdfplumber_page=None
    ) -> List[TableData]:
        """
        Extract tables from a PDF page.

        Args:
            file_path: Path to PDF (needed for Camelot)
            page_number: Page number (1-indexed)
            pdfplumber_page: pdfplumber page object (if using pdfplumber)

        Returns:
            List of TableData objects
        """
        tables = []
        if self.method == TableExtractionMethod.CAMELOT and self.camelot_available:
            tables = self._extract_with_camelot(file_path, page_number)

        # ---- FIX: Ensure pdfplumber always has an open PDF ----
        if not tables:
            if pdfplumber_page is None:
                with pdfplumber.open(file_path) as pdf:
                    page = pdf.pages[page_number - 1]
                    tables = self._extract_with_pdfplumber(page_number=page_number, page=page)

            else:
                # If the caller already provided an open page
                tables = self._extract_with_pdfplumber(page_number=page_number, file_path=file_path, page=pdfplumber_page)

            # 3. OCR fallback
        if not tables and self.ocr_extractor.available:
            logger.debug(f"No tables detected on page {page_number}, applying OCR fallback")
            ocr_result = self.ocr_extractor.extract_text(file_path, page_number)
            if ocr_result and ocr_result.get("confidence", 0) > 0.6:
                text = ocr_result["text"]
                # Simple heuristic: split by lines, then by whitespace
                rows = [line.split() for line in text.splitlines() if line.strip()]
                if rows:
                    headers = rows[0]
                    data_rows = rows[1:]
                    tables.append(TableData(
                        page_number=page_number,
                        table_index=1,
                        headers=headers,
                        rows=data_rows,
                        extraction_method=TableExtractionMethod.OCR,
                        confidence=ocr_result["confidence"]
                    ))
                    logger.debug(f"OCR extracted table from page {page_number}: {len(data_rows)}x{len(headers)}")

        return tables


    def _extract_with_pdfplumber(
            self,
            page_number: int,
            file_path: Path = None,
            page: pdf_page.Page = None,
    ) -> List[TableData]:
        """Extract tables using pdfplumber."""
        tables = []

        try:
            raw_tables = page.extract_tables()

            if not raw_tables:
                return tables

            for table_idx, raw_table in enumerate(raw_tables):
                if not raw_table or len(raw_table) < 2:
                    continue

                # Get max columns
                max_cols = max(len(row) for row in raw_table)

                # Extract headers
                headers = [
                    str(cell).strip() if cell else ""
                    for cell in (raw_table[0] + [None] * max_cols)[:max_cols]
                ]

                # Extract rows
                rows = [
                    [str(cell or "").strip() for cell in row]
                    for row in raw_table[1:]
                ]

                table = TableData(
                    page_number=page_number,
                    table_index=table_idx + 1,
                    headers=headers,
                    rows=rows,
                    extraction_method=TableExtractionMethod.PDFPLUMBER
                )

                tables.append(table)

                logger.debug(
                    f"Extracted table {table_idx + 1} from page {page_number}: "
                    f"{table.row_count}x{table.column_count}"
                )

        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {e}")

        return tables


    def _extract_with_camelot(
            self,
            file_path: Path,
            page_number: int
    ) -> List[TableData]:
        """Extract tables using Camelot."""
        tables = []

        try:
            # Try lattice (bordered tables) first
            camelot_tables = self.camelot.read_pdf(
                str(file_path),
                pages=str(page_number),
                flavor='lattice',
                suppress_stdout=True
            )

            # Fallback to stream if no tables found
            if len(camelot_tables) == 0 or all(t.df.shape[1] == 1 for t in camelot_tables):
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

                # Compute bounding box from cell coordinates
                # bbox = getattr(table, "_bbox", None)  # safe access


                table_data = TableData(
                    page_number=page_number,
                    table_index=idx + 1,
                    headers=[str(h).strip() for h in headers],
                    rows=[[str(cell).strip() for cell in row] for row in rows],
                    # bbox=bbox,
                    extraction_method=TableExtractionMethod.CAMELOT,
                    confidence=getattr(table, 'accuracy', None)
                )

                tables.append(table_data)

                logger.debug(
                    f"Camelot extracted table {idx + 1} from page {page_number}: "
                    f"{table_data.row_count}x{table_data.column_count}"
                )

        except Exception as e:
            logger.warning(f"Camelot table extraction failed: {e}")

        return tables