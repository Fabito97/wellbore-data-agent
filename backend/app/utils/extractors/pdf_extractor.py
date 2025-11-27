"""
PDF Extractor - Handles PDF document extraction.

Responsibilities:
1. Extract text from PDFs
2. Coordinate table extraction
3. Coordinate OCR for scanned pages
4. Build PageContent objects

Delegates to:
- text_extractor.py for text extraction
- table_extractor.py for table extraction
- ocr_extractor.py for OCR processing
"""
import time
from pathlib import Path
from typing import List, Optional
import pdfplumber
import fitz  # PyMuPDF

from app.core.config import settings
from app.models.document import (
    DocumentContent, PageContent, DocumentStatus,
    TableExtractionMethod, ImageData
)
from app.utils.extractors.image_extractor import ImageExtractor
from app.utils.extractors.table_extractor import TableExtractor
from app.utils.extractors.ocr_extractor import OCRExtractor
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PDFExtractor:
    """
    Handles PDF document extraction.

    Coordinates specialized extractors for different content types.
    """

    def __init__(
            self,
            extract_tables: bool = True,
            table_method: TableExtractionMethod = TableExtractionMethod.PDFPLUMBER,
            enable_ocr: bool = False
    ):
        """
        Initialize PDF extractor.

        Args:
            extract_tables: Whether to extract tables
            table_method: Which library to use for tables
            enable_ocr: Whether to enable OCR
        """
        self.extract_tables = extract_tables
        self.table_method = table_method
        self.enable_ocr = enable_ocr

        self.table_extractor = None
        if extract_tables:
            self.table_extractor = TableExtractor(method=table_method)

        self.ocr_extractor = None
        if enable_ocr:
            self.ocr_extractor = OCRExtractor()

        logger.info(
            f"PDFExtractor initialized: "
            f"tables={extract_tables}, ocr={enable_ocr}"
        )

    def extract(
            self,
            file_path: Path,
            document_id: str,
            well_name: Optional[str] = None,
            well_id: Optional[str] = None,
            document_type: Optional[str] = None
    ) -> DocumentContent:
        """
        Extract content from PDF.

        Args:
            file_path: Path to PDF
            document_id: Document ID
            well_name: Well name
            well_id: Well ID
            document_type: Document type

        Returns:
            DocumentContent with all extracted data
        """
        start_time = time.time()

        logger.info(f"Extracting PDF: {file_path.name}")

        try:
            # Extract pages
            pages = self._extract_all_pages(file_path, well_id)

            # Build document
            processing_time = time.time() - start_time

            document = DocumentContent(
                document_id=document_id,
                filename=file_path.name,
                file_path=file_path,
                pages=pages,
                page_count=len(pages),
                status=DocumentStatus.PROCESSED,
                processing_time_seconds=processing_time,
                extraction_method=self.table_method,
                well_name=well_name,
                well_id=well_id,
                document_type=document_type,
                ocr_enabled=self.enable_ocr
            )

            logger.info(f"PDF extracted: {document.summary}")

            return document

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}", exc_info=True)
            raise

    def _extract_all_pages(self, file_path: Path, well_id: Optional[str] = None) -> List[PageContent]:
        """
        Extract content from all pages.

        Args:
            file_path: Path to PDF

        Returns:
            List of PageContent objects
        """
        pages = []

        try:
            # Try PyMuPDF first (faster for text)
            doc = fitz.open(file_path)
            total_pages = doc.page_count

            logger.debug(f"Extracting {total_pages} pages")

            for page_num in range(total_pages):
                pymupdf_page = doc[page_num]

                try:
                    page_content = self._extract_page(
                        file_path,
                        page_num + 1,
                        pymupdf_page=pymupdf_page,
                        well_id=well_id
                    )
                    pages.append(page_content)

                except Exception as e:
                    logger.warning(f"PyMuPDF failed for page {page_num + 1}: {e}")

                    # Fallback to pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        pdfplumber_page = pdf.pages[page_num]
                        page_content = self._extract_page(
                            file_path,
                            page_num + 1,
                            pdfplumber_page=pdfplumber_page
                        )
                        pages.append(page_content)

            doc.close()

        except Exception as e:
            # Fallback to pdfplumber for entire document
            logger.warning(f"PyMuPDF failed, using pdfplumber: {e}")

            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        page_content = self._extract_page(
                            file_path,
                            page_num,
                            pdfplumber_page=page
                        )
                        pages.append(page_content)

                    except Exception as page_error:
                        logger.error(f"Failed to extract page {page_num}: {page_error}")

                        # Add error page to maintain page numbering
                        pages.append(PageContent(
                            page_number=page_num,
                            text=f"[Error extracting page {page_num}]",
                            tables=[]
                        ))

        return pages

    def _extract_page(
            self,
            file_path: Path,
            page_number: int,
            pymupdf_page=None,
            pdfplumber_page=None,
            well_id: Optional[str] = None
    ) -> PageContent:
        """
        Extract content from a single page.

        Args:
            file_path: Path to PDF
            page_number: Page number (1-indexed)
            pymupdf_page: PyMuPDF page object (if available)
            pdfplumber_page: pdfplumber page object (if available)

        Returns:
            PageContent object
        """
        # Extract text
        text = self._extract_text(
            pymupdf_page=pymupdf_page,
            pdfplumber_page=pdfplumber_page
        )

        # Check if scanned
        word_count = len(text.split())
        has_text = word_count > 0
        images: List[ImageData] = []

        # Check for images
        has_images = False
        if pymupdf_page:
            has_images = len(pymupdf_page.get_images(full=True)) > 0

            if has_images:
                images.extend(self._extract_images(
                    pymupdf_page,
                    page_number,
                    file_path,
                    well_id=well_id
                ))

        elif pdfplumber_page:
            has_images = len(pdfplumber_page.images) > 0

        is_scanned = not has_text and has_images

        # OCR if needed
        ocr_confidence = None
        if is_scanned and self.ocr_extractor:
            logger.debug(f"Page {page_number} is scanned, applying OCR")

            ocr_result = self.ocr_extractor.extract_text(file_path, page_number)

            if ocr_result and ocr_result.get('confidence', 0) > 0.6:
                text = ocr_result['text']
                ocr_confidence = ocr_result['confidence']
                logger.debug(f"OCR successful: confidence={ocr_confidence:.2f}")
            else:
                logger.debug(f"OCR confidence too low: {ocr_result.get('confidence', 0)}")

        # Extract tables
        tables = []
        if self.table_extractor:
            tables = self.table_extractor.extract_tables(
                file_path=file_path,
                page_number=page_number,
                pdfplumber_page=pdfplumber_page
            )

        return PageContent(
            page_number=page_number,
            text=text,
            tables=tables,
            images=images,
            has_images=has_images,
            is_scanned=is_scanned,
            ocr_confidence=ocr_confidence
        )


    def _extract_images(
            self,
            page: fitz.Page,
            page_number: int,
            file_path: Path,
            well_id: str = None
    ) -> List[ImageData]:
        """
        Extract images from a page.

        Args:
            file_path: Path to PDF
            page_number: Page number
            pymupdf_page: PyMuPDF page object

        Returns:
            List of image metadata dicts
        """
        # Build path:
        # <UPLOAD_DIR>/<well_id>/images/
        folder = well_id or "unknown_well"
        images_dir = settings.UPLOAD_DIR / folder / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        image_list = []
        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(page.parent, xref)
            image_path = images_dir / f"{file_path.stem}_page{page_number}_img{idx}.png"
            pix.save(image_path)

            image_data = ImageData(
                page_number=page_number,
                image_index=idx,
                file_path=str(image_path),
                bbox=None,  # could be filled with img[1:5] if you want coordinates
                detected_by="pymupdf",
                extraction_method="embedded",
                ocr_text=None,
                ocr_confidence=None,
                is_scanned=False,
                is_inline=True
            )
            image_list.append(image_data)
        return image_list


    def _extract_text(
            self,
            pymupdf_page=None,
            pdfplumber_page=None
    ) -> str:
        """
        Extract text from a PDF page.
        Args:
            pymupdf_page: PyMuPDF page object
            pdfplumber_page: pdfplumber page object
        Returns:
            Extracted text string
        """
        text = ""

        try:
            if pymupdf_page:
                text = pymupdf_page.get_text("text") or ""

            elif pdfplumber_page:
                text = pdfplumber_page.extract_text() or ""

            return text.strip()
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""

    def get_page_count(self, file_path: Path) -> int:
        """
        Quick page count without full extraction.

        Args:
            file_path: Path to PDF

        Returns:
            Number of pages
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                return len(pdf.pages)
        except Exception as e:
            logger.error(f"Failed to get page count: {e}")
            return 0