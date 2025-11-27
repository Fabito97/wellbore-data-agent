
# ==================== ocr_extractor.py ====================

"""
OCR Extractor - Handles OCR processing for scanned pages.

Single responsibility: Extract text from images using OCR.
"""
from pathlib import Path
from typing import Optional, Dict, Any
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OCRExtractor:
    """Extracts text from scanned pages using OCR."""

    def __init__(self):
        """Initialize OCR extractor."""
        self.available = False

        try:
            import pytesseract
            from pdf2image import convert_from_path

            self.pytesseract = pytesseract
            self.convert_from_path = convert_from_path
            self.available = True

            logger.info("OCR extractor initialized")

        except ImportError:
            logger.warning(
                "pytesseract or pdf2image not installed. "
                "OCR will not be available."
            )

    def extract_text(
            self,
            file_path: Path,
            page_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Extract text from a page using OCR.

        Args:
            file_path: Path to PDF
            page_number: Page number (1-indexed)

        Returns:
            Dict with 'text' and 'confidence' keys, or None if failed
        """
        if not self.available:
            return None

        try:
            # Convert PDF page to image
            images = self.convert_from_path(
                file_path,
                first_page=page_number,
                last_page=page_number,
                dpi=300
            )

            if not images:
                return None

            image = images[0]

            # Run OCR
            ocr_text = self.pytesseract.image_to_string(image)

            # Get confidence
            ocr_data = self.pytesseract.image_to_data(
                image,
                output_type=self.pytesseract.Output.DICT
            )

            confidences = [
                int(c) for c in ocr_data['conf']
                if c != '-1'
            ]

            avg_confidence = (
                sum(confidences) / len(confidences) / 100.0
                if confidences else 0.0
            )

            logger.debug(f"OCR page {page_number}: confidence={avg_confidence:.2f}")

            return {
                'text': ocr_text.strip(),
                'confidence': avg_confidence
            }

        except Exception as e:
            logger.error(f"OCR failed for page {page_number}: {e}")
            return None
