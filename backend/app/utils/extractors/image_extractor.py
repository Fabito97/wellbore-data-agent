
# ==================== image_extractor.py (Placeholder) ====================

"""
Image Extractor - Handles image extraction from PDFs.

Single responsibility: Extract and save embedded images.

NOT IMPLEMENTED YET - Placeholder for future use.
"""
from pathlib import Path
from typing import List, Dict

import fitz

from app.models.document import ImageData
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ImageExtractor:
    """Extracts images from PDF pages (future implementation)."""

    def __init__(self):
        """Initialize image extractor."""
        logger.info("ImageExtractor initialized (not implemented)")

    def extract_images(
            self, page: fitz.Page,
            page_number: int,
            file_path: Path
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
        image_list = []
        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(page.parent, xref)
            image_path = file_path.parent / f"{file_path.stem}_page{page_number}_img{idx}.png"
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
