
# ==================== image_extractor.py (Placeholder) ====================

"""
Image Extractor - Handles image extraction from PDFs.

Single responsibility: Extract and save embedded images.

NOT IMPLEMENTED YET - Placeholder for future use.
"""
from pathlib import Path
from typing import List, Dict, Optional

import fitz

from app.core.config import settings
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
        # Build path: e.g. UPLOAD_DIR/<well_id>/images/

        folder = well_id or "unknown_well"
        images_dir = settings.UPLOAD_DIR / folder / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        image_list = []
        seen_hashes = set()

        pix = None
        try:
            for idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                image_path = images_dir / f"{file_path.stem}_page{page_number}_img{idx}.png"
                pix.save(image_path)

                # ---- Filtering heuristics ---
                x0, y0, x1, y1 = img[1:5]
                width, height = x1 - x0, y1 - y0

                if width < 50 or height < 50:  # skip tiny images
                    continue

                if image_path.stat().st_size < 10_000:  # skip very small files (10kb)
                    continue

                # duplicate check
                import hashlib
                with open(image_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in seen_hashes:
                    continue

                seen_hashes.add(file_hash)

                # Keep relevant images
                image_data = ImageData(
                    page_number=page_number,
                    image_index=idx,
                    file_path=str(image_path),
                    bbox=img[1:5],  # could be filled with img[1:5] if you want coordinates
                    detected_by="pymupdf",
                    extraction_method="embedded",
                    ocr_text=None,
                    ocr_confidence=None,
                    is_scanned=False,
                    is_inline=True
                )
                image_list.append(image_data)

                logger.debug(f"Extracted image {idx+1} from page {page_number}")

            return image_list

        except Exception as e:
            logger.warning(f"Failed to extract images from page {page_number}: {e}")
            return []

        finally:
            if pix is not None:
                pix = None  # release Pixmap explicitly
