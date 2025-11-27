"""
Specialized Extractors - Each handles one thing well.

Structure:
- text_extractor.py: Text extraction
- table_extractor.py: Table extraction
- ocr_extractor.py: OCR processing
- image_extractor.py: Image extraction (future)
"""

# ==================== text_extractor.py ====================

"""
Text Extractor - Handles text extraction from PDFs.

Single responsibility: Extract text from PDF pages.
"""
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)




# ==================== __init__.py for utils/extractors/ ====================

"""
Extractors package - Specialized content extractors.

Available extractors:
- TextExtractor: Extract text
- TableExtractor: Extract tables  
- OCRExtractor: OCR for scanned pages
- ImageExtractor: Extract images (placeholder)
"""

__all__ = [
    'TextExtractor',
    'TableExtractor',
    'OCRExtractor',
    'ImageExtractor'
]