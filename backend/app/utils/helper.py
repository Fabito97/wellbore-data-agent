import re
from pathlib import Path
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

def normalize_score(raw_score: float) -> float:
    # Chroma returns distance → convert to similarity
    if raw_score >= 0 and raw_score <= 2:
        return 1 - raw_score   # reasonable heuristic for cosine distance

    # FAISS cosine similarity already in [-1, 1]
    return raw_score


def normalize_well_name(raw_name: str) -> str:
    """
    Normalize well names for consistency.
    'well_1' → 'well-1'
    'WELL 001' → 'well-1'
    'W1' → 'well-1'
    """
    import re

    # Convert to lowercase
    name = raw_name.lower()

    # Remove common prefixes and clean
    name = re.sub(r'^(well|w)[\s_-]*', '', name)

    # Remove leading zeros: '001' → '1'
    name = name.lstrip('0') or '0'

    # Standardize format

    return f"well-{name}"


def detect_well_from_query(query: str) -> Optional[str]:
    """
    Use this tool to extract well reference from user query.

    Input:
        query (str): The user's query string.

    Returns:
        Normalized well name (e.g., "well-4") or None
    """
    query_lower = query.lower()

    # Pattern 1: "well 4", "well-4", "well_4"
    match = re.search(r'\bwell[\s\-_]*(\d+)\b', query_lower)
    if match:
        well_num = match.group(1).lstrip('0') or '0'
        normalized = f"well-{well_num}"
        logger.info(f"Detected well from query: {normalized}")
        return normalized

    # Pattern 2: "w4", "w-4"
    match = re.search(r'\bw[\s\-_]*(\d+)\b', query_lower)
    if match:
        well_num = match.group(1).lstrip('0') or '0'
        normalized = f"well-{well_num}"
        logger.info(f"Detected well from query (short form): {normalized}")
        return normalized

    logger.debug("No well reference detected in query")
    return None