from pathlib import Path
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
