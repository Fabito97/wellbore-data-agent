"""
Folder utilities for well document processing.

Handles:
- ZIP structure normalization (multiple wrapper layers)
- Well name extraction
- Document type detection
- Well report filtering (ONLY well reports processed)
"""
from pathlib import Path
from typing import Dict, List, Optional
import re
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ONLY process PDFs for now
SUPPORTED_EXTENSIONS = {".pdf"}

# Document type detection (case-insensitive)
DOCUMENT_TYPE_FOLDERS = {
    "pvt": "PVT",
    "production": "PRODUCTION",
    "production data": "PRODUCTION",
    "technical logs": "TECHNICAL_LOGS",
    "technical log": "TECHNICAL_LOGS",
    "well report": "WELL_REPORT",  # ← TARGET: Only process these!
    "well test": "WELL_TEST",
}


def normalize_zip_structure(root: Path) -> Path:
    """
    Unwrap multiple wrapper folders to find the actual well folder.

    Handles structures like:
    - Well 4-20251107T093405Z-1-001/well 4/PVT/...
    - well_1/pvt/...
    - well_1.zip -> well_1/...

    Returns the innermost well folder containing report subfolders.
    """
    current = root

    # Keep unwrapping until we find a folder with multiple subdirectories
    # (which indicates it's the well folder with report types)
    max_depth = 5  # Safety limit
    depth = 0

    while depth < max_depth:
        children = list(current.iterdir())

        # If we have multiple folders, this is likely the well folder
        folders = [c for c in children if c.is_dir()]
        if len(folders) > 1:
            logger.info(f"Found well folder with {len(folders)} report types: {current.name}")
            return current

        # If exactly one folder, go deeper
        if len(children) == 1 and children[0].is_dir():
            logger.debug(f"Unwrapping layer: {children[0].name}")
            current = children[0]
            depth += 1
        else:
            # No more folders to unwrap
            break

    logger.info(f"Final well folder: {current.name}")
    return current


def extract_well_name(folder_name: str) -> str:
    """
    Extract clean well name from folder name.

    Examples:
    - "well 4" → "well-4"
    - "Well 4-20251107T093405Z-1-001" → "well-4"
    - "WELL_001" → "well-1"
    - "W4" → "well-4"
    """
    # Convert to lowercase
    name = folder_name.lower()

    # Remove timestamps and other noise
    # Pattern: anything after a dash followed by 8+ digits (timestamp)
    name = re.sub(r'-\d{8}.*$', '', name)

    # Extract well number
    # Look for patterns like: "well 4", "well_4", "w4", "004"
    match = re.search(r'(?:well|w)[\s_-]*(\d+)', name)
    if match:
        well_num = match.group(1).lstrip('0') or '0'  # Remove leading zeros
        return f"well-{well_num}"

    # Fallback: just clean the name
    clean = name.strip().replace(' ', '-').replace('_', '-')
    # Remove any remaining non-alphanumeric except hyphens
    clean = re.sub(r'[^a-z0-9-]', '', clean)

    return clean if clean else "unknown"


def infer_document_type(folder_name: str) -> Optional[str]:
    """
    Infer document type from folder name (case-insensitive).

    Args:
        folder_name: Name of the folder (e.g., "Well report", "PVT")

    Returns:
        Document type string or None if not recognized
    """
    name_lower = folder_name.lower().strip()

    # Direct match first
    if name_lower in DOCUMENT_TYPE_FOLDERS:
        return DOCUMENT_TYPE_FOLDERS[name_lower]

    # Partial match (for variations like "well report 2025")
    for key, value in DOCUMENT_TYPE_FOLDERS.items():
        if key in name_lower:
            logger.debug(f"Matched '{folder_name}' to type '{value}' via key '{key}'")
            return value

    logger.debug(f"No document type match for folder: {folder_name}")
    return None


def scan_folder_structure(well_root: Path) -> Dict:
    """
    Scan well folder and return structure with ONLY well reports.

    Process:
    1. Find all subfolders
    2. Identify document types
    3. Filter to ONLY "WELL_REPORT" folders
    4. Find all PDFs in those folders

    Returns:
        {
            "well_name": "well-4",
            "reports": [
                {
                    "document_type": "WELL_REPORT",
                    "folder_name": "Well report",
                    "folder": Path(...),
                    "documents": [Path(...), ...]
                }
            ]
        }
    """
    if not well_root.exists():
        raise ValueError(f"Folder does not exist: {well_root}")

    logger.info(f"Scanning well folder: {well_root}")

    # Extract well name from folder
    well_name = extract_well_name(well_root.name)
    logger.info(f"Extracted well name: {well_name}")

    structure = {
        "well_name": well_name,
        "reports": []
    }

    # Scan all subfolders
    for subfolder in well_root.iterdir():
        if not subfolder.is_dir():
            logger.debug(f"Skipping non-folder: {subfolder.name}")
            continue

        # Identify document type
        doc_type = infer_document_type(subfolder.name)

        # ✅ CRITICAL: Only process WELL_REPORT folders!
        if doc_type != "WELL_REPORT":
            logger.info(f"⏭️  Skipping folder (not well report): {subfolder.name} (type: {doc_type})")
            continue

        logger.info(f"✅ Processing WELL REPORT folder: {subfolder.name}")

        # Find all PDFs in this folder (recursive)
        documents = []
        for file_path in subfolder.rglob("*"):
            if not file_path.is_file():
                continue

            # Only PDFs
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.debug(f"Skipping non-PDF: {file_path.name}")
                continue

            documents.append(file_path)
            logger.debug(f"Found PDF: {file_path.name}")

        # Add to structure if we found documents
        if documents:
            structure["reports"].append({
                "document_type": doc_type,
                "folder_name": subfolder.name,
                "folder": subfolder,
                "documents": documents
            })
            logger.info(f"Added {len(documents)} documents from {subfolder.name}")
        else:
            logger.warning(f"No PDFs found in {subfolder.name}")

    # Summary
    total_docs = sum(len(r["documents"]) for r in structure["reports"])
    logger.info(
        f"Scan complete: {well_name} → "
        f"{len(structure['reports'])} well report folders, "
        f"{total_docs} total PDFs"
    )

    return structure


def list_all_wells(root: Path) -> List[str]:
    """
    List all well names found in a directory.

    Useful for batch processing multiple wells.
    """
    wells = []

    for item in root.iterdir():
        if item.is_dir():
            well_name = extract_well_name(item.name)
            wells.append(well_name)

    return sorted(set(wells))  # Unique, sorted