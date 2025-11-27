"""
Batch ingestion runner for Wellbore AI Agent.

This script scans the RAW_DIR defined in app.core.config.Settings,
handles ZIP extraction or folder ingestion, and moves items into
PROCESSED_DIR or FAILED_DIR after processing.
"""
from app.utils.logger import  get_logger
import shutil

from app.core.config import settings
from app.core.database import get_db  # your SQLAlchemy session factory
from app.services.document_service import DocumentService
from app.utils.folder_utils import extract_zip_to_temp, cleanup_temp_paths

logger = get_logger(__name__)


def ingest_uploads():
    """
    Scan the upload directory, extract ZIPs or use folders directly,
    and run ingestion via DocumentService.ingest_folder.
    """
    # Manually create a DB session (context manager recommended)
    db = next(get_db())  # get_db is usually a generator in FastAPI
    service = DocumentService(db)


    upload_dir = settings.RAW_DIR
    processed_dir = settings.PROCESSED_DIR
    failed_dir = settings.DATA_DIR / "failed"  # optional extra folder for failed uploads
    processed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    # Ensure upload dir exists
    if not upload_dir.exists():
        logger.warning(f"📂 Upload directory '{upload_dir}' does not exist. Creating it now...")
        upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Upload directory created, but it is empty. Nothing to ingest.")
        return

    # Validation: check if upload dir has data
    items = list(upload_dir.iterdir())
    if not items:
        logger.info(f"📂 Upload directory '{upload_dir}' is empty. Nothing to ingest.")
        return

    for item in upload_dir.iterdir():
        try:
            logger.info(f"📂 Found item: {item.name}")

            # Case 1: ZIP file
            if item.suffix.lower() == ".zip":
                temp_zip_path, extract_dir = extract_zip_to_temp(source=item)
                try:
                    result = service.ingest_folder(folder_path=extract_dir, original_filename=item.name)
                    target_dir = processed_dir if not result.errors else failed_dir
                    shutil.move(str(item), target_dir / item.name)
                    logger.info(f"✅ Moved {item.name} to {target_dir}")
                finally:
                    cleanup_temp_paths(temp_zip_path, extract_dir)

            # Case 2: Already a folder
            elif item.is_dir():
                result = service.ingest_folder(folder_path=item, original_filename=item.name)
                target_dir = processed_dir if not result.errors else failed_dir
                item.rename(target_dir / item.name)
                logger.info(f"✅ Moved {item.name} to {target_dir}")

            else:
                logger.warning(f"⚠️ Unsupported item in upload dir: {item.name}")

        except Exception as e:
            logger.error(f"💥 Failed to ingest {item.name}: {e}")
            item.rename(failed_dir / item.name)

def ask_keep_mode():
    print("Choose how to archive after ingestion:")
    print("1. Keep ZIP only")
    print("2. Keep extracted only")
    print("3. Keep both")
    choice = input("Enter choice [1/2/3]: ").strip()
    return {"1": "zip", "2": "extracted", "3": "both"}.get(choice, "zip")

if __name__ == "__main__":
    ingest_uploads()