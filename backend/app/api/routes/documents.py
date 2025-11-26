"""
Document Management Routes.

Handles document upload, listing, and deletion.
All operations are synchronous REST endpoints.
"""
from pydantic import BaseModel

from app.utils.logger import get_logger
from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Path as FastAPIPath
from fastapi.responses import JSONResponse

from app.services.document_service import DocumentService, get_document_service
from app.models.document import DocumentUploadResponse, DocumentProcessingError, BatchUploadResponse

logger = get_logger(__name__)


router = APIRouter()

# ==================== Upload Document ====================

class DocumentUploadApiResponse(BaseModel):
    message: str
    status: str
    data: BatchUploadResponse | DocumentUploadResponse

@router.post("/upload-zip", response_model=DocumentUploadApiResponse)
async def upload_document_zip(
        file: UploadFile = File(..., description="PDF document to upload"),
        service: DocumentService = Depends(get_document_service)
):
    logger.info(f"Received upload: {file.filename}")

    if file.content_type not in ["application/zip", "application/x-zip-compressed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only ZIP files are supported"
        )

    temp_zip_path = None
    extract_dir = None

    try:
        # Save ZIP to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
            shutil.copyfileobj(file.file, temp_zip)
            temp_zip_path = Path(temp_zip.name)

        logger.info(f"ZIP saved to temp: {temp_zip_path}")

        # Create temp extraction directory
        extract_dir = Path(tempfile.mkdtemp())

        # Extract ZIP
        shutil.unpack_archive(temp_zip_path, extract_dir)
        logger.info(f"ZIP extracted to: {extract_dir}")

        # Process folder instead of single file
        result = service.ingest_folder(
            folder_path=extract_dir,
            original_filename=file.filename
        )

        if isinstance(result, DocumentProcessingError):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error
            )

        return DocumentUploadApiResponse(
            status="success",
            message=f"Folder uploaded and processed successfully in {result.total_time:.2f} seconds",
            data=result
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))

    finally:
        # Clean up temp files/folders
        if temp_zip_path and temp_zip_path.exists():
            try:
                temp_zip_path.unlink()
                logger.debug(f"Deleted temp ZIP: {temp_zip_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp ZIP: {e}")

        if extract_dir and extract_dir.exists():
            try:
                shutil.rmtree(extract_dir)
                logger.debug(f"Deleted temp extraction folder: {extract_dir}")
            except Exception as e:
                logger.warning(f"Failed to delete temp extraction folder: {e}")


@router.post("/upload", response_model=DocumentUploadApiResponse)
async def upload_document(
        file: UploadFile = File(..., description="PDF document to upload"),
        service: DocumentService = Depends(get_document_service)
):
    """
    Upload and process a document.

    Process:
    1. Validate file (type, size)
    2. Save to temp location (don't fill RAM)
    3. Process (extract, chunk, embed, index)
    4. Return status

    Args:
        file: Uploaded PDF file
        service: Document service (injected)

    Returns:
        DocumentUploadResponse with processing status

    Raises:
        400: Invalid file type
        413: File too large
        500: Processing failed
    """
    logger.info(f"Received upload: {file.filename}")

    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    # Create temp file
    # - Don't fill RAM - Process from disk (safer for large files)
    # - Auto-cleanup with tempfile
    temp_path = None

    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = Path(temp.name)

        logger.info(f"File saved to temp: {temp_path}")

        # Process document
        result = service.ingest_document(
            file_path=temp_path,
            original_filename=file.filename
        )

        # Check if processing failed
        if isinstance(result, DocumentProcessingError):
            logger.error(f"Processing failed: {result.error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error
            )

        logger.info(f"Document processed successfully: {result.document_id}")
        return DocumentUploadApiResponse(
            status="success",
            message=f"Document uploaded and processed successfully in {result.elapsed_time:.2f} seconds",
            data=result
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )

    finally:
        # Clean up temp file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean temp file: {e}")


# ==================== List Documents ====================

@router.get("/")
async def list_documents(
        service: DocumentService = Depends(get_document_service)
):
    """
    List all documents in the system.

    Returns:
        List of document summaries
    """
    try:
        documents = service.list_documents()
        return {
            "total": len(documents),
            "documents": documents
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )


# ==================== Get Document Status ====================

@router.get("/{document_id}/status")
async def get_document_status(
        document_id: str,
        service: DocumentService = Depends(get_document_service)
):
    """
    Get processing status of a document.
    """
    status_info = service.get_document_status(document_id)

    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )

    return status_info


# ==================== Get Document Details ====================

@router.delete("/store/clear-store")
async def clear_store(
        service: DocumentService = Depends(get_document_service)
):
    """
    Clear vector store
    """
    try:
        store = service.vector_store
        store.clear_store()
        return {
            "message": "Store cleared successfully"
        }
    except Exception as e:
        logger.error(f"Failed to clear store: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear store"
        )

@router.get("/{document_id}")
async def get_document(
        document_id: str,
        service: DocumentService = Depends(get_document_service)
):
    """
    Get complete document metadata.
    """
    document = service.get_document(document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )

    # Return as dict (Pydantic model to JSON)
    return document.model_dump()


# ==================== Delete Document ====================

@router.delete("/{document_id}")
async def delete_document(
        document_id: str,
        service: DocumentService = Depends(get_document_service)
):
    """
    Delete a document and all its data.
    """
    success = service.delete_document(document_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found or deletion failed"
        )

    return {
        "message": f"Document {document_id} deleted successfully"
    }


# ==================== Get Service Stats ====================

@router.get("/stats/summary")
async def get_stats(
        service: DocumentService = Depends(get_document_service)
):
    """
    Get overall system statistics.
    """
    try:
        stats = service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


# ==================== WELL MANAGEMENT ====================

class WellCreateRequest(BaseModel):
    name: str


@router.get("/wells")
async def list_wells(
    service: DocumentService = Depends(get_document_service)
):
    """List all wells (summary)."""
    try:
        wells = service.list_wells()
        return {"total": len(wells), "wells": wells}
    except Exception as e:
        logger.error(f"Failed to list wells: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list wells")


@router.get("/wells/with-documents")
async def list_wells_with_documents(
    service: DocumentService = Depends(get_document_service)
):
    """List wells including their documents."""
    try:
        wells = service.list_wells_with_documents()
        return {"total": len(wells), "wells": wells}
    except Exception as e:
        logger.error(f"Failed to list wells with documents: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list wells")


@router.post("/wells", status_code=status.HTTP_201_CREATED)
async def create_or_get_well(
    payload: WellCreateRequest,
    service: DocumentService = Depends(get_document_service)
):
    """Create a well or return existing one by normalized name."""
    try:
        well = service.get_or_create_well(payload.name)
        return {
            "id": well.id,
            "name": well.name,
            "document_count": well.document_count,
            "created_at": well.created_at.isoformat() if getattr(well, "created_at", None) else None
        }
    except Exception as e:
        logger.error(f"Failed to create/get well: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/wells/{well_id}")
async def get_well_by_id(
    well_id: str = FastAPIPath(..., description="Well id (e.g. well-xxxx)"),
    service: DocumentService = Depends(get_document_service)
):
    """Get well by id (includes documents list)."""
    try:
        well = service.get_well(well_id=well_id)
        if not well:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Well {well_id} not found")
        # Convert datetimes to isoformat where present
        well["created_at"] = well["created_at"].isoformat() if well.get("created_at") else None
        return well
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get well by id: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/wells/by-name/{well_name}")
async def get_well_by_name(
    well_name: str,
    service: DocumentService = Depends(get_document_service)
):
    """Get well by name (normalized)"""
    try:
        well = service.get_well(well_name=well_name)
        if not well:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Well {well_name} not found")
        well["created_at"] = well["created_at"].isoformat() if well.get("created_at") else None
        return well
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get well by name: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/wells/{well_id}/documents")
async def list_documents_for_well(
    well_id: str = FastAPIPath(..., description="Well id"),
    service: DocumentService = Depends(get_document_service)
):
    """List documents for a given well id."""
    try:
        docs = service.list_documents_by_well(well_id)
        return {"total": len(docs), "documents": docs}
    except Exception as e:
        logger.error(f"Failed to list documents for well {well_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
