"""
Document Management Routes.

Handles document upload, listing, and deletion.
All operations are synchronous REST endpoints.
"""
from pydantic import BaseModel

from app.utils.folder_utils import extract_zip_to_temp, cleanup_temp_paths
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

    temp_zip_path , extract_dir = None, None

    try:
        # Extract ZIP to temp
        temp_zip_path, extract_dir = extract_zip_to_temp(source_file=file)

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
            cleanup_temp_paths(temp_zip_path, extract_dir)


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


@router.get("/{document_id}/chunks")
async def list_documents_with_chunks(
        document_id: str,
        service: DocumentService = Depends(get_document_service)
):
    """
    List all documents in the system.

    Returns:
        List of document summaries
    """
    try:
        documents = service.get_document_and_chunks(document_id)
        return {
            "total": len(documents),
            "data": documents
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

@router.delete("/store/clear")
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
    return {
        "message": "Document retrieved successfully",
        "document": document
    }


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

@router.delete("/state/wipe")
async def clear_all_documents(
        service: DocumentService = Depends(get_document_service)
):
    """
    Clears all documents including upload and vector storage
    """
    success = service.reset_system()

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to clear all documents"
        )

    return {
        "message": f"Successfully wiped all documents from the system"
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
