"""
Document Management Routes.

Handles document upload, listing, and deletion.
All operations are synchronous REST endpoints.
"""

from app.utils.logger import get_logger
from pathlib import Path
from typing import List
import shutil
import tempfile

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.services.document_service import DocumentService, get_document_service
from app.models.document import DocumentUploadResponse, DocumentProcessingError
from app.core.config import settings

logger = get_logger(__name__)


router = APIRouter()

# ==================== Upload Document ====================

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
        file: UploadFile = File(..., description="PDF document to upload"),
        service: DocumentService = Depends(get_document_service)
):
    """
    Upload and process a document.

    Process:
    1. Validate file (type, size) 2. Save to temp location (don't fill RAM) 3. Process (extract, chunk, embed, index) 4. Return status

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
    # Teaching: Why temp file?
    # - Don't load entire file in memory  (don't fill RAM)
    # - Process from disk (safer for large files)
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
        return result

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

    Args:
        document_id: Document identifier

    Returns:
        Document status information

    Raises:
        404: Document not found
    """
    status_info = service.get_document_status(document_id)

    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )

    return status_info


# ==================== Get Document Details ====================

@router.get("/{document_id}")
async def get_document(
        document_id: str,
        service: DocumentService = Depends(get_document_service)
):
    """
    Get complete document metadata.

    Returns:
        Full document information

    Raises:
        404: Document not found
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

    Args:
        document_id: Document to delete

    Returns:
        Success message

    Raises:
        404: Document not found
        :param service:
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

    Returns:
        Stats about documents, chunks, vector store
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



