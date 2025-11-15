"""
Document Service - Orchestrates document ingestion pipeline.

This service coordinates the RAG pipeline:
- File handling
- Document processing
- Chunking
- Embedding
- Vector storage

Teaching: Service Layer Pattern
- Sits between API routes and core logic
- Orchestrates multiple components
- Handles business logic
- Returns clean results to API

Why separate service?
- API routes stay thin (just HTTP handling)
- Business logic in one place
- Easy to test
- Reusable across different interfaces (REST, WebSocket, CLI)
"""
from app.utils.logger import get_logger
from datetime import datetime
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
import time

from app.models.document import (
    DocumentContent,
    DocumentStatus,
    DocumentUploadResponse,
    DocumentProcessingError
)
from app.rag.document_processor import process_pdf
from app.rag.chunking import chunk_document
from app.rag.embeddings import embed_chunks
from app.rag.vector_store_manager import get_vector_store
from app.core.config import settings

logger = get_logger(__name__)



class DocumentService:
    """
    Service for document management and ingestion.
    """

    def __init__(self):
        """Initialise service with required components."""
        self.vector_store = get_vector_store()

        # In-memory document registry (simple for hackathon)
        self._documents: Dict[str, DocumentContent] = {}

        logger.info("DocumentService initialized")


    def ingest_document(
            self,
            file_path: Path,
            original_filename: str,
            document_id: Optional[str] = None
    ) -> DocumentUploadResponse | DocumentProcessingError:
        """
        Complete document ingestion pipeline.

        Process:
        1. Generate document ID
        2. Move file to permanent storage
        3. Process PDF (extract text/tables)
        4. Chunk content
        5. Generate embeddings
        6. Store in vector database
        7. Update registry

        Args:
            file_path: Path to uploaded file (temp location)
            original_filename: Original filename from user
            document_id: Optional custom ID

        Returns:
            DocumentUploadResponse with status and metadata

        Raises:
            Exception: If any step fails

        Teaching: Pipeline orchestration
        - Each step depends on previous
        - If any fails, whole pipeline fails
        - Return meaningful error to user
        - Log everything for debugging
        """
        start_time = time.time()

        # Generate unique ID
        doc_id = document_id or str(uuid.uuid4())
        print(f"Starting ingestion for: {original_filename} (ID: {doc_id})")
        logger.info(f"Starting ingestion for: {original_filename} (ID: {doc_id})")

        try:
            # Step 1: Save file to permanent storage
            logger.debug("Step 1/5: Saving file to permanent storage")
            permanent_path = self._save_file_permanently(
                file_path,
                doc_id,
                original_filename,
            )

            # Step 2: Process PDF
            print(f"Step 2/5: Processing PDF with {permanent_path}")
            document = process_pdf(
                permanent_path,
                document_id=doc_id,
            )
            # Override filename to use original
            document.filename = original_filename

            # Step 3: Chunk document
            logger.debug("Step 3/5: Chunking document")
            chunks = chunk_document(document)

            # Update document with chunk count
            document.chunk_count = len(chunks)

            # Step 4: Generate embeddings
            logger.debug("Step 4/5: Generating embeddings")
            chunks_with_embeddings = embed_chunks(chunks)

            # Step 5: Store in vector database
            logger.debug("Step 5/5: Storing embeddings in vector database")
            added = self.vector_store.add_chunks(chunks_with_embeddings)

            if added != len(chunks):
                logger.warning(f"Added {added} chunks but expected {len(chunks)}")

            # Update status
            document.status = DocumentStatus.INDEXED
            document.processed_at = datetime.utcnow()
            document.processing_time_seconds = time.time() - start_time

            # Store in registry
            self._documents[doc_id] = document

            elapsed = time.time() - start_time

            logger.info(
                f"Document ingestion complete: {original_filename} "
                f"({elapsed:.2f}s, {len(chunks)} chunks)"
            )

            # Return response
            return DocumentUploadResponse(
                document_id=doc_id,
                filename=original_filename,
                status=DocumentStatus.INDEXED,
                page_count=document.page_count,
                word_count=document.total_word_count,
                table_count=document.table_count,
                chunk_count=document.chunk_count,
                uploaded_at=document.processed_at.isoformat(),
                message=f"Document processed successfully in {elapsed:.1f}s"
            )

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}", exc_info=True)

            # Return error response
            return DocumentProcessingError(
                document_id=doc_id,
                filename=original_filename,
                error=str(e),
                details="See logs for full traceback",
            )

    def get_document(self, document_id: str) -> Optional[DocumentContent]:
        """
        Retrieve document metadata by ID.

        Teaching: Simple registry lookup
        - In-memory for hackathon
        - Production: Query database
        """
        return self._documents.get(document_id)

    def list_documents(self) -> list[Dict[str, Any]]:
        """
        List all documents with summary info.

        Use case:
        - Show user their uploaded documents
        - Display in UI
        - Status tracking
        """
        return [doc.summary for doc in self._documents.values()]

    def delete_document(self, document_id: str) -> bool:
        """
        Delete document and all its chunks.

        Process:
        1. Remove from vector store
        2. Delete file from disk
        3. Remove from registry

        Teaching: Cleanup is important!
        - Don't leave orphaned data
        - Free up storage
        - Maintain consistency
        """
        try:
            logger.info(f"Deleting document {document_id}")

            # Get document
            document = self._documents.get(document_id)
            if not document:
                logger.warning(f"Document {document_id} not found in registry")
                return False

            # Step 1: Remove from vector store
            deleted = self.vector_store.delete_by_document_id(document_id)
            logger.debug(f"Deleted {deleted} chunks from vector store")

            # Step 2: Delete file from disk (optional - might want to keep)
            try:
                if document.file_path.exists():
                    document.file_path.unlink()
                    logger.info(f"Deleted file: {document.file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete file: {e}")

            # Step 3: Remove from registry
            del self._documents[document_id]

            logger.info(f"Document {document_id} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False


    def get_document_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of document processing.

        Use case:
        - Progress tracking for long uploads
        - Show status in UI
        - Debugging
        """
        document = self._documents.get(document_id)
        if not document:
            return None

        return {
            "document_id": document_id,
            "filename": document.filename,
            "status": document.status,
            "page_count": document.page_count,
            "chunk_count": document.chunk_count,
            "uploaded_at": document.uploaded_at.isoformat(),
            "processed_at": document.processed_at.isoformat() if document.processed_at else None,
        }

    def reprocess_document(self, document_id: str) -> DocumentUploadResponse:
        """
        Reprocess an existing document.

        Use case:
        - Settings changed (chunk size, etc.)
        - Bug fixed in processing
        - Want to regenerate embeddings

        Process:
        1. Get original file
        2. Delete old chunks
        3. Re-run ingestion
        """
        document = self._documents.get(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")

        logger.info(f"Reprocessing document: {document.filename}")

        # Delete old chunks
        self.vector_store.delete_by_document_id(document_id)

        # Re-ingest
        return self.ingest_document(
            document.file_path,
            document.filename,
            document_id=document_id,
        )

    def _save_file_permanently(
            self,
            temp_path: Path,
            document_id: str,
            original_filename: str
    ) -> Path:
        """
        Move uploaded file to permanent storage.

        Teaching: File management
        - Uploads come to temp directory
        - Move to permanent location
        - Use document_id in filename (avoid conflicts)
        - Preserve extension

        Storage structure:
        data/uploads/{document_id}_{original_filename}
        """
        # Get file extension
        extension = Path(original_filename).suffix

        # Create permanent filename
        permanent_filename= f"{document_id}{extension}"
        permanent_path = settings.UPLOAD_DIR / permanent_filename

        # Move file
        shutil.move(str(temp_path), str(permanent_path))

        logger.debug(f"File saved: {permanent_path}")
        return permanent_path

    def get_stats(self) -> Dict[str, Any]:
        """
         Get service statistics.
         """
        total_docs = len(self._documents)
        total_pages = sum(doc.page_count for doc in self._documents.values())
        total_chunks = sum(doc.chunk_count for doc in self._documents.values())

        vector_stats = self.vector_store.get_stats()

        return {
            "total_documents": total_docs,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "vector_store": vector_stats,
        }



# ==================== Module-level instance ====================

# Global service instance
_service_instance: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """
    Get or create global document service instance.

    Teaching: Service singleton
    - One service instance for whole app
    - Maintains document registry
    - Thread-safe (Python GIL)

    Usage:
        from app.services.document_service import get_document_service

        service = get_document_service()
        result = service.ingest_document(path, filename)
    """
    global _service_instance

    if _service_instance is None:
        _service_instance = DocumentService()

    return _service_instance
