"""
Document Service with Database Persistence.

Now stores:
- Well metadata in DB
- Document metadata in DB
- Chunks in vector store (as before)
"""
from app.utils.folder_utils import scan_folder_structure, normalize_zip_structure, extract_well_name
from app.utils.helper import normalize_well_name
from app.utils.logger import get_logger
from datetime import datetime
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import time
from sqlalchemy.orm import Session
from sqlalchemy import select
from fastapi import Depends

from app.models.document import (
    DocumentContent,
    DocumentStatus,
    DocumentUploadResponse,
    DocumentProcessingError,
    BatchUploadResponse
)
from app.core.database import Well, Document, get_db
from app.rag.document_processor import process_pdf
from app.rag.chunking import chunk_document
from app.rag.embeddings import embed_chunks
from app.rag.vector_store_manager import get_vector_store
from app.core.config import settings

logger = get_logger(__name__)


class DocumentService:
    """Service for document management with database persistence."""

    def __init__(self, db: Session):
        """Initialize with database session."""
        self.db = db
        self.vector_store = get_vector_store()
        logger.info("DocumentService initialized with database")

    # ==================== Well Management ====================

    def get_or_create_well(self, well_name: str) -> Well:
        """Get existing well or create new one."""
        # Check if exists
        stmt = select(Well).where(Well.name == well_name)
        well = self.db.execute(stmt).scalar_one_or_none()

        if well:
            logger.debug(f"Found existing well: {well_name}")
            return well

            # Base ID from timestamp
        now = datetime.utcnow()
        timestamp_str = now.strftime("%Y%m%d%H%M%S")
        base_id = f"{well_name}_{timestamp_str}"

        # Check for existing wells with same base_id
        stmt = select(Well).where(Well.id.like(f"{base_id}_%"))
        existing = self.db.execute(stmt).scalars().all()
        version = len(existing) + 1

        new_well = Well(
            id=f"{base_id}_{version:03d}",  # e.g. Well-1_20251127013945_001

            name=well_name,
            document_count=0,
            created_at=now
        )

        self.db.add(new_well)
        self.db.commit()
        self.db.refresh(new_well)

        logger.info(f"✅ Created well: {well_name} ({new_well.id})")
        return new_well

    def list_wells(self) -> List[Dict[str, Any]]:
        """List all wells."""
        stmt = select(Well).order_by(Well.created_at.desc())
        wells = self.db.execute(stmt).scalars().all()

        return [
            {
                "id": w.id,
                "name": w.name,
                "document_count": w.document_count,
                "created_at": w.created_at.isoformat()
            }
            for w in wells
        ]

    def list_wells_with_documents(self) -> List[Dict[str, Any]]:
        stmt = select(Well).order_by(Well.created_at.desc())
        wells = self.db.execute(stmt).scalars().all()

        return [
            {
                "id": w.id,
                "name": w.name,
                "document_count": len(w.documents),
                "documents": [
                    {
                        "id": d.id,
                        "filename": d.filename,
                        "document_type": d.document_type,
                        "uploaded_at": d.uploaded_at.isoformat()
                    }
                    for d in w.documents
                ],
                "created_at": w.created_at.isoformat()
            }
            for w in wells
        ]

    def list_documents_by_well(self, well_id: str) -> List[Dict[str, Any]]:
        stmt = select(Document).where(Document.well_id == well_id).order_by(Document.uploaded_at.desc())
        docs = self.db.execute(stmt).scalars().all()

        return [
            {
                "document_id": d.id,
                "filename": d.filename,
                "document_type": d.document_type,
                "uploaded_at": d.uploaded_at.isoformat()
            }
            for d in docs
        ]

    def get_well(
            self,
            well_id: int | None = None,
            well_name: str | None = None
    ):
        # Validate
        if (well_id and well_name) or (not well_id and not well_name):
            raise ValueError("Provide exactly one of well_id or well_name.")

        stmt = None

        if well_id:
            stmt = select(Well).where(Well.id == well_id)
        else:
            normalized_name = normalize_well_name(well_name)
            stmt = select(Well).where(Well.name == normalized_name)

        well = self.db.execute(stmt).scalar_one_or_none()
        if not well:
            return None  # or raise NotFound

        # Load documents for the well
        docs_stmt = select(Document).where(Document.well_id == well.id)
        docs = self.db.execute(docs_stmt).scalars().all()

        return {
            "id": well.id,
            "name": well.name,
            "created_at": well.created_at,
            "document_count": len(docs),
            "documents": [
                {
                    "document_id": d.id,
                    "filename": d.filename,
                    "document_type": d.document_type
                }
                for d in docs
            ]
        }

    # ==================== Folder Ingestion ====================

    def ingest_folder(
        self,
        folder_path: Path,
        original_filename: str,
        well_name: Optional[str] = None,
    ) -> BatchUploadResponse:
        """Ingest well reports from folder with DB persistence."""
        start_time = time.time()
        logger.info(f"🚀 [BatchUpload] Starting: {original_filename}")

        try:
            # Normalize folder
            normalized_folder = normalize_zip_structure(folder_path)
            logger.info(f"📁 Normalized: {normalized_folder.name}")

            # Extract well name
            if not well_name:
                well_name = extract_well_name(normalized_folder.name)
            logger.info(f"🔍 Well name: {well_name}")

            # Get or create well in DB
            well = self.get_or_create_well(well_name)
            well_id = well.id

            # Scan structure
            structure = scan_folder_structure(normalized_folder)
            report_groups = structure.get("reports", [])

            if not report_groups:
                logger.warning("⚠️  No well reports found!")
                return BatchUploadResponse(
                    total_documents=0,
                    successful=0,
                    failed=0,
                    documents=[],
                    errors=[{"filename": original_filename, "error": "No well reports found"}],
                    total_time=time.time() - start_time,
                    message="No well reports found"
                )

            total_documents = sum(len(r.get("documents", [])) for r in report_groups)
            logger.info(f"📊 Found {total_documents} PDFs")

            # Process documents
            documents = []
            errors = []

            for report in report_groups:
                doc_type = report.get("document_type")
                file_list = report.get("documents", [])

                for idx, file_path in enumerate(file_list, 1):
                    logger.info(f"  [{idx}/{len(file_list)}] {file_path.name}")

                    try:
                        resp = self.ingest_document(
                            file_path=file_path,
                            original_filename=file_path.name,
                            well_id=well_id,
                            well_name=well_name,
                            document_type=doc_type,
                            original_folder_path=str(file_path.parent.relative_to(folder_path))
                        )

                        if isinstance(resp, DocumentUploadResponse):
                            documents.append(resp)
                            logger.info(f"  ✅ Success")
                        else:
                            errors.append({"filename": file_path.name, "error": resp.error})
                            logger.error(f"  ❌ Error")

                    except Exception as e:
                        logger.exception(f"  💥 Exception: {e}")
                        errors.append({"filename": file_path.name, "error": str(e)})

            # Update well document count
            well.document_count = len(documents)
            self.db.commit()

            elapsed = time.time() - start_time
            message = f"Well {well_name}: {len(documents)}/{total_documents} processed in {elapsed:.1f}s"
            logger.info(f"✅ {message}")

            return BatchUploadResponse(
                total_documents=total_documents,
                successful=len(documents),
                failed=len(errors),
                documents=documents,
                errors=errors,
                total_time=elapsed,
                message=message
            )

        except Exception as e:
            logger.exception(f"💥 Batch upload failed: {e}")
            return BatchUploadResponse(
                total_documents=0,
                successful=0,
                failed=1,
                documents=[],
                errors=[{"filename": original_filename, "error": str(e)}],
                total_time=time.time() - start_time,
                message=f"Failed: {str(e)}"
            )

    # ==================== Document Ingestion ====================

    def ingest_document(
        self,
        file_path: Path,
        original_filename: str,
        document_id: Optional[str] = None,
        well_id: Optional[str] = None,
        well_name: Optional[str] = None,
        document_type: Optional[str] = None,
        original_folder_path: Optional[str] = None,
        file_format: Optional[str] = None,
    ) -> Union[DocumentUploadResponse, DocumentProcessingError]:
        """Ingest single document with DB persistence."""
        start_time = time.time()
        doc_id = document_id or str(uuid.uuid4())

        logger.info(f"Ingesting {well_name}: {original_filename} ({doc_id})")

        permanent_path = ""
        try:
            # Save file
            permanent_path = self._save_file_permanently(
                file_path, doc_id, original_filename, well_id
            )

            # Process PDF
            document_content = process_pdf(permanent_path, document_id=doc_id)
            document_content.filename = original_filename
            document_content.well_id = well_id
            document_content.well_name = well_name
            document_content.document_type = document_type
            document_content.original_folder_path = original_folder_path
            document_content.file_format = file_format or "pdf"

            # Chunk
            chunks = chunk_document(document_content)
            document_content.chunk_count = len(chunks)

            # Embed
            chunks_with_embeddings = embed_chunks(chunks)

            # Store in vector DB
            added = self.vector_store.add_chunks(chunks_with_embeddings)
            if added != len(chunks):
                logger.warning(f"Added {added}/{len(chunks)} chunks")


            # Store metadata in DB
            db_document = Document(
                id=doc_id,
                filename=original_filename,
                file_path=str(permanent_path),
                well_id=well_id,
                well_name=well_name,
                document_type=document_type,
                file_format=file_format or "pdf",
                original_folder_path=original_folder_path,
                status="indexed",
                page_count=document_content.page_count,
                word_count=document_content.total_word_count,
                table_count=document_content.table_count,
                chunk_count=len(chunks),
                processing_time_seconds=time.time() - start_time,
                extraction_method=str(document_content.extraction_method.value),
                ocr_enabled=document_content.ocr_enabled,
                processed_at=datetime.utcnow()
            )

            self.db.add(db_document)
            self.db.commit()
            self.db.refresh(db_document)

            elapsed = time.time() - start_time
            logger.info(f"✅ Indexed: {elapsed:.2f}s, {len(chunks)} chunks")

            return DocumentUploadResponse(
                document_id=doc_id,
                filename=original_filename,
                well_id=well_id or "unknown",
                well_name=well_name or "unknown",
                document_type=document_type or "unknown",
                format=file_format or "pdf",
                status=DocumentStatus.INDEXED,
                page_count=document_content.page_count,
                word_count=document_content.total_word_count,
                table_count=document_content.table_count,
                chunk_count=len(chunks),
                uploaded_at=db_document.uploaded_at.isoformat(),
                elapsed_time=elapsed
            )

        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            self.db.rollback()

            try:
                if permanent_path.exists() and permanent_path.is_file():
                    permanent_path.unlink()
                    logger.debug(f"Deleted failed file: {permanent_path}")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed: {cleanup_error}")

            return DocumentProcessingError(
                document_id=doc_id,
                filename=original_filename,
                error=str(e),
                details="See logs"
            )

    # ==================== Query Methods ====================

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.db.get(Document, document_id)

    def get_document_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Return processing/status information for a document (JSON-serializable)."""
        doc = self.db.get(Document, document_id)
        if not doc:
            return None

        return {
            "document_id": doc.id,
            "status": doc.status,
            "processing_time_seconds": getattr(doc, "processing_time_seconds", None),
            "uploaded_at": doc.uploaded_at.isoformat() if getattr(doc, "uploaded_at", None) else None,
            "processed_at": doc.processed_at.isoformat() if getattr(doc, "processed_at", None) else None,
            "page_count": getattr(doc, "page_count", None),
            "chunk_count": getattr(doc, "chunk_count", None),
            "table_count": getattr(doc, "table_count", None),
            "well_id": getattr(doc, "well_id", None),
            "well_name": getattr(doc, "well_name", None),
        }

    def list_documents(
        self,
        well_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List documents with optional well filter."""
        stmt = select(Document).order_by(Document.uploaded_at.desc())

        if well_name:
            stmt = stmt.where(Document.well_name == well_name)

        docs = self.db.execute(stmt).scalars().all()

        return [
            {
                "document_id": d.id,
                "filename": d.filename,
                "well_name": d.well_name,
                "well_id": d.well_id,
                "file_format": d.file_format,
                "table_count": d.table_count,
                "document_type": d.document_type,
                "status": d.status,
                "page_count": d.page_count,
                "chunk_count": d.chunk_count,
                "uploaded_at": d.uploaded_at.isoformat(),
                "processed_at": d.processed_at.isoformat() if d.processed_at else ""
            }
            for d in docs
        ]

    def delete_document(self, document_id: str) -> bool:
        """Delete document from DB and vector store."""
        try:
            document = self.db.get(Document, document_id)
            if not document:
                return False

            # Delete from vector store
            deleted = self.vector_store.delete_by_document_id(document_id)
            logger.debug(f"Deleted {deleted} chunks")

            # Delete file
            try:
                Path(document.file_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete file: {e}")

            # Delete from DB
            self.db.delete(document)

            # Update well count
            if document.well_id:
                well = self.db.get(Well, document.well_id)
                if well:
                    well.document_count = max(0, well.document_count - 1)

            self.db.commit()

            logger.info(f"Deleted document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            self.db.rollback()
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        from sqlalchemy import func as sql_func

        total_docs = self.db.execute(select(sql_func.count(Document.id))).scalar()
        total_wells = self.db.execute(select(sql_func.count(Well.id))).scalar()
        total_pages = self.db.execute(select(sql_func.sum(Document.page_count))).scalar() or 0
        total_chunks = self.db.execute(select(sql_func.sum(Document.chunk_count))).scalar() or 0

        vector_stats = self.vector_store.get_stats()

        return {
            "total_wells": total_wells,
            "total_documents": total_docs,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "vector_store": vector_stats,
        }

    def _save_file_permanently(
            self,
            temp_path: Path,
            document_id: str,
            original_filename: str,
            well_id: Optional[str] = None,
            doc_type: Optional[str] = None,
            is_image: bool = False
    ) -> Path:
        """Save file to permanent storage."""
        well_folder = well_id or "unknown_well"

        subfolder = "images" if is_image else "docs"
        # if doc_type:
        #     lower_doc_type = doc_type.lower()
        #     subfolder = Path(lower_doc_type) / "images" if is_image else Path(lower_doc_type)

        folder = settings.UPLOAD_DIR / well_folder / subfolder
        folder.mkdir(parents=True, exist_ok=True)

        extension = Path(original_filename).suffix
        permanent_filename = f"{document_id}{extension}"
        permanent_path = folder / permanent_filename

        shutil.copy(str(temp_path), str(permanent_path))
        logger.debug(f"Saved: {permanent_path}")
        return permanent_path


# ==================== FastAPI Dependency ====================

def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    """Get document service with DB session."""
    return DocumentService(db)