from fastapi import APIRouter, UploadFile, File
from typing import List

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    # Placeholder: simulate document ingestion
    doc_ids = [f.filename.replace(".pdf", "") for f in files]
    return {"document_ids": doc_ids}


@router.get("/")
async def list_documents():
    # Return list of uploaded document IDs or metadata
    ...

@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    # Remove document from storage and vector DB
    ...


