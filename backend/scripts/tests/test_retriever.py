"""
Test script for DocumentRetriever.

Validates retrieval pipeline after document ingestion.

Usage:
    python scripts/test_retriever.py path/to/test.pdf
"""

import sys
from pathlib import Path
import shutil

from app.db.database import SessionLocal

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.logger import get_logger
from app.services.document_service import get_document_service
from app.rag.retriever import DocumentRetriever

logger = get_logger(__name__)


def ingest_test_document(pdf_path: Path) -> str:
    """Ingest document and return document_id."""
    logger.info("=" * 70)
    logger.info("STEP 1: Ingesting Document")
    logger.info("=" * 70)

    db = SessionLocal()
    service = get_document_service(db)

    temp_path = Path("temp_test.pdf")
    shutil.copy(pdf_path, temp_path)

    try:
        well_id = "well-1_27288447748_010"
        result = service.ingest_document(
            file_path=temp_path,
            original_filename=pdf_path.name,
            well_name="well-1",
            well_id=well_id,
            file_format="pdf",
            document_type="PVT",
            original_folder_path=str(temp_path)
        )

        if hasattr(result, 'error'):
            logger.info(f"\n❌ Ingestion failed: {result.error}")
            return None

        logger.info(f"\n✅ Ingestion complete: {result.filename}")
        logger.info(f"   Document ID: {result.document_id}")
        logger.info(f"   Chunks: {result.chunk_count}")
        return result.document_id

    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_basic_retrieval(document_id: str):
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Basic Retrieval")
    logger.info("=" * 70)

    retriever = DocumentRetriever(top_k=5)
    query = "water production"

    results = retriever.retrieve(query)

    logger.info(f"\n🔍 Query: '{query}'")
    if results:
        logger.info(f"✅ Retrieved {len(results)} chunks")
        for r in results:
            logger.info(f"   - {r.citation} (score: {r.similarity_score:.3f})")
    else:
        logger.info("❌ No results returned")

    return results


def test_table_retrieval(document_id: str):
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Table-Only Retrieval")
    logger.info("=" * 70)

    retriever = DocumentRetriever()
    query = "energy consumption"

    results = retriever.retrieve_tables_only(query)

    logger.info(f"\n🔍 Query: '{query}'")
    if results:
        logger.info(f"✅ Retrieved {len(results)} table chunks")
        for r in results:
            logger.info(f"   - {r.citation} (score: {r.similarity_score:.3f})")
    else:
        logger.info("⚠️ No table chunks found")


def test_page_filtered_retrieval(document_id: str):
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Page-Filtered Retrieval")
    logger.info("=" * 70)

    retriever = DocumentRetriever()
    query = "well depth"
    pages = [1, 2, 3]

    results = retriever.retrieve_from_pages(query, page_numbers=pages)

    logger.info(f"\n🔍 Query: '{query}' on pages {pages}")
    if results:
        logger.info(f"✅ Retrieved {len(results)} chunks")
        for r in results:
            logger.info(f"   - {r.citation} (score: {r.similarity_score:.3f})")
    else:
        logger.info("❌ No results found on specified pages")


def test_summarization_chunks(document_id: str):
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Summarization Chunk Retrieval")
    logger.info("=" * 70)

    retriever = DocumentRetriever()
    results = retriever.retrieve_for_summarization(document_id=document_id)

    if results:
        logger.info(f"✅ Retrieved {len(results)} chunks for summarization")
        logger.info(f"   First chunk: {results[0].filename}")
    else:
        logger.info("❌ No chunks retrieved")


def test_context_window(chunK_id: str):
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: Context Window")
    logger.info("=" * 70)

    retriever = DocumentRetriever()
    results = retriever.get_context_window(chunk_id=chunK_id)

    if not results:
        logger.info("❌ Cannot test context window — no chunks available")
        return

    target_chunk = results[len(results) // 2]
    window = retriever.get_context_window(target_chunk.chunk_id)

    logger.info(f"\n🧠 Context window for chunk: {target_chunk.chunk_id}")
    if window:
        logger.info(f"✅ Retrieved {len(window)} surrounding chunks")
        for r in window:
            logger.info(f"   - {r.filename}")
    else:
        logger.info("⚠️ No context window returned")


def test_llm_context_formatting(document_id: str):
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: LLM Context Formatting")
    logger.info("=" * 70)

    retriever = DocumentRetriever()
    results = retriever.retrieve_for_summarization(document_id=document_id)

    context = retriever.format_context_for_llm(results[:5], max_tokens=500)
    logger.info("\n🧠 Formatted Context:\n")
    logger.info(context[:1000] + "...\n")  # Truncate for display


def main():
    if len(sys.argv) < 2:
        logger.info("Usage: python scripts/test_retriever.py <pdf_file>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        logger.info(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    logger.info("\n" + "=" * 70)
    logger.info("📋 DOCUMENT RETRIEVER TEST SUITE")
    logger.info("=" * 70)

    document_id = ingest_test_document(pdf_path)

    if document_id:
        result = test_basic_retrieval(document_id)
        test_table_retrieval(document_id)
        test_page_filtered_retrieval(document_id)
        test_summarization_chunks(document_id)
        test_context_window(result[2].chunk_id)
        test_llm_context_formatting(document_id)

    logger.info("\n" + "=" * 70)
    logger.info("✅ RETRIEVER TESTS COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()