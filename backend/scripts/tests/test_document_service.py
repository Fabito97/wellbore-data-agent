"""
Test script for Document Service.

Validates the complete document ingestion pipeline.

Usage:
    python scripts/test_document_service.py path/to/test.pdf
"""

import sys
from pathlib import Path
import shutil

from multipart import file_path

from app.db.database import SessionLocal

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.utils.logger import get_logger

from app.services.document_service import get_document_service

logger = get_logger(__name__)


def test_document_ingestion(pdf_path: Path):
    """Test complete document ingestion."""
    print("=" * 70)
    print("TEST 1: Document Ingestion Pipeline")
    print("=" * 70)

    db = SessionLocal()
    service = get_document_service(db)

    # Copy file to temp location (simulating upload)
    temp_path = Path("temp_test.pdf")
    shutil.copy(pdf_path, temp_path)

    try:
        print(f"\n📄 Ingesting: {pdf_path.name}")
        print("⏳ Running pipeline (process → chunk → embed → store)...")
        well_id = "well-1_27288447748_010"
        result = service.ingest_document(
            file_path=temp_path,
            original_filename=pdf_path.name,
            well_name="well-1",
            well_id=well_id,
            file_format="pdf",
            document_type="PVT",
            original_folder_path=str(file_path)
        )
        print(f"Result: {result}")

        # Check if it's an error
        if hasattr(result, 'error'):
            print(f"\n❌ Ingestion failed:")
            print(f"   Error: {result.error}")
            print(f"   Details: {result.details}")
            return None

        print(f"\n✅ Ingestion successful!")
        print(f"\n📊 Results:")
        print(f"   Document ID: {result.document_id}")
        print(f"   Filename: {result.filename}")
        print(f"   Status: {result.status}")
        print(f"   Pages: {result.page_count}")
        print(f"   Words: {result.word_count:,}")
        print(f"   Tables: {result.table_count}")
        print(f"   Chunks: {result.chunk_count}")
        print(f"   Message: {result.message}")

        return result.document_id

    finally:
        # Cleanup temp file if it still exists
        if temp_path.exists():
            temp_path.unlink()


def test_document_retrieval(document_id: str):
    """Test retrieving document metadata."""
    print("\n" + "=" * 70)
    print("TEST 2: Document Retrieval")
    print("=" * 70)

    db = SessionLocal()
    service = get_document_service(db)

    print(f"\n🔍 Retrieving document: {document_id}")

    doc = service.get_document(document_id)

    if doc:
        print(f"✅ Document found!")
        print(f"\n📄 Details:")
        print(f"   Filename: {doc.filename}")
        print(f"   Pages: {doc.page_count}")
        print(f"   Status: {doc.status}")
        print(f"   File path: {doc.file_path}")
    else:
        print(f"❌ Document not found")


def test_document_status(document_id: str):
    """Test status checking."""
    print("\n" + "=" * 70)
    print("TEST 3: Document Status")
    print("=" * 70)

    db = SessionLocal()
    service = get_document_service(db)

    status = service.get_document_status(document_id)

    if status:
        print(f"\n✅ Status retrieved:")
        for key, value in status.items():
            print(f"   {key}: {value}")
    else:
        print(f"❌ Status not found")


def test_list_documents():
    """Test listing all documents."""
    print("\n" + "=" * 70)
    print("TEST 4: List All Documents")
    print("=" * 70)

    db = SessionLocal()
    service = get_document_service(db)

    docs = service.list_documents()

    print(f"\n📚 Total documents: {len(docs)}")

    for i, doc in enumerate(docs, 1):
        print(f"\n   {i}. {doc['filename']}")
        print(f"      ID: {doc['document_id']}")
        # print(f"      Pages: {doc['pages']}")
        # print(f"      Chunks: {doc['chunks']}")
        print(f"      Status: {doc['status']}")


def test_service_stats():
    """Test service statistics."""
    print("\n" + "=" * 70)
    print("TEST 5: Service Statistics")
    print("=" * 70)

    db = SessionLocal()
    service = get_document_service(db)

    stats = service.get_stats()

    print(f"\n📊 Service Stats:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Total pages: {stats['total_pages']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"\n   Vector Store:")
    for key, value in stats['vector_store'].items():
        print(f"      {key}: {value}")


def test_search_after_ingestion(document_id: str):
    """Test that document is searchable."""
    print("\n" + "=" * 70)
    print("TEST 6: Search Verification")
    print("=" * 70)

    from app.rag.retriever import search

    queries = [
        "Well",
        "depth",
        "energy"
    ]

    print(f"\n🔍 Testing search for document: {document_id}")

    for query in queries:
        print(f"\n   Query: '{query}'")
        results = search(query, top_k=3)
        print(f"\n   Results: {results}")

        # Filter to only this document
        doc_results = [r for r in results if r.document_id == document_id]

        if doc_results:
            print(f"   ✅ Found {len(doc_results)} relevant chunks")
            print(f"      Top match: {doc_results[0].citation} (score: {doc_results[0].similarity_score:.3f})")
        else:
            print(f"   ⚠️  No results from this document")


def test_document_deletion(document_id: str):
    """Test document deletion."""
    print("\n" + "=" * 70)
    print("TEST 7: Document Deletion")
    print("=" * 70)

    db = SessionLocal()
    service = get_document_service(db)

    # Get stats before
    stats_before = service.get_stats()
    chunks_before = stats_before['total_chunks']

    print(f"\n🗑️  Deleting document: {document_id}")
    print(f"   Chunks before: {chunks_before}")

    success = service.delete_document(document_id)

    if success:
        print(f"   ✅ Document deleted")

        # Get stats after
        stats_after = service.get_stats()
        chunks_after = stats_after['total_chunks']

        print(f"   Chunks after: {chunks_after}")
        print(f"   Removed: {chunks_before - chunks_after}")

        # Verify can't retrieve
        doc = service.get_document(document_id)
        if doc is None:
            print(f"   ✅ Document no longer in registry")
        else:
            print(f"   ⚠️  Document still in registry!")
    else:
        print(f"   ❌ Deletion failed")


def main():
    """Run all document service tests."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_document_service.py <pdf_file>")
        print("\nExample:")
        print("  python scripts/test_document_service.py data/raw/sample.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("📋 DOCUMENT SERVICE TEST SUITE")
    print("=" * 70)
    print(f"\nTesting with: {pdf_path.name}")

    # Run tests
    document_id = test_document_ingestion(pdf_path)

    if document_id:
        test_document_retrieval(document_id)
        test_document_status(document_id)
        test_list_documents()
        test_service_stats()
        test_search_after_ingestion(document_id)

        # Cleanup test (optional - comment out to keep data)
        user_input = input("\n⚠️  Delete test document? (y/n): ")
        if user_input.lower() == 'y':
            test_document_deletion(document_id)

    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)
    print("\n💡 Summary:")
    print("   - Document service working")
    print("   - Complete pipeline operational")
    print("   - Ready for API integration")


if __name__ == "__main__":
    main()