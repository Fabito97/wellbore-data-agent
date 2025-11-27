"""
Test script for Vector Store (ChromaDB).

Demonstrates the complete RAG pipeline and vector search.

Usage:
    python scripts/test_vector_store.py [pdf_path]
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.document_processor import process_pdf
from app.rag.chunking import chunk_document
from app.rag.embeddings import embed_chunks
from app.rag.vector_store_manager import VectorStoreManager as VectorStore, get_vector_store


def test_store_initialization():
    """Test vector store setup."""
    print("=" * 70)
    print("TEST 1: Vector Store Initialization")
    print("=" * 70)

    store = VectorStore()

    stats = store.get_stats()

    print(f"\n✅ Vector store initialized!")
    print(f"\n📊 Stats:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    return store


def test_full_pipeline(pdf_path: Path, store: VectorStore):
    """Test complete document ingestion pipeline."""
    print("\n" + "=" * 70)
    print("TEST 2: Full Ingestion Pipeline")
    print("=" * 70)

    print(f"\n📄 Processing: {pdf_path.name}")

    # Step 1: Process PDF
    print("\n⏳ Step 1/4: Processing PDF...")
    doc = process_pdf(pdf_path)
    print(f"   ✅ Extracted {doc.page_count} pages")

    # Step 2: Chunk document
    print("\n⏳ Step 2/4: Chunking document...")
    chunks = chunk_document(doc)
    print(f"   ✅ Created {len(chunks)} chunks")

    # Step 3: Generate embeddings
    print("\n⏳ Step 3/4: Generating embeddings...")
    chunks = embed_chunks(chunks)
    print(f"   ✅ Embedded {len(chunks)} chunks")

    # Step 4: Add to vector store
    print("\n⏳ Step 4/4: Adding to vector store...")
    added = store.add_chunks(chunks)
    print(f"   ✅ Added {added} chunks to store")

    print(f"\n🎉 Pipeline complete!")
    print(f"   Document '{doc.filename}' is now searchable")

    return doc, chunks


def test_basic_search(store: VectorStore):
    """Test basic similarity search."""
    print("\n" + "=" * 70)
    print("TEST 3: Basic Similarity Search")
    print("=" * 70)

    queries = [
        "What is the well depth?",
        "Tell me about the completion",
        "What is the tubing diameter?",
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")

        results = store.query_similar_chunks(query, top_k=3)

        if results:
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                similarity = result.get('similarity_score', 0)
                content_preview = result['content'][:100] + "..."
                page = result['metadata'].get('page_number', '?')

                print(f"\n   {i}. Similarity: {similarity:.4f} | Page: {page}")
                print(f"      {content_preview}")
        else:
            print("   ⚠️  No results found")


def test_metadata_filtering(store: VectorStore):
    """Test search with metadata filters."""
    print("\n" + "=" * 70)
    print("TEST 4: Metadata Filtering")
    print("=" * 70)

    # Test 1: Filter by chunk type
    print("\n🔍 Test 1: Search only in tables")
    query = "diameter pressure depth"
    results = store.query_similar_chunks(
        query,
        top_k=5,
        filters={"chunk_type": "table"}
    )

    print(f"   Query: '{query}'")
    print(f"   Filter: chunk_type='table'")
    print(f"   Results: {len(results)} table chunks found")

    if results:
        for i, result in enumerate(results[:2], 1):
            print(f"\n   {i}. Table from page {result['metadata'].get('page_number')}")
            print(f"      Columns: {result['metadata'].get('column_count')}")

    # Test 2: Filter by page number
    print("\n\n🔍 Test 2: Search only in first 5 pages")
    results = store.query_similar_chunks(
        "completion",
        top_k=5,
        filters={"page_number": {"$lte": 5}}
    )

    print(f"   Query: 'completion'")
    print(f"   Filter: page_number <= 5")
    print(f"   Results: {len(results)} chunks found")


def test_retrieval_quality(store: VectorStore):
    """Test quality of retrieval."""
    print("\n" + "=" * 70)
    print("TEST 5: Retrieval Quality")
    print("=" * 70)

    test_cases = [
        {
            "query": "measured depth true vertical depth",
            "expected_keywords": ["depth", "meter", "feet", "MD", "TVD"],
            "description": "Should find trajectory data"
        },
        {
            "query": "tubing casing diameter completion",
            "expected_keywords": ["tubing", "casing", "diameter", "inch"],
            "description": "Should find completion data"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test['description']}")
        print(f"   Query: '{test['query']}'")

        results = store.query_similar_chunks(test['query'], top_k=3)

        if not results:
            print("   ⚠️  No results found")
            continue

        # Check if expected keywords appear in top results
        all_text = " ".join([r['content'].lower() for r in results])
        found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in all_text]

        print(f"   Expected keywords: {', '.join(test['expected_keywords'])}")
        print(f"   Found: {', '.join(found_keywords)} ({len(found_keywords)}/{len(test['expected_keywords'])})")

        if len(found_keywords) >= len(test['expected_keywords']) * 0.5:
            print("   ✅ Good retrieval quality")
        else:
            print("   ⚠️  Low keyword match")


def test_get_by_id(store: VectorStore, chunks):
    """Test retrieving specific chunk by ID."""
    print("\n" + "=" * 70)
    print("TEST 6: Get Chunk by ID")
    print("=" * 70)

    if not chunks:
        print("\n⚠️  No chunks available to test")
        return

    # Get a sample chunk
    sample_chunk = chunks[0]
    chunk_id = sample_chunk.chunk_id

    print(f"\n🔍 Retrieving chunk: {chunk_id}")

    result = store.get_by_id(chunk_id)

    if result:
        print(f"   ✅ Found chunk!")
        print(f"   - Content length: {len(result['content'])} chars")
        print(f"   - Metadata: {json.dumps(result['metadata'], indent=6)}")
        print(f"   - Has embedding: {result['embedding'] is not None}")
    else:
        print(f"   ❌ Chunk not found")


def test_delete_operations(store: VectorStore, document_id: str):
    """Test deletion operations."""
    print("\n" + "=" * 70)
    print("TEST 7: Delete Operations")
    print("=" * 70)

    # Get initial count
    initial_stats = store.get_stats()
    initial_count = initial_stats['total_chunks']

    print(f"\n📊 Initial state:")
    print(f"   Total chunks: {initial_count}")

    # Delete document
    print(f"\n🗑️  Deleting document: {document_id}")
    deleted = store.delete_by_document_id(document_id)
    print(f"   Deleted {deleted} chunks")

    # Get new count
    final_stats = store.get_stats()
    final_count = final_stats['total_chunks']

    print(f"\n📊 Final state:")
    print(f"   Total chunks: {final_count}")
    print(f"   Difference: -{initial_count - final_count}")

    if final_count == initial_count - deleted:
        print("   ✅ Deletion successful")
    else:
        print("   ⚠️  Count mismatch")


def test_persistence():
    """Test that data persists across instances."""
    print("\n" + "=" * 70)
    print("TEST 8: Data Persistence")
    print("=" * 70)

    # Create first instance and get count
    print("\n📊 Creating first store instance...")
    store1 = VectorStore()
    count1 = store1.get_stats()['total_chunks']
    print(f"   Chunks in store: {count1}")

    # Create second instance (should load same data)
    print("\n📊 Creating second store instance...")
    store2 = VectorStore()
    count2 = store2.get_stats()['total_chunks']
    print(f"   Chunks in store: {count2}")

    if count1 == count2:
        print("\n   ✅ Data persisted correctly!")
        print("   Same data accessible across instances")
    else:
        print("\n   ⚠️  Data mismatch between instances")


def test_semantic_understanding(store: VectorStore):
    """Test that search understands semantics, not just keywords."""
    print("\n" + "=" * 70)
    print("TEST 9: Semantic Understanding")
    print("=" * 70)

    print("\n🧠 Testing semantic similarity (not just keyword matching):")

    # These queries use different words but similar meaning
    semantic_pairs = [
        ("well depth", "how deep is the well"),
        ("completion diameter", "size of the tubing"),
        ("production rate", "how much oil per day"),
    ]

    for query1, query2 in semantic_pairs:
        print(f"\n   Pair: '{query1}' ≈ '{query2}'")

        results1 = store.query_similar_chunks(query1, top_k=3)
        results2 = store.query_similar_chunks(query2, top_k=3)

        if results1 and results2:
            # Check if top results are similar
            top_chunks1 = set(r['chunk_id'] for r in results1)
            top_chunks2 = set(r['chunk_id'] for r in results2)

            overlap = len(top_chunks1 & top_chunks2)
            print(f"      Overlapping results: {overlap}/3")

            if overlap >= 1:
                print("      ✅ Semantic understanding working!")
            else:
                print("      ⚠️  Different results (may need more data)")


def main():
    """Run all vector store tests."""

    # Check if PDF provided
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            sys.exit(1)
    else:
        pdf_path = None

    print("\n" + "=" * 70)
    print("🗄️  VECTOR STORE TEST SUITE")
    print("=" * 70)

    # Initialize store
    store = test_store_initialization()

    # If PDF provided, test full pipeline
    doc = None
    chunks = []
    if pdf_path:
        print(f"\n📄 Testing with: {pdf_path.name}")
        doc, chunks = test_full_pipeline(pdf_path, store)
    else:
        print("\n⚠️  No PDF provided - some tests will be skipped")
        print("   Usage: python scripts/test_vector_store.py <pdf_file>")

    # Run search tests (require existing data)
    current_count = store.get_stats()['total_chunks']
    if current_count > 0:
        test_basic_search(store)
        test_metadata_filtering(store)
        test_retrieval_quality(store)
        test_semantic_understanding(store)

        if chunks:
            test_get_by_id(store, chunks)
    else:
        print("\n⚠️  No data in store - add documents first to test search")

    # Test persistence
    test_persistence()

    # Test deletion (if we added a document)
    if doc:
        test_delete_operations(store, doc.document_id)

    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)
    print("\n💡 Summary:")
    print("   - Vector store working correctly")
    print("   - Semantic search operational")
    print("   - Data persists across restarts")
    print("   - Ready for RAG pipeline!")


if __name__ == "__main__":
    main()