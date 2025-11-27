"""
Test script for Document Chunking.

Demonstrates chunking strategy and validates chunk quality.

Usage:
    python scripts/test_chunking.py path/to/test.pdf
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.document_processor import process_pdf
from app.rag.chunking import DocumentChunker, ChunkingConfig, chunk_document


def test_basic_chunking(document):
    """Test basic chunking with default settings."""
    print("=" * 70)
    print("TEST 1: Basic Chunking (Default Settings)")
    print("=" * 70)

    chunks = chunk_document(document)

    print(f"\n✅ Chunking complete!")
    print(f"\n📊 Statistics:")
    print(f"  - Total chunks: {len(chunks)}")
    print(f"  - Document pages: {document.page_count}")
    print(f"  - Chunks per page: {len(chunks) / document.page_count:.1f}")

    # Analyze chunk types
    text_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "text"]
    table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]

    print(f"\n📝 Chunk Types:")
    print(f"  - Text chunks: {len(text_chunks)}")
    print(f"  - Table chunks: {len(table_chunks)}")

    # Chunk size distribution
    sizes = [len(c.content) for c in chunks]
    avg_size = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)

    print(f"\n📏 Chunk Sizes (characters):")
    print(f"  - Average: {avg_size:.0f}")
    print(f"  - Minimum: {min_size}")
    print(f"  - Maximum: {max_size}")

    return chunks


def test_chunk_content(chunks):
    """Display sample chunks."""
    print("\n" + "=" * 70)
    print("TEST 2: Chunk Content Preview")
    print("=" * 70)

    # Show first 3 chunks
    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n📄 Chunk {i} (ID: {chunk.chunk_id}):")
        print(f"  - Page: {chunk.page_number}")
        print(f"  - Type: {chunk.metadata.get('chunk_type')}")
        print(f"  - Size: {chunk.char_count} chars, {chunk.word_count} words")
        print(f"\n  Content preview:")
        preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        for line in preview.split('\n')[:5]:  # First 5 lines
            print(f"    {line}")


def test_table_chunks(chunks):
    """Examine table chunks specifically."""
    print("\n" + "=" * 70)
    print("TEST 3: Table Chunks")
    print("=" * 70)

    table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]

    if not table_chunks:
        print("\n⚠️  No table chunks found")
        return

    print(f"\n✅ Found {len(table_chunks)} table chunk(s)")

    for i, chunk in enumerate(table_chunks[:2], start=1):  # Show first 2 tables
        print(f"\n📊 Table Chunk {i}:")
        print(f"  - ID: {chunk.chunk_id}")
        print(f"  - Page: {chunk.page_number}")
        print(f"  - Columns: {chunk.metadata.get('column_count')}")
        print(f"  - Rows: {chunk.metadata.get('row_count')}")
        print(f"  - Headers: {', '.join(chunk.metadata.get('headers', []))}")
        print(f"\n  Markdown content:")
        lines = chunk.content.split('\n')
        for line in lines[:10]:  # First 10 lines
            print(f"    {line}")


def test_chunk_overlap(chunks):
    """Verify overlap between consecutive chunks."""
    print("\n" + "=" * 70)
    print("TEST 4: Chunk Overlap Verification")
    print("=" * 70)

    # Get text chunks from same page
    text_chunks = [
        c for c in chunks
        if c.metadata.get("chunk_type") == "text" and c.page_number == 1
    ]

    if len(text_chunks) < 2:
        print("\n⚠️  Not enough text chunks on page 1 to check overlap")
        return

    print(f"\n🔍 Checking overlap between consecutive chunks on page 1:")

    for i in range(len(text_chunks) - 1):
        chunk1 = text_chunks[i]
        chunk2 = text_chunks[i + 1]

        # Find overlap: end of chunk1 should appear in start of chunk2
        # Check last 200 chars of chunk1 against first 200 of chunk2
        end_of_chunk1 = chunk1.content[-200:]
        start_of_chunk2 = chunk2.content[:200]

        # Simple overlap detection
        has_overlap = any(
            end_of_chunk1[j:j + 50] in start_of_chunk2
            for j in range(0, len(end_of_chunk1) - 50, 10)
        )

        status = "✅" if has_overlap else "⚠️"
        print(f"\n  {status} Chunk {i} → Chunk {i + 1}:")
        print(f"      Chunk {i} ends: '...{end_of_chunk1[-50:]}'")
        print(f"      Chunk {i + 1} starts: '{start_of_chunk2[:50]}...'")


def test_custom_chunk_size():
    """Test chunking with different sizes."""
    print("\n" + "=" * 70)
    print("TEST 5: Custom Chunk Sizes")
    print("=" * 70)

    sample_text = "This is a test. " * 100  # 1600 characters

    configs = [
        (500, 100),  # Small chunks
        (1000, 200),  # Default
        (2000, 400),  # Large chunks
    ]

    print(f"\n📝 Sample text: {len(sample_text)} characters")

    for chunk_size, overlap in configs:
        config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=overlap)
        chunker = DocumentChunker(config)

        pieces = chunker.text_splitter.split_text(sample_text)

        print(f"\n  Chunk size={chunk_size}, overlap={overlap}:")
        print(f"    - Produced: {len(pieces)} chunks")
        print(f"    - Avg size: {sum(len(p) for p in pieces) / len(pieces):.0f} chars")


def test_metadata_richness(chunks):
    """Verify metadata completeness."""
    print("\n" + "=" * 70)
    print("TEST 6: Metadata Completeness")
    print("=" * 70)

    print(f"\n🔍 Checking metadata for {len(chunks)} chunks...")

    required_fields = ["filename", "page_number", "chunk_type"]

    missing_metadata = []
    for chunk in chunks:
        for field in required_fields:
            if field not in chunk.metadata:
                missing_metadata.append((chunk.chunk_id, field))

    if missing_metadata:
        print(f"\n⚠️  Found {len(missing_metadata)} missing metadata fields:")
        for chunk_id, field in missing_metadata[:5]:
            print(f"    - Chunk {chunk_id}: missing '{field}'")
    else:
        print("\n✅ All chunks have complete metadata")

    # Show example metadata
    if chunks:
        print(f"\n📋 Example metadata (first chunk):")
        print(json.dumps(chunks[0].metadata, indent=2))


def main():
    """Run all chunking tests."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_chunking.py <pdf_file>")
        print("\nExample:")
        print("  python scripts/test_chunking.py data/raw/sample.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("📚 DOCUMENT CHUNKING TEST SUITE")
    print("=" * 70)
    print(f"\nTesting with: {pdf_path.name}")

    # First, process the document
    print("\n⏳ Processing document...")
    document = process_pdf(pdf_path)
    print(f"✅ Document processed: {document.page_count} pages")

    # Run tests
    chunks = test_basic_chunking(document)
    test_chunk_content(chunks)
    test_table_chunks(chunks)
    test_chunk_overlap(chunks)
    test_custom_chunk_size()
    test_metadata_richness(chunks)

    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)
    print(f"\n💡 Summary:")
    print(f"   - Document: {document.filename}")
    print(f"   - Pages: {document.page_count}")
    print(f"   - Chunks: {len(chunks)}")
    print(f"   - Ready for embedding!")


if __name__ == "__main__":
    main()