"""
Test script for Document Processor.

This demonstrates how to use the document processor and validates it works.

Usage:
    python scripts/test_document_processor.py path/to/test.pdf
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.document_processor import DocumentProcessor, process_pdf
from app.models.document import DocumentStatus


def test_basic_processing(pdf_path: Path):
    """Test basic document processing."""
    print("=" * 70)
    print("TEST 1: Basic Document Processing")
    print("=" * 70)

    try:
        well_id = "well-1_27288447748_010"
        # Use convenience function
        doc = process_pdf(pdf_path, well_id=well_id, well_name="well-1", document_type="PVT")

        print(f"\n✅ Processing successful!")
        print(f"\n📄 Document Summary:")
        print(json.dumps(doc.summary, indent=2))

        print(f"\n📊 Statistics:")
        print(f"  - Pages: {doc.page_count}")
        print(f"  - Words: {doc.total_word_count:,}")
        print(f"  - Characters: {doc.total_char_count:,}")
        print(f"  - Tables: {doc.table_count}")
        print(f"  - Status: {doc.status}")
        print(f"  - Processing time: {doc.processing_time_seconds:.2f}s")

        # Show first page preview
        if doc.pages:
            first_page = doc.pages[0]
            preview = first_page.text[:200] + "..." if len(first_page.text) > 200 else first_page.text
            print(f"\n📝 First Page Preview:")
            print(f"  {preview}")

        return doc

    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        return None


def test_table_extraction(doc):
    """Test table extraction results."""
    print("\n" + "=" * 70)
    print("TEST 2: Table Extraction")
    print("=" * 70)

    if doc.table_count == 0:
        print("\n⚠️  No tables found in document")
        return

    print(f"\n✅ Found {doc.table_count} table(s)")

    for i, table in enumerate(doc.all_tables, start=1):
        print(f"\n📊 Table {i} (Page {table.page_number}):")
        print(f"  - Size: {table.row_count} rows × {table.column_count} columns")
        print(f"  - Headers: {', '.join(table.headers[:5])}")

        # Show table in markdown
        markdown = table.to_markdown()
        lines = markdown.split('\n')
        preview_lines = lines[:5]  # Show first 5 lines

        print(f"\n  Markdown Preview:")
        for line in preview_lines:
            print(f"    {line}")

        if len(lines) > 5:
            print(f"    ... ({len(lines) - 5} more rows)")


def test_page_details(doc):
    """Test page-level details."""
    print("\n" + "=" * 70)
    print("TEST 3: Page-Level Details")
    print("=" * 70)

    print(f"\n📄 Processing {doc.page_count} pages:")

    for page in doc.pages[:3]:  # Show first 3 pages
        print(f"\n  Page {page.page_number}:")
        print(f"    - Words: {page.word_count:,}")
        print(f"    - Characters: {page.char_count:,}")
        print(f"    - Tables: {len(page.tables)}")
        print(f"    - Has images: {'Yes' if page.has_images else 'No'}")

    if doc.page_count > 3:
        print(f"\n  ... and {doc.page_count - 3} more pages")

#
# def test_text_only_mode(pdf_path: Path):
#     """Test fast text-only extraction."""
#     print("\n" + "=" * 70)
#     print("TEST 4: Text-Only Mode (Fast)")
#     print("=" * 70)
#
#     import time
#
#     processor = DocumentProcessor(extract_tables=False)
#
#     start = time.time()
#
#     well_id = "well-1_27288447748_010"
#     doc = processor.process_document(pdf_path, well_id=well_id, well_name="well-1", document_type="PVT")
#     elapsed = time.time() - start
#
#     print(f"\n✅ Text-only extraction complete")
#     print(f"  - Time: {elapsed:.2f}s")
#     print(f"  - Words: {doc.total_word_count:,}")
#     print(f"  - Tables extracted: {doc.table_count} (should be 0)")


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 70)
    print("TEST 5: Error Handling")
    print("=" * 70)

    processor = DocumentProcessor()

    # Test 1: Non-existent file
    # Use convenience function
    well_id = "well-1_27288447748_010"
    try:
        processor.process_document(Path("nonexistent.pdf"), well_id=well_id, well_name="well-1", document_type="PVT")
        print("❌ Should have raised FileNotFoundError")
    except FileNotFoundError:
        print("✅ Correctly raised FileNotFoundError for missing file")

    # Test 2: Non-PDF file
    test_file = None
    try:
        test_file = Path("test.txt")
        test_file.touch()  # Create empty file
        processor.process_document(test_file, well_id=well_id, well_name="well-1", document_type="PVT")
        test_file.unlink()  # Clean up
        print("❌ Should have raised ValueError")
    except ValueError:
        print("✅ Correctly raised ValueError for non-PDF file")
        test_file.unlink()  # Clean up


def main():
    """Run all tests."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_document_processor.py <pdf_file>")
        print("\nExample:")
        print("  python scripts/test_document_processor.py data/raw/sample.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("📚 DOCUMENT PROCESSOR TEST SUITE")
    print("=" * 70)
    print(f"\nTesting with: {pdf_path.name}")

    # Run tests
    doc = test_basic_processing(pdf_path)

    if doc:
        test_table_extraction(doc)
        test_page_details(doc)
        # test_text_only_mode(pdf_path)

    test_error_handling()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()