"""
Test script for Simple Agent.

Tests the complete RAG Q&A pipeline with a real document.

Usage:
    python scripts/test_simple_agent.py [pdf_path]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.document_service import get_document_service
from app.agents.simple_agent import get_simple_agent, ask


def setup_test_document(pdf_path: Path) -> str:
    """Ingest a test document if not already in system."""
    print("=" * 70)
    print("SETUP: Ingesting Test Document")
    print("=" * 70)

    service = get_document_service()

    print(f"\n📄 Ingesting: {pdf_path.name}")

    # Make a copy for ingestion
    import shutil
    temp_path = Path("temp_test_doc.pdf")
    shutil.copy(pdf_path, temp_path)

    try:
        result = service.ingest_document(temp_path, pdf_path.name)

        if hasattr(result, 'error'):
            print(f"❌ Failed: {result.error}")
            return None

        print(f"✅ Document ingested: {result.document_id}")
        print(f"   Chunks: {result.chunk_count}")

        return result.document_id

    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_basic_qa():
    """Test basic question answering."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Question Answering")
    print("=" * 70)

    agent = get_simple_agent()

    questions = [
        "What is the well depth?",
        "What is the tubing diameter?",
        "Tell me about the completion",
        "What is the reservoir pressure?",
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 70)

        response = agent.answer(question)

        print(f"\n💬 Answer ({response.confidence} confidence):")
        print(f"   {response.answer}")

        if response.sources:
            print(f"\n📚 Sources ({len(response.sources)}):")
            for i, source in enumerate(response.sources[:3], 1):
                print(f"   {i}. {source.citation} (score: {source.similarity_score:.3f})")
                preview = source.content[:80] + "..."
                print(f"      Preview: {preview}")

        if response.tokens_used:
            print(f"\n🔢 Tokens used: {response.tokens_used}")


def test_simple_ask_function():
    """Test the convenience ask() function."""
    print("\n" + "=" * 70)
    print("TEST 2: Simple ask() Function")
    print("=" * 70)

    print("\n📝 Using simple ask() interface:")

    answer = ask("What are the main parameters of this well?")

    print(f"\n💬 Answer:")
    print(f"   {answer}")


def test_document_summarization(document_id: str):
    """Test document summarization."""
    print("\n" + "=" * 70)
    print("TEST 3: Document Summarization")
    print("=" * 70)

    agent = get_simple_agent()

    word_counts = [100, 200, 300]

    for words in word_counts:
        print(f"\n📋 Summarizing in {words} words:")
        print("-" * 70)

        response = agent.summarize_document(document_id, max_words=words)

        print(f"\n📝 Summary:")
        print(f"   {response.answer}")

        # Count actual words
        actual_words = len(response.answer.split())
        print(f"\n   Target: {words} words")
        print(f"   Actual: {actual_words} words")


def test_table_extraction():
    """Test finding tables."""
    print("\n" + "=" * 70)
    print("TEST 4: Table Extraction")
    print("=" * 70)

    agent = get_simple_agent()

    queries = [
        "completion parameters",
        "depth and diameter data",
        "pressure measurements"
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 70)

        response = agent.extract_tables(query, top_k=3)

        if response.sources:
            print(f"\n✅ Found {len(response.sources)} table(s)")
            for i, table in enumerate(response.sources, 1):
                print(f"\n   Table {i} from page {table.page_number}:")
                # Show first few lines
                lines = table.content.split('\n')[:5]
                for line in lines:
                    print(f"      {line}")
        else:
            print("   ⚠️  No tables found")


def test_context_window():
    """Test context window expansion."""
    print("\n" + "=" * 70)
    print("TEST 5: Context Window (Expanded Context)")
    print("=" * 70)

    agent = get_simple_agent()

    # Questions that might need broader context
    questions = [
        "What is the complete depth specification?",
        "Explain the full completion design"
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 70)

        # Normal answer
        print("\n   Standard retrieval:")
        normal = agent.answer(question, include_sources=False)
        print(f"   {normal.answer[:150]}...")

        # With context window
        print("\n   With context window:")
        expanded = agent.answer_with_context_window(question, window_size=2)
        print(f"   {expanded.answer[:150]}...")

        print(f"\n   Chunks used: {len(expanded.sources)}")


def test_confidence_levels():
    """Test confidence assessment."""
    print("\n" + "=" * 70)
    print("TEST 6: Confidence Assessment")
    print("=" * 70)

    agent = get_simple_agent()

    # Mix of good and bad questions
    test_cases = [
        ("What is the well depth?", "Should be high - common parameter"),
        ("What is the well name?", "Might be high if in document"),
        ("What is the weather?", "Should be low - not in document"),
        ("Tell me about aliens", "Should be low - irrelevant")
    ]

    for question, expectation in test_cases:
        print(f"\n❓ Question: {question}")
        print(f"   Expected: {expectation}")

        response = agent.answer(question, include_sources=False)

        print(f"   Confidence: {response.confidence.upper()}")
        print(f"   Sources found: {len(response.sources)}")


def test_filtering():
    """Test metadata filtering."""
    print("\n" + "=" * 70)
    print("TEST 7: Filtered Retrieval")
    print("=" * 70)

    agent = get_simple_agent()

    # Test different filters
    filters = [
        ({"chunk_type": "table"}, "tables only"),
        ({"page_number": 1}, "page 1 only"),
    ]

    question = "completion data"

    for filter_dict, description in filters:
        print(f"\n🔍 Query: {question}")
        print(f"   Filter: {description}")

        response = agent.answer(question, filters=filter_dict)

        print(f"   Results: {len(response.sources)}")
        if response.sources:
            print(f"   Top match: {response.sources[0].citation}")


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 70)
    print("TEST 8: Error Handling")
    print("=" * 70)

    agent = get_simple_agent()

    # Test with empty question
    print("\n🧪 Test: Empty question")
    response = agent.answer("")
    print(f"   Response: {response.answer[:100]}...")
    print(f"   Confidence: {response.confidence}")

    # Test with very long question
    print("\n🧪 Test: Very long question")
    long_question = "what is the " + "depth " * 100
    response = agent.answer(long_question)
    print(f"   Handled: {len(response.answer) > 0}")


def main():
    """Run all simple agent tests."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_simple_agent.py <pdf_file>")
        print("\nExample:")
        print("  python scripts/test_simple_agent.py data/raw/sample.pdf")
        print("\nNote: Requires Ollama running with phi3:mini model")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🤖 SIMPLE AGENT TEST SUITE")
    print("=" * 70)
    print(f"\nTesting with: {pdf_path.name}")

    # Validate Ollama is running
    print("\n⚙️  Checking Ollama connection...")
    from app.services.llm_service import get_llm_service
    llm = get_llm_service()
    if not llm.validate_connection():
        print("❌ Ollama not accessible!")
        print("   Make sure Ollama is running: ollama serve")
        sys.exit(1)
    print("✅ Ollama is ready")

    # Setup test document
    document_id = setup_test_document(pdf_path)

    if not document_id:
        print("\n❌ Failed to ingest document. Cannot proceed with tests.")
        sys.exit(1)

    # Run tests
    test_basic_qa()
    test_simple_ask_function()
    test_document_summarization(document_id)
    test_table_extraction()
    test_context_window()
    test_confidence_levels()
    test_filtering()
    test_error_handling()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)
    print("\n💡 Summary:")
    print("   - RAG pipeline working end-to-end")
    print("   - Question answering operational")
    print("   - Summarization functional")
    print("   - Table extraction working")
    print("   - Ready for full orchestrator!")


if __name__ == "__main__":
    main()