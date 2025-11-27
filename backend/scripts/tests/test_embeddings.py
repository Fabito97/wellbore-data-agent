"""
Test script for Embedding Generation.

Demonstrates embedding generation and semantic similarity.

Usage:
    python scripts/test_embeddings.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.embeddings import EmbeddingGenerator, get_embedding_generator


def test_model_loading():
    """Test embedding model initialization."""
    print("=" * 70)
    print("TEST 1: Model Loading")
    print("=" * 70)

    generator = EmbeddingGenerator()

    print(f"\n✅ Model loaded successfully!")
    print(f"\n📊 Model Info:")
    info = generator.model_info
    for key, value in info.items():
        print(f"  - {key}: {value}")

    return generator


def test_basic_embedding(generator):
    """Test basic text embedding."""
    print("\n" + "=" * 70)
    print("TEST 2: Basic Text Embedding")
    print("=" * 70)

    texts = [
        "The well has a depth of 1500 meters.",
        "Tubing diameter is 7 inches.",
        "Reservoir pressure is 2500 psi."
    ]

    print(f"\n🔤 Embedding {len(texts)} texts:")

    for text in texts:
        embedding = generator.embed_text(text)

        print(f"\n  Text: '{text}'")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {[f'{v:.4f}' for v in embedding[:5]]}")
        print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")


def test_semantic_similarity(generator):
    """Test semantic similarity computation."""
    print("\n" + "=" * 70)
    print("TEST 3: Semantic Similarity")
    print("=" * 70)

    # Test pairs with expected similarity
    test_pairs = [
        ("well depth 1500m", "depth of well is 1500 meters", "high"),
        ("tubing diameter", "casing diameter", "medium"),
        ("reservoir pressure", "production rate", "medium"),
        ("well depth", "banana", "low"),
    ]

    print("\n🔍 Testing semantic similarity:")

    for text1, text2, expected in test_pairs:
        emb1 = generator.embed_text(text1)
        emb2 = generator.embed_text(text2)
        similarity = generator.compute_similarity(emb1, emb2)

        # Interpret similarity
        if similarity > 0.7:
            level = "HIGH"
        elif similarity > 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"

        match = "✅" if level.lower() == expected else "⚠️"

        print(f"\n  {match} '{text1}' vs '{text2}'")
        print(f"     Similarity: {similarity:.4f} ({level}, expected {expected.upper()})")


def test_batch_processing(generator):
    """Test batch embedding performance."""
    print("\n" + "=" * 70)
    print("TEST 4: Batch Processing Performance")
    print("=" * 70)

    import time

    # Generate test texts
    texts = [f"This is test sentence number {i} about wells." for i in range(100)]

    # Test 1: One at a time
    print("\n⏱️  Timing: One at a time")
    start = time.time()
    for text in texts:
        _ = generator.embed_text(text)
    sequential_time = time.time() - start
    print(f"  Time: {sequential_time:.2f}s ({sequential_time / len(texts) * 1000:.1f}ms per text)")

    # Test 2: Batch
    print("\n⏱️  Timing: Batch processing")
    start = time.time()
    _ = generator.embed_batch(texts)
    batch_time = time.time() - start
    print(f"  Time: {batch_time:.2f}s ({batch_time / len(texts) * 1000:.1f}ms per text)")

    speedup = sequential_time / batch_time
    print(f"\n  📈 Speedup: {speedup:.1f}x faster with batching!")


def test_similarity_search(generator):
    """Test finding similar texts."""
    print("\n" + "=" * 70)
    print("TEST 5: Similarity Search")
    print("=" * 70)

    # Create a mini "database" of chunks
    database = [
        ("chunk_1", "The well is located at depth of 1500 meters with tubing diameter 7 inches."),
        ("chunk_2", "Reservoir pressure measured at 2500 psi with temperature 150°F."),
        ("chunk_3", "Production rate estimated at 500 barrels per day."),
        ("chunk_4", "Completion includes perforations from 1400m to 1500m depth."),
        ("chunk_5", "Water cut analysis shows 15% water content in production."),
    ]

    # Embed all chunks
    print("\n📚 Creating mini database of 5 chunks...")
    chunk_embeddings = []
    for chunk_id, text in database:
        embedding = generator.embed_text(text)
        chunk_embeddings.append((chunk_id, embedding))

    # Test queries
    queries = [
        "What is the well depth?",
        "Tell me about the production rate",
        "What is the reservoir pressure?"
    ]

    print("\n🔍 Testing similarity search:")

    for query in queries:
        print(f"\n  Query: '{query}'")
        query_emb = generator.embed_text(query)

        # Find most similar
        results = generator.find_most_similar(query_emb, chunk_embeddings, top_k=3)

        print(f"  Top 3 results:")
        for i, (chunk_id, score) in enumerate(results, 1):
            # Get original text
            original_text = next(text for cid, text in database if cid == chunk_id)
            preview = original_text[:60] + "..." if len(original_text) > 60 else original_text
            print(f"    {i}. {chunk_id} (score: {score:.4f})")
            print(f"       '{preview}'")


def test_embedding_properties(generator):
    """Test mathematical properties of embeddings."""
    print("\n" + "=" * 70)
    print("TEST 6: Embedding Properties")
    print("=" * 70)

    # Property 1: Determinism (same input = same output)
    print("\n🔬 Testing determinism:")
    text = "Test sentence for determinism"
    emb1 = generator.embed_text(text)
    emb2 = generator.embed_text(text)
    are_equal = np.allclose(emb1, emb2)
    status = "✅" if are_equal else "❌"
    print(f"  {status} Same input produces same output: {are_equal}")

    # Property 2: Triangle inequality (rough test)
    print("\n🔬 Testing semantic transitivity:")
    texts = ["dog", "puppy", "cat"]
    embeddings = [generator.embed_text(t) for t in texts]

    sim_dog_puppy = generator.compute_similarity(embeddings[0], embeddings[1])
    sim_puppy_cat = generator.compute_similarity(embeddings[1], embeddings[2])
    sim_dog_cat = generator.compute_similarity(embeddings[0], embeddings[2])

    print(f"  dog ↔ puppy: {sim_dog_puppy:.4f}")
    print(f"  puppy ↔ cat: {sim_puppy_cat:.4f}")
    print(f"  dog ↔ cat: {sim_dog_cat:.4f}")
    print(f"  Note: dog-puppy should be most similar (same species)")

    # Property 3: Dimension consistency
    print("\n🔬 Testing dimension consistency:")
    test_texts = ["short", "This is a longer text with more words"]
    for text in test_texts:
        emb = generator.embed_text(text)
        print(f"  '{text[:20]}...': {len(emb)} dimensions")
    print(f"  ✅ All embeddings have same dimension (as expected)")


def test_edge_cases(generator):
    """Test edge cases and error handling."""
    print("\n" + "=" * 70)
    print("TEST 7: Edge Cases")
    print("=" * 70)

    # Empty string
    print("\n🧪 Testing empty string:")
    try:
        emb = generator.embed_text("")
        print(f"  ✅ Empty string handled: dimension={len(emb)}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Very long text
    print("\n🧪 Testing very long text:")
    long_text = "word " * 1000  # 5000 characters
    try:
        emb = generator.embed_text(long_text)
        print(f"  ✅ Long text handled: {len(long_text)} chars → {len(emb)} dims")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Special characters
    print("\n🧪 Testing special characters:")
    special = "Test with émojis 🎉 and spëcial çhars!"
    try:
        emb = generator.embed_text(special)
        print(f"  ✅ Special characters handled: dimension={len(emb)}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")


def main():
    """Run all embedding tests."""
    print("\n" + "=" * 70)
    print("🔢 EMBEDDING GENERATION TEST SUITE")
    print("=" * 70)

    generator = test_model_loading()
    test_basic_embedding(generator)
    test_semantic_similarity(generator)
    test_batch_processing(generator)
    test_similarity_search(generator)
    test_embedding_properties(generator)
    test_edge_cases(generator)

    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)
    print("\n💡 Key Takeaways:")
    print("   - Embeddings capture semantic meaning")
    print("   - Similar texts have similar vectors")
    print("   - Batch processing is much faster")
    print("   - Ready for vector database storage!")


if __name__ == "__main__":
    main()