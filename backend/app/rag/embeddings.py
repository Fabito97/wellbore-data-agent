"""
Embeddings - Convert text chunks to vector representations.

What are embeddings?
- Dense vector representations of text
- Capture semantic meaning in numbers
- Enable similarity search

Teaching Concepts:
- Embedding models (sentence-transformers)
- Batch processing for efficiency
- Vector normalization
- Dimensionality (384 for our model)

Why sentence-transformers?
- Optimized for semantic similarity
- Works on CPU (important for hackathon)
- Consistent with research standards
- Fast inference
"""

from app.utils.logger import get_logger
import time
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.document import DocumentChunk
from app.core.config import settings

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generates vector embeddings for text chunks.

    Design Pattern: Singleton-like (expensive to load model)
    - Model loaded once at initialization
    - Reused for all embedding operations
    - ~80MB model cached in memory

    Teaching: Why this approach?
    - Loading model is SLOW (~1-2 seconds)
    - Inference is FAST (~10ms per chunk)
    - Load once, use many times = efficient

    Memory Usage:
    - Model: ~80MB
    - Each embedding: 384 floats × 4 bytes = ~1.5KB
    - 1000 chunks = ~1.5MB of embeddings
    """

    def __init__(
            self,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
    ):
        """
        Initialize embedding generator with model.

        Args:
            model_name: HuggingFace model name (default from settings)
            device: 'cpu' or 'cuda' (default from settings)

        Teaching: Model Selection

        all-MiniLM-L6-v2:
        - Dimensions: 384 (compact, fast)
        - Speed: ~10ms per sentence on CPU
        - Size: ~80MB
        - Quality: Good for general semantic similarity

        Alternatives (if we had GPU):
        - all-mpnet-base-v2: 768 dims, better quality, slower
        - e5-large: 1024 dims, SOTA quality, much slower

        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        start_time = time.time()

        # Load model from HuggingFace
        # First run downloads model (~80MB)
        # Subsequent runs load from cache (~/.cache/huggingface/)
        self.model = SentenceTransformer(self.model_name, device=self.device)

        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")

        # Get embedding dimension from model
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

        # Verify dimension matches config
        if self.dimension != settings.EMBEDDING_DIMENSION:
            logger.warning(
                f"Model dimension ({self.dimension}) does not match config dimension "
                f"({settings.EMBEDDINGS_DIMENSION}). Update config!"
            )


    def embed_chunks(
            self,
            chunks: List[DocumentChunk],
            batch_size: int = 32,
            show_progress: bool = True,
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for a list of chunks - Batch Processing

        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of chunks to process at once
            show_progress: Log progress every N batches

        Returns:
            Same chunks with .embedding field populated
        """
        if not chunks:
            logger.warning("No chunks to embed!")
            return chunks

        # Extract just the text content for embedding
        logger.info(f"Generating embeddings for {len(chunks)} chunks (batch_size={batch_size})")
        start_time = time.time()

        texts = [chunk.content for chunk in chunks]

        # Batch processing
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]

            # Generate embeddings for batch
            embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True, # Return numpy arrays
                show_progress_bar=False # We handle progress logging
            )

            # Assign embeddings back to chunks
            for chunk, embedding in zip(batch_chunks, embeddings):
                # Convert numpy array to list for Pydantic model
                chunk.embedding = embedding.tolist()

            # Progress loging
            if show_progress and (i // batch_size + 1) % 10 == 0:
                progress = (i + batch_size) / len(texts) * 100
                logger.debug(f"Embedding progress: {progress:.0f}%")

        elapsed = time.time() - start_time
        avg_time = (elapsed / len(chunks)) * 1000

        logger.info(
            f"Embedding complete: {len(chunks)} chunks in {elapsed:.2f}s "
            f"({avg_time:.2f}ms per chunk)"
        )

        return chunks


    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.

        Returns:
            List of floats (embedding vector)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        More efficient than calling embed_text() repeatedly.

        Use case:
        - Multiple queries at once
        - Batch processing for bulk operations
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]


    def compute_similarity(
            self,
            embedding1: List[float],
            embedding2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Teaching: Cosine Similarity

        Formula: similarity = (A · B) / (||A|| × ||B||)

        Range: -1 to 1
        - 1.0: Identical meaning
        - 0.0: Unrelated
        - -1.0: Opposite meaning (rare in practice)

        Why cosine instead of euclidean distance?
        - Insensitive to vector magnitude
        - Focuses on direction (meaning)
        - Standard in NLP

        Example:
        ```python
        emb1 = embed("well depth 1500m")
        emb2 = embed("depth of well is 1500 meters")
        similarity = compute_similarity(emb1, emb2)
        # → ~0.95 (very similar!)
        ```
        """
        # Convert to numpy for computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity = dot product of normalized vectors
        # If vectors already normalized: similarity = do product
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)


    def find_most_similar(
            self,
            query_embedding: List[float],
            chunk_embeddings: List[Tuple[str, List[float]]],
            top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
         Find most similar chunks to a query.

         Args:
             query_embedding: Query vector
             chunk_embeddings: List of (chunk_id, embedding) tuples
             top_k: Number of results to return

         Returns:
             List of (chunk_id, similarity_score) sorted by relevance

         Teaching: Brute Force Search

         Algorithm:
         1. Compare query to every chunk (O(n))
         2. Compute similarity for each
         3. Sort by similarity
         4. Return top K

         Why not use this for production?
         - Slow for large collections (thousands of chunks)
         - Better: Use vector database (ChromaDB, Pinecone, etc.)
         - They use approximate nearest neighbor (ANN) algorithms
         - Much faster: O(log n) instead of O(n)

         But useful for:
         - Testing
         - Small collections
         - Understanding the concept
         """
        similarities = []

        for chunk_id, chunk_embedding in chunk_embeddings:
            similarity = self.compute_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk_id, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top K
        return similarities[:top_k]

    @property
    def model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length,
        }



# ==================== Module-level convenience functions ====================

# Global instance (lazy loading)
# Teaching: Module-level singleton pattern
# - Avoids loading model multiple times
# - First call loads model
# - Subsequent calls reuse same instance
_global_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get or create global embedding generator.

    Teaching: Lazy Initialization Pattern
    - Don't load model until first use
    - Reuse same instance across calls
    - Saves memory and startup time

    Usage:
        generator = get_embedding_generator()
        embedding = generator.embed_text("hello world")
    """
    global _global_generator

    if _global_generator is None:
        _global_generator = EmbeddingGenerator()

    return _global_generator


def embed_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """
    Convenience function for embedding chunks.

    Teaching: Facade pattern
    - Hides complexity of generator initialization
    - Simple API for common use case

    Usage:
        chunks = chunk_document(doc)
        chunks = embed_chunks(chunks)  # That's it!
    """
    generator = get_embedding_generator()
    return generator.embed_chunks(chunks)


def embed_query(query: str) -> List[float]:
    """
    Convenience function for embedding a query string.

    Use case: Converting user questions to vectors

    Usage:
        query = "What is the well depth?"
        query_vector = embed_query(query)
        # Use query_vector to search vector database
    """
    generator = get_embedding_generator()
    return generator.embed_text(query)
