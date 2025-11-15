"""
Embeddings - Convert text chunks to vector representations (using sentence-transformers).
"""
from langchain_huggingface import HuggingFaceEmbeddings

from app.utils.logger import get_logger
import time
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.document import DocumentChunk
from app.core.config import settings

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generates vector embeddings for text chunks using huggingface model."""

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

        Default is all-MiniLM-L6-v2 (Size: ~80MB):
        - Dimensions: 384 (compact, fast) - Speed: ~10ms per sentence on CPU - Good for general semantic similarity
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        start_time = time.time()

        # Load model from HuggingFace - downloads model (~80MB) and caches it (~/.cache/huggingface/)
        self.model = get_embeddings_model()

        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")

        # # Get embedding dimension from model
        # self.dimension = getattr(self.model, "embedding_size", None)
        # logger.info(f"Embedding dimension: {self.dimension}")
        #
        # # Verify dimension matches config
        # if self.dimension != settings.EMBEDDING_DIMENSION:
        #     logger.warning(
        #         f"Model dimension ({self.dimension}) does not match config dimension "
        #         f"({settings.EMBEDDING_DIMENSION}). Update config!"
        #     )


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

        logger.info(f"Generating embeddings for {len(chunks)} chunks (batch_size={batch_size})")
        start_time = time.time()

        # Extract just the text content for embedding
        texts = [chunk.content for chunk in chunks]

        # Batch processing
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.debug(f"Processing {total_batches} number of batches")
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.embed_documents(batch)
            all_embeddings.extend(embeddings)

            if show_progress and (i // batch_size + 1) % 10 == 0:
                progress = (i + batch_size) / len(texts) * 100
                logger.info(f"Embedding progress: {progress:.0f}%")

        # Assign embeddings
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk.embedding = embedding

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
        return self.model.embed_query(text)


    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Batch processing for bulk operations - More efficient than calling embed_text() repeatedly.
        """
        return self.model.embed_documents(texts)


    def compute_similarity(
            self,
            embedding1: List[float],
            embedding2: List[float],
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        # Convert to numpy for computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity = dot product of normalized vectors
        # If vectors already normalized: similarity = dot product
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
            "dimension": self.model.dimension,
            "max_seq_length": self.model.max_seq_length,
        }



# ==================== Module-level convenience functions ====================
# Global instance (lazy loading)
_global_embeddings_model = None

def get_embeddings_model():
    """
    Get the shared HuggingFaceEmbeddings model.

    This ensures vector_store and embedding_generator
    use the SAME model instance (saves memory).
    """
    global _global_embeddings_model
    if _global_embeddings_model is None:
        _global_embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32
            }
        )
    return _global_embeddings_model

# Global instance (lazy loading) - reuse same instance after first call
_global_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get or create global embedding generator.
    """
    global _global_generator

    if _global_generator is None:
        _global_generator = EmbeddingGenerator()

    return _global_generator


def embed_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Convenience function for embedding chunks."""
    generator = get_embedding_generator()
    return generator.embed_chunks(chunks)


def embed_query(query: str) -> List[float]:
    """Convenience function for embedding a query string."""
    generator = get_embedding_generator()
    return generator.embed_text(query)
