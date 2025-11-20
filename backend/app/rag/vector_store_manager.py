"""
Enhanced Vector Store - Best of both worlds.
"""
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.models.document import DocumentChunk
from app.utils.helper import normalize_score
from app.utils.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class VectorStoreManager:
    """
    Enhanced vector store with LangChain + custom features.

    Improvements over basic LangChain:
    - Document-level operations
    - Metadata filtering
    - Statistics tracking
    - Chunk ID management
    """

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or str(settings.VECTOR_DB_DIR)

        logger.info(f"Initializing Chroma at: {self.persist_directory}")

        # Reuse embedding model
        from app.rag.embeddings import get_embeddings_model
        self.embeddings = get_embeddings_model()

        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        self.collection = self.vector_store._collection

        logger.info("Vector store ready")

    # -------------------- Enhanced Add Method --------------------

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add chunks with proper ID management.
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0

        # Validate chunks have IDs
        chunks_with_ids = [c for c in chunks if c.chunk_id]
        if len(chunks_with_ids) < len(chunks):
            logger.warning(f"{len(chunks) - len(chunks_with_ids)} chunks missing IDs")

        logger.info(f"Adding {len(chunks_with_ids)} chunks to vector store")

        # Extract data
        ids = [chunk.chunk_id for chunk in chunks_with_ids]
        texts = [chunk.content for chunk in chunks_with_ids]
        metadatas = [chunk.metadata for chunk in chunks_with_ids]

        # ✅ Add with explicit IDs (key improvement!)
        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids  # This ensures we control IDs
        )

        logger.info(f"Successfully added {len(chunks_with_ids)} chunks")
        return len(chunks_with_ids)

    # -------------------- Enhanced Query Methods --------------------

    def query_similar_chunks(
            self,
            query: str,
            top_k: int = 5,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query with metadata filtering support.

        Improvements:
        - Accepts filter dict
        - Returns similarity scores
        - Better result format
        """
        logger.info(f"Querying top {top_k} chunks for: '{query[:50]}...'")

        # Query with filters
        if filters:
            results = self.vector_store.similarity_search_with_score(
                query, k=top_k, filter=filters  # LangChain supports this!
            )
        else:
            results = self.vector_store.similarity_search_with_score(
                query, k=top_k
            )

        # Format results
        formatted = []
        for doc, distance in results:
            formatted.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": 1 - distance # normalize_score(distance),  # Convert distance to similarity
                "chunk_id": doc.metadata.get("chunk_id", "")
            })

        logger.info(f"Found {len(formatted)} results")
        return formatted

    # -------------------- Document-Level Operations (NEW!) --------------------

    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all chunks for a specific document.

        This was MISSING in your version!
        """
        logger.info(f"Deleting chunks for document: {document_id}")

        # Query to find all chunks for this document
        # LangChain Chroma supports where filters
        collection = self.vector_store._collection

        # Get all IDs matching this document
        results = collection.get(
            where={"document_id": document_id}
        )

        if not results or not results['ids']:
            logger.warning(f"No chunks found for document {document_id}")
            return 0

        # Delete by IDs
        collection.delete(ids=results['ids'])

        deleted_count = len(results['ids'])
        logger.info(f"Deleted {deleted_count} chunks")
        return deleted_count

    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID."""
        try:
            results = self.collection.get(ids=[chunk_id])

            if not results or not results['ids']:
                return None

            return {
                'chunk_id': results['ids'][0],
                'content': results['documents'][0],
                'metadata': results['metadatas'][0],
                'embedding': results['embeddings'][0] if results.get('embeddings') else None
            }
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None

    # -------------------- Stats (NEW!) --------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.

        """
        collection = self.vector_store._collection

        count = collection.count()

        return {
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "total_chunks": count,
            "persist_directory": self.persist_directory,
            "distance_metric": settings.CHROMA_DISTANCE_METRIC,
            "has_data": count > 0
        }

    # -------------------- Optional: Clear Store --------------------

    def clear_store(self):
        """Clear all data from the store."""
        logger.warning("Clearing entire vector store!")

        # LangChain method
        self.vector_store.delete_collection()

        # Recreate empty collection
        self.vector_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        logger.info("Vector store cleared and recreated")


# ==================== Convenience Layer ====================

_global_store: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    """Get or initialize the global vector store."""
    global _global_store
    if _global_store is None:
        _global_store = VectorStoreManager()
    return _global_store


def add_document_to_store(chunks: List[DocumentChunk]) -> int:
    """Convenience function to add chunks."""
    store = get_vector_store()
    return store.add_chunks(chunks)


def search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Convenience function for search."""
    store = get_vector_store()
    return store.query_similar_chunks(query, top_k=top_k)