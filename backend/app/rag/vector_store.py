"""
Vector Store - ChromaDB integration for storing and retrieving vector embeddings.

Concepts:
- Approximate nearest neighbor (ANN) algorithms for speed
- Vector indexing (HNSW algorithm)
- Similarity search vs keyword search
"""

from app.utils.logger import  get_logger
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import chromadb
from chromadb import Settings

from app.db import db_session
from app.models.document import DocumentChunk
from app.core.config import settings

logger = get_logger(__name__)


class VectorStore:
    """
    Manages vector storage and retrieval using ChromaDB.

    """

    def __init__(
            self,
            collection_name: Optional[str] = None,
            persist_directory: Optional[Path] = None
    ):
        """
        Initialize vector store with ChromaDB.

        Args:
            collection_name: Name of the collection (like a table)
        """
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or settings.VECTOR_DB_DIR

        logger.info(f"Initializing vector store with collection {self.collection_name}")
        logger.info(f"Persist directory: {self.persist_directory}")

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = db_session.get_chroma_client(str(self.persist_directory))

        # Get or create collection - Safe to call multiple times
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": settings.CHROMA_DISTANCE_METRIC,
                    "description": "Wellbore document embeddings for RAG pipeline",
                }
            )
            logger.info(f"Collection '{self.collection_name}' ready")
            logger.info(f"Current document count: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise


    def add_chunks(
            self,
            chunks: List[DocumentChunk],
            batch_size: int = 100
    ) -> int:
        """
        Add document chunks to vector store (batch insertions).

        Args:
            chunks: List of chunks with embeddings
            batch_size: Number of chunks per batch (for large collections)

        Returns:
            Number of chunks successfully added
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0

        # Validate chunks have embeddings
        chunks_with_embeddings = [c for c in chunks if c.embedding is not None]

        if len(chunks_with_embeddings) < len(chunks):
            logger.warning(
                f"{len(chunks) - len(chunks_with_embeddings)} chunks missing embeddings, skipping"
            )

        logger.info(f"Adding {len(chunks_with_embeddings)} chunks to vector store")

        added_count = 0

        #Process in batches
        for i in range(0, len(chunks_with_embeddings), batch_size):
            batch = chunks_with_embeddings[i:i + batch_size]
            logger.debug(f"Adding batch with {len(batch)} chunks to vector store")

            try:
                # Prepare batch data
                # Eg: ids=[...], embeddings=[...], documents=[...]
                ids = [chunk.chunk_id for chunk in batch]
                embeddings = [chunk.embedding for chunk in batch]
                documents = [chunk.content for chunk in batch]
                metadatas = [chunk.metadata for chunk in batch]

                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )

                added_count += len(batch)

                if (i + batch_size) % 500 == 0:
                    logger.debug(f"Progress: {added_count}/{len(chunks_with_embeddings)} chunks added")

            except Exception as e:
                logger.error(f"Failed to add batch starting at index {i} - {len(batch)}: {e}")

        logger.info(f"Successfully added {added_count} chunks to vector store")
        return added_count


    def query(
            self,
            query_embedding: List[float],
            top_k: int = 10,
            where: Optional[Dict[str, Any]] = None,
            where_document: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using query embedding.

        Args:
            query_embedding: Vector representation of query
            top_k: Number of results to return
            where: Metadata filters (e.g., {"page_number": 5})
            where_document: Document content filters

        Returns:
            List of matching chunks with metadata and scores
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding], # ChromaDB expects list
                n_results=top_k,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"],
            )

            # Transform ChromaDB results to our format - returns nested lists (batched queries)
            # We sent 1 query, so results are at index [0]
            formatted_results = []

            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'chunk_id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': self._distance_to_similarity(
                            results['distances'][0][i],
                            metric=settings.CHROMA_DISTANCE_METRIC
                        )
                    })

            logger.debug(f"Query returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to query embedding: {e}")
            return []


    def query_by_text(
            self,
            query_text: str,
            top_k: int = 10,
            where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (embedding generated automatically).

        This is the typical RAG retrieval flow!
        1. Embeds the query text
        2. Calls query() with the embedding
        """
        from app.rag.embeddings import embed_query

        query_embedding = embed_query(query_text)
        return self.query(query_embedding, top_k=top_k, where=where)


    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by ID.

        Use case:
        - User wants to see original source
        - Debugging which chunk was retrieved
        - Building citations
        """
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"]
            )

            if results['ids']:
                return {
                    'chunk_id': results['ids'][0],
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0],
                    'embedding': results['embeddings'][0] if results['embeddings'] is not None else None
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None


    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        - Returns count of deleted items
        """
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=[]
            )
            if not results['ids']:
                logger.info(f"No chunks found for document {document_id}")
                return 0

            # Delete by IDs
            self.collection.delete(ids=results['ids'])

            deleted_count = len(results['ids'])
            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {e}")
            return 0


    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        """
        try:
            count = self.collection.count()

            # Get sample to check metadata
            sample = self.collection.peek(limit=1)

            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": str(self.persist_directory),
                "distance_metric": settings.CHROMA_DISTANCE_METRIC,
                "has_data": count > 0,
                # "sample": sample
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


    def reset(self) -> bool:
        """
        Delete all data from collection.

        ⚠️ DESTRUCTIVE: Use only for testing or Development: Reset when schema changes
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.warning(f"Collection '{self.collection_name}' deleted")

            # Recreate empty collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": settings.CHROMA_DISTANCE_METRIC,
                    "description": "Wellbore document embeddings for RAG pipeline"
                }
            )
            logger.info(f"Collection '{self.collection_name}' recreated")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False


    def _distance_to_similarity(self, distance: float, metric: str) -> float:
        """
        Convert distance metric to similarity score (0-1 range).

        Conversions:
        - Cosine distance: similarity = 1 - distance
        - L2 distance: similarity = 1 / (1 + distance)
        - Inner product: similarity = (1 + distance) / 2
        """
        if metric == "cosine":
            # Cosine distance: 0 (identical) to 2 (opposite)
            # Similarity: 1 (identical) to -1 (opposite), normalized to 0-1
            return 1 - distance
        elif metric == "l2":
            # Euclidean distance: 0 (identical) to infinity
            # Convert to 0-1 range
            return 1 / (1 + distance)
        elif metric == "ip":
            # Inner product: -infinity to +infinity
            # Normalize to 0-1 range (rough approximation)
            return (1 + distance) / 2
        else:
            return distance


# ==================== Module-level convenience functions ====================
# Global instance (lazy loading)
_global_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get or create global vector store instance.

    Usage:
        store = get_vector_store()
        results = store.query_by_text("well depth")
    """
    global _global_vector_store

    if _global_vector_store is None:
        _global_vector_store = VectorStore()

    return _global_vector_store


def add_document_to_store(chunks: List[DocumentChunk]) -> int:
    """
    Convenience function to add chunks to vector store.

    Usage:
        doc = process_pdf(path)
        chunks = chunk_document(doc)
        chunks = embed_chunks(chunks)
        add_document_to_store(chunks)  # Done!
    """
    store = get_vector_store()
    return store.add_chunks(chunks)


def search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function for simple search.

    Usage:
        results = search_documents("What is the well depth?")
        for result in results:
            print(result['content'])
    """
    store = get_vector_store()
    return store.query_by_text(query, top_k=top_k)