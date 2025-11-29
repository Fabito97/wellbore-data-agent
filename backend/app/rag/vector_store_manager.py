"""
Enhanced Vector Store - Best of both worlds.
"""
from typing import List, Optional, Dict, Any, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.db.chroma import get_chroma_client
from app.models.document import DocumentChunk
from app.models.schema import RetrievalResult
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
        self.vector_store = get_chroma_client(embedding_func=self.embeddings)

        self.collection = self.vector_store._collection

        logger.info("Vector store ready")



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

        # If embeddings are already present, use add_embeddings
        if hasattr(chunks_with_ids[0], "embedding") and chunks_with_ids[0].embedding is not None:
            embeddings = [chunk.embedding for chunk in chunks_with_ids]
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
                documents=texts
            )

            logger.info(f"Added {len(chunks_with_ids)} chunks with precomputed embeddings")

        else:
            # Fall back to auto-embedding if none provided
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Successfully added {len(chunks_with_ids)} chunks")

        return len(chunks_with_ids)


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
        filter_clause = {}
        if filters:
            # If multiple conditions, wrap in $and
            if len(filters) > 1:
                filter_clause = {
                    "$and": [
                        {key: value} for key, value in filters.items()
                    ]
                }
            else:
                filter_clause = filters  # ← FIXED: was "filters_clause"

            results = self.vector_store.similarity_search_with_score(
                query, k=top_k, filter=filter_clause  # LangChain supports this!
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
                "similarity_score": 1 - distance,  # normalize_score(distance)  # Convert distance to similarity
                "chunk_id": doc.id
            })

        logger.info(f"Found {len(formatted)} results")
        return formatted


    # -------------------- Document-Level Operations --------------------

    def get_by_document_id(
            self, document_id: Optional[str] = None,
            chunk_type: Optional[str] = None,
            page_range: Optional[Tuple[int, int]] = None,
            max_chunks: Optional[int] = None,
            chunk_index_range: Optional[Tuple[int, int]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks for a specific document with optional filters.
        Supports compound filtering using $and:
        - document_id (required)
        - chunk_type (optional)
        - page_range (optional, inclusive range of page_number)
        - max_chunks (optional, limit number of chunks returned)
        """

        logger.info(f"Retrieving chunks for document: {document_id}")

        # Query to find all chunks for this document
        # LangChain Chroma supports where filters
        collection = self.collection

        # Build compound filters
        filters: List[Dict[str, Any]] = [{"document_id": document_id}]  # always included

        if chunk_type:
            filters.append({"chunk_type": chunk_type})
        if page_range:
            filters.append({"page_number": {"$gte": page_range[0]}})
            filters.append({"page_number": {"$lte": page_range[1]}})

        if chunk_index_range:
            filters.append({"chunk_index": {"$gte": chunk_index_range[0]}})
            filters.append({"chunk_index": {"$lte": chunk_index_range[1]}})

        # Combine with $and if multiple filters
        where = filters[0] if len(filters) == 1 else {"$and": filters}

        # Query collection
        # Get all chunks matching this document with filters
        results = collection.get(
            where=where,
            include=["metadatas", "documents", ],
            limit=max_chunks if max_chunks else None
        )

        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        if not ids:
            logger.warning(f"No chunks found for document {document_id}")
            return []

        logger.info(f"Retrieved {len(results)} results")
        return [
            RetrievalResult(
                chunk_id=ids[i],
                content=docs[i],
                page_number=metas[i].get("page_number", 0),
                document_id=metas[i].get("document_id", ""),
                filename=metas[i].get("filename", "unknown"),
                document_type=metas[i].get("document_type"),
                well_id=metas[i].get("well_id", "unknown"),
                well_name=metas[i].get("well_name", "unknown"),
                similarity_score=1.0,
                chunk_type=metas[i].get("chunk_type", "text"),
                metadata=metas[i],
            )
            for i in range(len(ids))
        ]


    def get_by_well_name(
            self,
            well_name: str,
            chunk_type: Optional[str] = None,
            page_range: Optional[Tuple[int, int]] = None,
            max_chunks: Optional[int] = None,
            document_type: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks for a specific well with optional filters.
        Supports compound filtering using $and:
        - well_name (required)
        - chunk_type (optional)
        - page_range (optional)
        - chunk_index_range (optional)
        - max_chunks (optional limit)
        """

        logger.info(f"Retrieving chunks for well: {well_name} and type: {document_type}, "
                    f"chunk_type={chunk_type}, page_range={page_range}, ")

        filters: List[Dict[str, Any]] = [{"well_name": well_name}]
        if document_type:
            filters.append({"document_type": document_type})
        if chunk_type:
            filters.append({"chunk_type": chunk_type})
        if page_range:
            filters.append({"page_number": {"$gte": page_range[0], "$lte": page_range[1]}})

        where = filters[0] if len(filters) == 1 else {"$and": filters}

        results = self.collection.get(
            where=where,
            include=["metadatas", "documents"],
            limit=max_chunks if max_chunks else None
        )

        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        if not ids:
            logger.warning(f"No chunks found for well {well_name}")
            return []

        return [
            RetrievalResult(
                chunk_id=ids[i],
                content=docs[i],
                page_number=metas[i].get("page_number", 0),
                document_id=metas[i].get("document_id", ""),
                filename=metas[i].get("filename", "unknown"),
                document_type=metas[i].get("document_type"),
                well_id=metas[i].get("well_id", "unknown"),
                well_name=metas[i].get("well_name", "unknown"),
                similarity_score=1.0,
                chunk_type=metas[i].get("chunk_type", "text"),
                metadata=metas[i],
            )
            for i in range(len(ids))
        ]


    def get_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID."""
        try:
            results = self.collection.get(ids=[chunk_id], include=["metadatas", "documents", "embeddings"])

            logger.debug("Checking if chunk exists")
            ids = results.get('ids', [])
            logger.debug("Checking if chunk exists")
            if len(ids) == 0:
                logger.warning(f"No chunks found for document {chunk_id}")
                return None

            logger.debug(f"Retrieved {len(ids)} results")

            embeddings = results.get('embeddings', [])
            embedding = embeddings[0] if len(embeddings) > 0 else None

            return {
                'chunk_id': ids[0],
                'content': results['documents'][0],
                'metadata': results['metadatas'][0],
                'embedding': embedding,
            }
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None


    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all chunks for a specific document.
        """
        logger.info(f"Deleting chunks for document: {document_id}")

        # Query to find all chunks for this document
        # LangChain Chroma supports where filters
        collection = self.collection

        # Get all IDs matching this document
        results = collection.get(
            where={"document_id": document_id}
        )

        ids = results.get('ids', [])
        if len(ids) == 0:
            logger.warning(f"No chunks found for document {document_id}")
            return 0

        # Delete by IDs
        collection.delete(ids=results['ids'])

        deleted_count = len(results['ids'])
        logger.info(f"Deleted {deleted_count} chunks")
        return deleted_count


    # -------------------- Stats (NEW!) --------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.

        """
        count = self.collection.count()

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
        self.vector_store = get_chroma_client(embedding_func=self.embeddings)

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