"""
Retriever - High-level interface for document retrieval.

This module provides the clean RAG interface that the agent will use.
It abstracts away the complexity of embedding, vector search, and result formatting.
"""
import numpy as np
from sentence_transformers import CrossEncoder
from app.utils.logger import get_logger
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.rag.vector_store_manager import get_vector_store
from app.rag.embeddings import embed_query
from app.core.config import settings

logger = get_logger(__name__)

@dataclass
class RetrievalResult:
    """
    Single retrieval result with all necessary information.
    """
    chunk_id: str
    content: str
    page_number: int
    document_id: str
    well_id: str
    well_name: str
    document_type: str
    filename: str
    similarity_score: float
    chunk_type: str  # "text" or "table"
    metadata: Dict[str, Any]
    # well_name: Optional[str] = None

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"[{self.filename} - Page {self.page_number}]"
            f"(score: {self.similarity_score:.3f})\n"
            f"{self.content[:100]}..."
        )

    @property
    def citation(self):
        """Generate citation string."""
        return f"{self.well_name} - {self.document_type} in {self.filename}, page {self.page_number}, chunk_type {self.chunk_type}"  #


class DocumentRetriever:
    """
    High-level retrieval interface for RAG pipeline.

    """

    def __init__(
            self,
            top_k: int = None,
            score_threshold: float = None,
    ):
        """
        Initialize retriever.

        Args:
            top_k: Max results to return (default from settings)
            score_threshold: Min similarity score (default from settings)
        """
        self.vector_store = get_vector_store()
        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        self.score_threshold = score_threshold or settings.RETRIEVAL_SCORE_THRESHOLD

        # Load once at startup
        self.reranker_model = CrossEncoder("BAAI/bge-reranker-base")  # or any other reranker model

        logger.info(
            f"DocumentRetriever initialized: "
            f"top_k={self.top_k}, threshold={self.score_threshold}"
        )


    def retrieve(
            self,
            query: str,
            top_k: Optional[int] = None,
            filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        This is the main RAG retrieval method! - Retrieve relevant chunks for a query.

         Process:
         1. Convert query to embedding - Search vector store - Filter by score threshold - Format results - Return sorted by relevance

         Args:
             query: User's question or search query
             top_k: Override default number of results
             filters: Metadata filters (e.g., {"chunk_type": "table"})

         Returns:
             List of RetrievalResult objects, sorted by relevance
         """
        k = top_k or self.top_k

        logger.debug(f"Retrieving top {k} results chunks for query: `{query[:50]}...`")

        try:
            # Search vector store
            raw_results = self.vector_store.query_similar_chunks(
                query=query,
                top_k=k,
                filters=filters,
            )

            # Convert to Retrieval Result objects and filter by threshold
            results = []
            for raw in raw_results:
                similarity = raw.get('similarity_score', 0.0)

                # # Filter by threshold
                # if similarity < self.score_threshold:
                #     logger.debug(
                #         f"Filtering out result with score {similarity:.3f} "
                #         f"(threshold: {self.score_threshold})"
                #     )
                #     continue

                # Extract metadata
                metadata = raw.get('metadata', {})

                result = RetrievalResult(
                    chunk_id=raw['chunk_id'],
                    content=raw['content'],
                    page_number=metadata.get('page_number', 0),
                    document_id=metadata.get('document_id', ''),
                    document_type=metadata.get('document_type'),
                    well_id=metadata.get('well_id', 'unknown'),
                    well_name=metadata.get('well_name', 'unknown'),
                    filename=metadata.get('filename', 'unknown'),
                    similarity_score=similarity,
                    chunk_type=metadata.get('chunk_type', 'text'),
                    metadata=metadata,
                )
                results.append(result)
                logger.debug(
                    f"Retrieved result with score {similarity:.3f} "
                    f"(threshold: {self.score_threshold})"
                )

            if not results:
                logger.warning("No candidates retrieved from vector store, returning empty list")
                return []

            # Step 2: Apply reranker if available
            if self.reranker_model and results:
                results = self._rerank_results(query, results)
                logger.info('Applied reranking to retrieved chunks')

            # Step 3: Ensure at least 3 results
            if len(results) < 3:
                logger.warning(f"Only {len(results)} results retrieved; duplicating to ensure minimum 3")
                while len(results) < 3: results.append(results[-1])

            logger.info(
                f"Retrieved {len(results)} chunks (reranked from {len(raw_results)} candidates"
            )
            # Step 4: Return top-k
            return results[:k]

        except Exception as e:
            logger.error(f"Retrieval failed for query: `{query}`: {e}")
            return []



    def _rerank_results(self, query: str, chunks: list[RetrievalResult]) -> list[RetrievalResult]:
        """
        Reranks retrieved chunks using a Hugging Face cross-encoder.
        Each chunk must have a 'content' field.
        Returns chunks sorted by relevance score (descending).
        """
        logger.debug("Reranking results...")
        # Prepare input pairs: (query, chunk_text)
        pairs = [(query, r.content) for r in chunks]

        # Predict raw scores
        raw_scores = np.array(self.reranker_model.predict(pairs))

        # Normalize
        min_val, max_val = raw_scores.min(), raw_scores.max()
        normalized = (raw_scores - min_val) / (max_val - min_val + 1e-8)

        # Attach scores + log per chunk
        for chunk, raw, norm in zip(chunks, raw_scores, normalized):
            chunk.metadata["rerank_score"] = float(raw)
            chunk.metadata["rerank_score_norm"] = float(norm)

            logger.debug(
                f"Chunk {chunk.chunk_id} | RAW: {raw:.4f} | NORMALIZED: {norm:.4f} | Previous Score: {chunk.similarity_score}"
                f"\nPreview: {chunk.content[:60]}..."
            )

        # Sort by normalized score
        return sorted(chunks, key=lambda c: c.metadata["rerank_score_norm"], reverse=True)



    def retrieve_for_summarization(
            self,
            well_name: Optional[str],
            document_id: Optional[str],
            max_chunks: int = 20,
            chunk_index_range: Optional[Tuple[int, int]] = None
    ) -> List[RetrievalResult]:
         """
          Retrieve chunks for document summarization (Sub-challenge 1).

          Different from search:
          - Get chunks from specific document - Coverage-based (whole document)
          - Order by page/chunk index (not relevance)
         """
         try:
            where: Dict[str, any] = {}
            if document_id:
                where["document_id"] = document_id
            elif well_name:
                where["well_name"] = well_name

            elif chunk_index_range:
                where["chunk_index"] = {
                    "$gte": chunk_index_range[0],
                    "$lte": chunk_index_range[1]
                }

            raw_results = self.vector_store.collection.get(
                where=where,
                include=["documents", "metadatas"],
                limit=max_chunks
            )

            # Convert to results
            results = []
            for i in range(len(raw_results['ids'])):
                metadata = raw_results['metadatas'][i]

                result = RetrievalResult(
                    chunk_id=raw_results['ids'][i],
                    content=raw_results['documents'][i],
                    page_number=metadata.get('page_number', 0),
                    document_id=metadata.get('document_id', ''),
                    filename=metadata.get('filename', 'unknown'),
                    document_type=metadata.get('document_type'),
                    well_id=metadata.get('well_id', 'unknown'),
                    well_name=metadata.get('well_name', 'unknown'),
                    similarity_score=1.0,  # Not relevance-based
                    chunk_type=metadata.get('chunk_type', 'text'),
                    metadata=metadata
                )
                results.append(result)

            # Sort by page and chunk index for logical order
            results.sort(key=lambda x: (x.page_number, x.metadata.get('chunk_index', 0)))

            logger.info(f"Retrieved {len(results)} chunks for summarization (well={well_name}, doc={document_id})")
            return results

         except Exception as e:
            logger.error(f"Failed to retrieve for summarization: {e}")
            return []



    def retrieve_tables_only(
            self,
            query: str,
            well_name: Optional[str],
            document_id: Optional[str],
            top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
         Retrieve only table chunks (for parameter extraction).
         """
        filters = {"chunk_type": "table"}
        if well_name:
            filters["well_name"] = well_name
        if document_id:
            filters["document_id"] = document_id

        return self.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
        )



    def retrieve_from_pages(
            self,
            query: str,
            well_name: Optional[str],
            document_id: Optional[str],
            page_numbers: List[int],
            top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve from specific pages only.
        """
        page_filter = {"page_number": {"$in": page_numbers}}
        # $in operator: match any of the list values

        if well_name:
            page_filter["well_name"] = well_name
        if document_id:
            page_filter["document_id"] = document_id
        # ChromaDB filter for page numbers

        return self.retrieve(
            query=query,
            top_k=top_k,
            filters=page_filter,
        )



    def get_context_window(
            self,
            chunk_id: str,
            window_size: int = 2,
    ) -> List[RetrievalResult]:
        """
        Get surrounding chunks for context - Provides fuller context like reading paragraph around a sentence

        Example:
        If chunk_index = 10, window_size = 2:
        - Get chunks: 8, 9, [10], 11, 12
        - Returns 5 chunks total
        """
        try:
            # Get the target chunk
            target = self.vector_store.get_by_id(chunk_id)
            if not target:
                return []

            metadata = target['metadata']
            document_id = metadata.get('document_id', '')
            chunk_index = metadata.get('chunk_index', 0)

            # Calculate range
            start_index = max(0, chunk_index - window_size)
            end_index = chunk_index + window_size

            # Get all chunks in range
            return self.retrieve_for_summarization(
                document_id,
                max_chunks=100,
                chunk_index_range=(start_index, end_index)
            )

        except Exception as e:
            logger.error(f"Failed to get context window: {e}")
            return []



    def format_context_for_llm(
            self,
            results: List[RetrievalResult],
            max_tokens: int = 2000
    ) -> str:
        """
        Format retrieval results for LLM consumption. This method creates the "Context from documents" part.

        Args:
            results: Retrieved chunks
            max_tokens: Approximate token limit (~4 chars = 1 token)

        Returns:
            Formatted string ready for LLM prompt
        """
        if not results:
            return "No relevant information found."

        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4 # Rough approximation

        for i, result in enumerate(results, 1):
            citation = result.citation or "No source"
            # Format each chunk
            chunk_text = (
                f"[Source {i}: {citation}]\n"
                f"{result.content}\n"
            )

            # Check if adding this would exceed limit
            if total_chars + len(chunk_text) > max_chars:
                logger.debug(f"Reached token limit, using {i+1}/{len(results)} chunks")
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        context = "\n----\n\n".join(context_parts)

        logger.debug(
            f"Formatted context: {len(context_parts)} chunks, "
            f"~{total_chars / 4:.0f} tokens"
        )

        return context


# ==================== Module-level convenience functions ====================

# Global instance
_global_retriever: Optional[DocumentRetriever] = None

def get_retriever() -> DocumentRetriever:
    """Get or create global retriever instance."""
    global _global_retriever

    if _global_retriever is None:
        _global_retriever = DocumentRetriever()

    return _global_retriever


def search(query: str, top_k: int = 5) -> List[RetrievalResult]:
    """
    Simple search function.

    Usage:
        results = search("well depth")
        for r in results:
            print(r.citation, r.similarity_score)
    """
    retriever = get_retriever()
    return retriever.retrieve(query, top_k=top_k)