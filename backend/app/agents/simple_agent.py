"""
Simple Agent - Basic Q&A using RAG.

This is a straightforward RAG agent that:
1. Takes a user question
2. Retrieves relevant chunks from documents
3. Sends question + context to LLM
4. Returns answer with citations

Teaching: RAG Architecture
This is the fundamental RAG pattern that everything builds on.
The orchestrator will use this as a tool, but it works standalone too.

Flow:
User Question → Retriever → Context → LLM → Answer
"""
from app.utils.logger import get_logger
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from app.services.llm_service import get_llm_service, LLMResponse
from app.rag.retriever import get_retriever, RetrievalResult
from app.core.config import settings

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """
    Response from the agent.
    """
    answer: str
    sources: List[RetrievalResult]
    confidence: str  # "high", "medium", "low"
    tokens_used: Optional[int] = None

    def format_for_display(self) -> str:
        """Format response for text display."""
        output = f"Answer: {self.answer}\n\n"

        if self.sources:
            output += "Sources:\n"
            for i, source in enumerate(self.sources, 1):
                output += f"{i}. {source.citation} (score: {source.similarity_score:.2f})\n"

        output += f"\nConfidence: {self.confidence}"

        return output


class SimpleAgent:
    """
    Basic RAG agent for question answering.
    """

    def __init__(
            self,
            top_k: int = None,
            score_threshold: float = None,
            max_context_tokens: int = 2000
    ):
        """
        Initialize simple agent.

        Args:
            top_k: Max chunks to retrieve
            score_threshold: Min similarity for retrieval
            max_context_tokens: Max tokens for context
        """
        self.llm = get_llm_service()
        self.retriever = get_retriever()

        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        self.score_threshold = score_threshold or settings.RETRIEVAL_SCORE_THRESHOLD
        self.max_context_tokens = max_context_tokens

        logger.info(
            f"SimpleAgent initialized: top_k={self.top_k}, "
            f"threshold={self.score_threshold}"
        )

    def answer(
            self,
            question: str,
            filters: Optional[Dict[str, Any]] = None,
            include_sources: bool = True
    ) -> AgentResponse:
        """
        Answer a question using RAG - This is the main method (the core RAG loop).

        Args:
            question: User's question
            filters: Optional metadata filters
            include_sources: Include source citations

        Returns:
            AgentResponse with answer and sources
        """
        logger.info(f"Answering question: {question[:100]}...")

        try:
            # Step 1: Retrieve relevant chunks
            logger.debug(f"Retrieving top {self.top_k} chunks")

            retrieved = self.retriever.retrieve(
                query=question,
                top_k=self.top_k,
                filters=filters
            )

            if not retrieved:
                # No relevant documents found
                logger.warning(f"No documents retrieved for question: {question}")
                return AgentResponse(
                    answer="I couldn't find any relevant information in the documents to answer your question.",
                    sources=[],
                    confidence="low"
                )

            logger.debug(f"Retrieved {len(retrieved)} chunks")

            # Step 2: Format context for LLM
            context = self.retriever.format_context_for_llm(
                retrieved,
                max_tokens=self.max_context_tokens
            )

            # Step 3: Generate answer with LLM
            logger.debug("Generating answer with LLM")

            llm_response = self.llm.generate_with_context(
                question=question,
                context=context
            )

            # Step 4: Assess confidence
            # Teaching: Simple heuristic
            # - High: Multiple high-scoring chunks
            # - Medium: Some good chunks
            # - Low: Only low-scoring chunks or few results
            confidence = self._assess_confidence(retrieved)

            # Step 5: Create response
            response = AgentResponse(
                answer=llm_response.content,
                sources=retrieved if include_sources else [],
                confidence=confidence,
                tokens_used=llm_response.total_tokens
            )

            logger.info(f"Answer generated (confidence: {confidence})")

            return response

        except Exception as e:
            logger.error(f"Failed to answer question: {e}", exc_info=True)

            # Return error response
            return AgentResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence="low"
            )

    def summarize_document(
            self,
            document_id: str,
            max_words: int = 200
    ) -> AgentResponse:
        """
        Summarize a complete document (Sub-challenge 1).

        Args:
            document_id: Document to summarize
            max_words: Target summary length

        Returns:
            AgentResponse with summary
        """
        logger.info(f"Summarizing document: {document_id}")

        try:
            # Get representative chunks
            chunks = self.retriever.retrieve_for_summarization(
                document_id=document_id,
                max_chunks=20  # Get spread across document
            )

            if not chunks:
                return AgentResponse(
                    answer="Document not found or has no content.",
                    sources=[],
                    confidence="low"
                )

            # Format context
            context = self.retriever.format_context_for_llm(chunks)

            # Create summarization prompt
            prompt = f"""Please provide a concise summary of this document in approximately {max_words} words.
                    
                        Focus on:
                        - Well identification and location
                        - Key technical specifications (depth, diameter, etc.)
                        - Completion details
                        - Important findings or observations
                        
                        Context:
                        {context}
                        
                        Summary (target {max_words} words):
                    """

            # Generate summary
            llm_response = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a technical summarization assistant specialized in petroleum engineering."
            )

            return AgentResponse(
                answer=llm_response.content,
                sources=chunks,
                confidence="high",  # Summarization is reliable with full context
                tokens_used=llm_response.total_tokens
            )

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return AgentResponse(
                answer=f"Failed to summarize document: {str(e)}",
                sources=[],
                confidence="low"
            )

    def extract_tables(
            self,
            query: str,
            top_k: int = 5
    ) -> AgentResponse:
        """
        Find and return relevant tables.

        Args:
            query: What to look for
            top_k: Max tables to return

        Returns:
            AgentResponse with table chunks
        """
        logger.info(f"Extracting tables for query: {query}")

        tables = self.retriever.retrieve_tables_only(query, top_k=top_k)

        if not tables:
            return AgentResponse(
                answer="No relevant tables found.",
                sources=[],
                confidence="low"
            )

        # Format tables for display
        table_descriptions = []
        for i, table in enumerate(tables, 1):
            desc = f"Table {i} (Page {table.page_number}):\n{table.content[:300]}..."
            table_descriptions.append(desc)

        answer = "Found relevant tables:\n\n" + "\n\n---\n\n".join(table_descriptions)

        return AgentResponse(
            answer=answer,
            sources=tables,
            confidence="high"
        )

    def _assess_confidence(self, results: List[RetrievalResult]) -> str:
        """
        Assess confidence in answer based on retrieval quality.

        Args:
            results: Retrieved chunks

        Returns:
            "high", "medium", or "low"
        """
        if not results:
            return "low"

        # Count high-quality results
        high_quality = sum(1 for r in results if r.similarity_score > 0.7)
        medium_quality = sum(1 for r in results if r.similarity_score > 0.5)

        if high_quality >= 3:
            return "high"
        elif medium_quality >= 2:
            return "medium"
        else:
            return "low"

    def answer_with_context_window(
            self,
            question: str,
            window_size: int = 2
    ) -> AgentResponse:
        """
        Answer with expanded context (neighboring chunks).

        Args:
            question: User's question
            window_size: How many chunks before/after

        Returns:
            AgentResponse with expanded context
        """
        # First, retrieve best match
        initial_results = self.retriever.retrieve(question, top_k=1)

        if not initial_results:
            return self.answer(question)  # Fallback to normal answer

        # Get context window around best match
        best_match = initial_results[0]
        window_chunks = self.retriever.get_context_window(
            chunk_id=best_match.chunk_id,
            window_size=window_size
        )

        if not window_chunks:
            return self.answer(question)  # Fallback

        # Format expanded context
        context = self.retriever.format_context_for_llm(window_chunks)

        # Generate answer
        llm_response = self.llm.generate_with_context(
            question=question,
            context=context
        )

        return AgentResponse(
            answer=llm_response.content,
            sources=window_chunks,
            confidence=self._assess_confidence(window_chunks),
            tokens_used=llm_response.total_tokens
        )


# ==================== Module-level instance ====================

_simple_agent_instance: Optional[SimpleAgent] = None


def get_simple_agent() -> SimpleAgent:
    """
    Get or create global simple agent instance.

    Usage:
        from app.agents.simple_agent import get_simple_agent

        agent = get_simple_agent()
        response = agent.answer("What is the well depth?")
        print(response.answer)
    """
    global _simple_agent_instance

    if _simple_agent_instance is None:
        _simple_agent_instance = SimpleAgent()

    return _simple_agent_instance


# ==================== Convenience functions ====================

def ask(question: str) -> str:
    """
    Simple function for quick Q&A.

    Usage:
        from app.agents.simple_agent import ask

        answer = ask("What is the well depth?")
        print(answer)
    """
    agent = get_simple_agent()
    response = agent.answer(question)
    return response.answer