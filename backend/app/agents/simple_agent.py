"""
Simple Agent - Basic Q&A using RAG.

This agent is the central orchestrator for the RAG application. It coordinates
the retrieval of documents, management of conversation history, and generation
of responses from the LLM.
"""
from app.utils.logger import get_logger
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from fastapi import Depends

from app.services.llm_service import get_llm_service, LLMService
from app.services.conversation_service import get_conversation_service, ConversationService
from app.rag.retriever import get_retriever, DocumentRetriever, RetrievalResult
from app.core.config import settings
from app.models.message import Message
from app.utils.prompts import system_prompt

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """
    Response from the agent.
    """
    answer: str
    sources: List[RetrievalResult]
    conversation_id: str
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
    Orchestrating agent for conversation-aware RAG.
    """
    def __init__(
        self,
        llm_service: LLMService,
        conversation_service: ConversationService,
        retriever: DocumentRetriever,
        top_k: int = None,
        score_threshold: float = None,
        max_context_tokens: int = 2000
    ):
        self.llm = llm_service
        self.conversations = conversation_service
        self.retriever = retriever
        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        self.score_threshold = score_threshold or settings.RETRIEVAL_SCORE_THRESHOLD
        self.max_context_tokens = max_context_tokens
        logger.info(f"SimpleAgent initialized: top_k={self.top_k}, threshold={self.score_threshold}")


    def answer(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Answer a question using conversation-aware RAG.
        """
        logger.info(f"Answering question for conversation_id: {conversation_id}")
        
        # 1. Get or create the conversation object from the database
        conversation = self.conversations.get_or_create_conversation(conversation_id)
        
        # 2. Add user message to DB
        self.conversations.add_message(conversation.id, "user", question)

        # 3. Retrieve documents
        retrieved = self.retriever.retrieve(query=question, top_k=self.top_k, filters=filters)
        if not retrieved:
            retrieved = "No relevant information could be found in the documents to for the query."
            # self.conversations.add_message(conversation.id, "assistant", answer_text)
            # return AgentResponse(answer=answer_text, sources=[], confidence="low", conversation_id=conversation.id)

        context = "No context found for the query."
        # 4. Construct Prompt
        if retrieved:
            context = self.retriever.format_context_for_llm(retrieved, max_tokens=self.max_context_tokens)

        prompt_with_context = system_prompt(question=question, context=context)

        # Get full history from DB for the LLM call
        history = self.conversations.get_history(conversation.id)

        # 5. Generate Answer
        try:
            llm_response = self.llm.generate(messages=history, system_prompt= prompt_with_context)
            
            # 6. Save Assistant's Response and Finalize
            self.conversations.add_message(conversation.id, "assistant", llm_response.content)
            
            # 7. Generate title if it's a new conversation
            # self.conversations.generate_title_if_needed(conversation)

            confidence = self._assess_confidence(retrieved)
            
            return AgentResponse(
                answer=llm_response.content,
                sources=retrieved,
                confidence=confidence,
                conversation_id=conversation.id,
                tokens_used=llm_response.total_tokens
            )
        except Exception as e:
            logger.error(f"Failed to answer question for conversation {conversation.id}: {e}", exc_info=True)
            error_answer = f"I encountered an error: {str(e)}"
            self.conversations.add_message(conversation.id, "assistant", error_answer)
            return AgentResponse(answer=error_answer, sources=[], confidence="low", conversation_id=conversation.id)

    def _assess_confidence(self, results: List[RetrievalResult]) -> str:
        if not results: return "low"
        high_quality = sum(1 for r in results if r.similarity_score > 0.7)
        if high_quality >= 2: return "high"
        medium_quality = sum(1 for r in results if r.similarity_score > 0.5)
        if medium_quality >= 1: return "medium"
        return "low"

# ==================== FastAPI Dependency ====================

def get_simple_agent(
    llm_service: LLMService = Depends(get_llm_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
    retriever: DocumentRetriever = Depends(get_retriever)
) -> SimpleAgent:
    """
    FastAPI dependency to get an instance of the SimpleAgent.
    This creates a new agent for each request, injecting the necessary, request-scoped services.
    """
    return SimpleAgent(
        llm_service=llm_service,
        conversation_service=conversation_service,
        retriever=retriever
    )
