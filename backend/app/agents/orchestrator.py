"""
Orchestrated Agent - Multi-step reasoning with RAG tools.

Workflow:
1. Detect well from query
2. Route to appropriate tool
3. Execute retrieval
4. Generate answer
"""
from app.rag.prompts import extraction_prompt, generate_summary_prompt
from app.utils.logger import get_logger
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from fastapi import Depends

from app.services.llm_service import get_llm_service, LLMService
from app.services.conversation_service import get_conversation_service, ConversationService
from app.rag.retriever import get_retriever, DocumentRetriever
from app.agents.tools.rag_tool import (
    detect_well_from_query,
    list_available_wells,
    rag_query_tool,
    summarize_well_report_tool,
    extract_parameters_tool,
    list_tool_descriptions
)
from app.core.config import settings

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """Response from agent."""
    answer: str
    well_name: Optional[str]
    sources_count: int
    confidence: str
    conversation_id: str
    tool_used: Optional[str] = None
    tokens_used: Optional[int] = None


class OrchestratedAgent:
    """
    Agent with multi-step reasoning and tool usage.

    Capabilities:
    1. Detect well from query
    2. Route to correct tool
    3. Execute well-scoped retrieval
    4. Generate contextual answer
    """

    def __init__(
            self,
            llm_service: LLMService,
            conversation_service: ConversationService,
            retriever: DocumentRetriever
    ):
        self.llm = llm_service
        self.conversations = conversation_service
        self.retriever = retriever

        logger.info("OrchestratedAgent initialized")

    def answer(
            self,
            question: str,
            conversation_id: Optional[str] = None
    ) -> AgentResponse:
        """
        Answer question with multi-step reasoning.

        Steps:
        1. Detect well reference
        2. Determine intent (summarize, extract params, general query)
        3. Execute appropriate tool
        4. Generate answer
        """
        logger.info(f"Processing query: '{question[:100]}...'")

        # Get conversation
        # conversation = self.conversations.get_or_create_conversation(conversation_id)
        # self.conversations.add_message(conversation.id, "user", question)

        try:
            # Step 1: Detect well
            well_name = detect_well_from_query(query=question)
            logger.debug(f"Extracted well name: {well_name}")

            if not well_name:
                logger.warning("No well detected, listing available wells")
                wells = list_available_wells()
                if not wells:
                    answer = "No wells found in the system. Please upload well documents first."
                else:
                    well_list = ", ".join([w["well_name"] for w in wells])
                    answer = (
                        f"I couldn't determine which well you're asking about. "
                        f"Available wells: {well_list}. "
                        f"Please specify a well name in your question."
                    )

                # self.conversations.add_message(conversation.id, "assistant", answer)
                return AgentResponse(
                    answer=answer,
                    well_name=None,
                    sources_count=0,
                    confidence="low",
                    # conversation_id=conversation.id
                )

            logger.info(f"Detected well: {well_name}")

            # Step 2: Determine intent and execute
            intent = self._determine_intent(question)
            logger.info(f"Determined intent: {intent}")

            if intent == "summarize":
                result = self._handle_summarization(question, well_name, "")
            elif intent == "extract_parameters":
                result = self._handle_parameter_extraction(question, well_name, "")
            else:
                result = self._handle_general_query(question, well_name, "")

            return result

        except Exception as e:
            logger.exception(f"Agent error: {e}")
            error_msg = f"I encountered an error: {str(e)}"
            # self.conversations.add_message(conversation.id, "assistant", error_msg)

            return AgentResponse(
                answer=error_msg,
                well_name=None,
                sources_count=0,
                confidence="low",
                # conversation_id=conversation.id
            )

    def _determine_intent(self, question: str) -> str:
        """
        Determine what the user wants.

        Intents:
        - "summarize": Generate report summary
        - "extract_parameters": Extract nodal analysis params
        - "general_query": Answer specific question
        """
        question_lower = question.lower()

        # Summarization keywords
        if any(kw in question_lower for kw in [
            "summarize", "summary", "overview", "describe the well",
            "tell me about", "what is in the report"
        ]):
            return "summarize"

        # Parameter extraction keywords
        if any(kw in question_lower for kw in [
            "extract parameters", "nodal analysis", "tubing specifications",
            "reservoir pressure", "productivity index", "well parameters",
            "completion data"
        ]):
            return "extract_parameters"

        # Default: general query
        return "general_query"

    def _handle_summarization(
            self,
            question: str,
            well_name: str,
            conversation_id: str
    ) -> AgentResponse:
        """Handle summarization requests (Sub-challenge 1)."""
        logger.info(f"Handling summarization for {well_name}")

        # Get context
        result = summarize_well_report_tool(well_name, max_words=200)

        if result["chunks_used"] == 0:
            answer = result["context"]  # Error message
        else:
            # Generate summary using LLM
            summary_system_prompt = generate_summary_prompt(well_name, result["context"])

            history = self.conversations.get_history(conversation_id)
            llm_response = self.llm.generate(
                messages=history,
                system_prompt=summary_system_prompt
            )

            answer = llm_response.content

        # Save response
        self.conversations.add_message(conversation_id, "assistant", answer)

        return AgentResponse(
            answer=answer,
            well_name=well_name,
            sources_count=result["chunks_used"],
            confidence="high" if result["chunks_used"] > 0 else "low",
            conversation_id=conversation_id,
            tool_used="summarize_report",
            tokens_used=getattr(llm_response, 'total_tokens', None) if result["chunks_used"] > 0 else None
        )

    def _handle_parameter_extraction(
            self,
            question: str,
            well_name: str,
            conversation_id: str
    ) -> AgentResponse:
        """Handle parameter extraction (Sub-challenge 2)."""
        logger.info(f"Handling parameter extraction for {well_name}")

        # Get context with tables and relevant text
        result = extract_parameters_tool(well_name)

        if result["total_chunks"] == 0:
            answer = f"No data found for {well_name} to extract parameters from."
        else:
            # Extraction prompt

            extraction_system_prompt = extraction_prompt(well_name, result["context"])

            history = self.conversations.get_history(conversation_id)
            llm_response = self.llm.generate(
                messages=history,
                system_prompt=extraction_system_prompt
            )

            answer = llm_response.content

        # Save response
        self.conversations.add_message(conversation_id, "assistant", answer)

        return AgentResponse(
            answer=answer,
            well_name=well_name,
            sources_count=result["total_chunks"],
            confidence="high" if result["table_chunks"] else "medium",
            conversation_id=conversation_id,
            tool_used="extract_parameters",
            tokens_used=getattr(llm_response, 'total_tokens', None) if result["total_chunks"] > 0 else None
        )

    def _handle_general_query(
            self,
            question: str,
            well_name: str,
            conversation_id: str
    ) -> AgentResponse:
        """Handle general queries."""
        logger.info(f"Handling general query for {well_name}")

        # Query with well filter
        result = rag_query_tool(
            query=question,
            well_name=well_name,
            top_k=5
        )

        if result["count"] == 0:
            answer = f"I couldn't find relevant information about {well_name} for your question."
        else:
            # Format context
            context = self.retriever.format_context_for_llm(
                result["results"],
                max_tokens=2000
            )

            # Generate answer
            answer_prompt = f"""
Answer the following question about {well_name} using the provided context from well documents.

Question: {question}

Context from {well_name} documents:
{context}

Provide a clear, accurate answer based on the context. If the context doesn't contain enough information, say so.
"""

            history = self.conversations.get_history(conversation_id)
            llm_response = self.llm.generate(
                messages=history,
                system_prompt=answer_prompt
            )

            answer = llm_response.content

        # Save response
        self.conversations.add_message(conversation_id, "assistant", answer)

        # Assess confidence
        if result["count"] >= 3:
            confidence = "high"
        elif result["count"] >= 1:
            confidence = "medium"
        else:
            confidence = "low"

        return AgentResponse(
            answer=answer,
            well_name=well_name,
            sources_count=result["count"],
            confidence=confidence,
            conversation_id=conversation_id,
            tool_used="rag_query",
            tokens_used=getattr(llm_response, 'total_tokens', None) if result["count"] > 0 else None
        )


# ==================== FastAPI Dependency ====================

def get_orchestrated_agent(
        llm_service: LLMService = Depends(get_llm_service),
        conversation_service: ConversationService = Depends(get_conversation_service),
        retriever: DocumentRetriever = Depends(get_retriever)
) -> OrchestratedAgent:
    """Get agent instance."""
    return OrchestratedAgent(
        llm_service=llm_service,
        conversation_service=conversation_service,
        retriever=retriever
    )