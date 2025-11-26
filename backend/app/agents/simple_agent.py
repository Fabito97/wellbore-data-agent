"""
Simple Agent with RAG Tools using LangChain create_agent.

The agent can:
1. Detect wells from queries
2. Search documents with well-scoping
3. Summarize well reports
4. Extract parameters

Uses modern LangChain create_agent (no AgentExecutor needed).
"""
from langchain_classic.agents import create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.utils.logger import get_logger
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from fastapi import Depends
import json

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from app.services.llm_service import get_llm_service, LLMService
from app.services.conversation_service import get_conversation_service, ConversationService
from app.rag.retriever import get_retriever, DocumentRetriever
from app.core.config import settings
from langchain.agents import create_agent
# Import RAG tool functions
from app.agents.tools.rag_tool import (
    detect_well_from_query,
    list_available_wells,
    rag_query_tool,
    summarize_well_report_tool,
    extract_parameters_tool,
)
from app.utils.prompts import SYSTEM_PROMPT, AGENT_PROMPT

logger = get_logger(__name__)


# ==================== Define LangChain Tools ====================

# @tool
# def detect_well(query: str) -> str:
#     """
#     Detect which well the user is asking about from their query.
#
#     Input: User's question
#     Output: Well name (e.g., 'well-4') or 'None' if not detected
#
#     Use this FIRST to understand which well to query.
#     """
#     result = detect_well_from_query(query)
#     return result if result else "None"
#

@tool
def list_wells(_: str = "") -> str:
    """
    List all available wells in the system.

    Input: Empty string (not used)
    Output: JSON list of wells

    Use when user asks "what wells exist" or when well detection fails.
    """
    # rag_tool functions are decorated with @tool; call the wrapped function
    func = getattr(list_available_wells, "__wrapped__", list_available_wells)
    wells = func()
    return json.dumps(wells, indent=2)


@tool
def search_documents(query: str) -> str:
    """
    Search well documents with well-scoping.

    Input format: "query|||well_name" (three pipes)
    Example: "tubing depth|||well-4"

    Output: Relevant document chunks with sources

    Use for specific questions about well data.
    """
    try:
        # parts = query_with_well.split("|||")
        # query = parts[0].strip()
        # well_name = parts[1].strip() if len(parts) > 1 else None
        logger.info("[Tool_call] Calling search document tools")
        func = getattr(rag_query_tool, "__wrapped__", rag_query_tool)
        result = func(query=query, top_k=10)

        if result["count"] == 0:
            return f"No results found for '{query}'"

        # Format results
        formatted = f"Found {result['count']} results:\n\n"
        for i, chunk in enumerate(result["results"][:3], 1):  # Top 3
            formatted += f"{i}. {chunk.content[:200]}...\n"
            formatted += f"   Source: {chunk.filename}, page {chunk.page_number}\n\n"

        return formatted

    except Exception as e:
        return f"Error searching: {str(e)}"


@tool
def get_summary_context(well_name: str) -> str:
    """
    Get context to summarize a well report.

    Input: Well name (e.g., 'well-4')
    Output: Document context for summarization

    Use when user asks to "summarize" or "describe" a well.
    """
    func = getattr(summarize_well_report_tool, "__wrapped__", summarize_well_report_tool)
    result = func(well_name)

    if result["chunks_used"] == 0:
        return f"No documents found for {well_name}"

    return result["context"]


@tool
def get_parameter_context(well_name: str) -> str:
    """
    Get tables and text for parameter extraction.

    Input: Well name (e.g., 'well-4')
    Output: Tables and text with parameter data

    Use when extracting: tubing specs, reservoir pressure, oil gravity, PI, etc.
    """
    func = getattr(extract_parameters_tool, "__wrapped__", extract_parameters_tool)
    result = func(well_name)

    if result["total_chunks"] == 0:
        return f"No parameter data found for {well_name}"

    return result["context"]


# ==================== Agent Response Model ====================

@dataclass
class AgentResponse(BaseModel):
    """Response from agent."""
    answer: str
    sources: List[str]  # Simplified - just source names
    # conversation_id: str
    confidence: str
    well_name: Optional[str] = None
    tool_calls: int = 0
    # token_usage: Optional[Dict[str, int]] = None

# ==================== Simple Agent with Tools ====================

class SimpleAgent:
    """
    Agent with RAG tools using LangChain create_agent.

    The agent autonomously:
    - Detects wells from queries
    - Searches documents with well-scoping
    - Generates contextual answers
    """

    def __init__(
        self,
        llm_service: LLMService,
        # conversation_service: ConversationService,
        retriever: DocumentRetriever
    ):
        self.llm_service = llm_service
        # self.conversations = conversation_service
        self.retriever = retriever


        # Define tools
        self.tools = [
            # detect_well,
            list_wells,
            search_documents,
            get_summary_context,
            get_parameter_context
        ]
        # Create LangChain LLM
        # Create the base Chat model instance. Do NOT call .bind_tools() here –
        # create_agent expects a `str` (model name) or a BaseChatModel instance.
        # .bind_tools() returns a Runnable which causes the type error seen.
        self.llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.1  # Deterministic for tool usage
        )
        self.parser = PydanticOutputParser(pydantic_object=AgentResponse)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a research assistant that will help generate a research paper.
                    Answer the user query and use necessary tools. 
                    Wrap the output in this format and provide no other text\n{format_instructions}
                    It is important to use the date tool to get exact date for queries when necessary
                    Always output valid json
                    """,
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}")
            ]
        ).partial(format_instructions=self.parser.get_format_instructions())

        # Create agent
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )



        logger.info("SimpleAgent initialized with RAG tools")

    def answer(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Answer question using agent with tools.

        The agent will:
        1. Analyze the question
        2. Decide which tools to use
        3. Call tools autonomously
        4. Generate final answer
        """
        logger.info(f"Agent processing: '{question[:100]}...'")

        # # Get conversation
        # conversation = self.conversations.get_or_create_conversation(conversation_id)
        # self.conversations.add_message(conversation.id, "user", question)

        try:
            # Invoke agent
            result = self.agent.invoke({'query': question})

            # Extract answer from agent response
            # answer = self._extract_answer(result)
            response = self.parser.parse(result.get("output"))
            logger.info(f"Response: {response}")
            # Detect well from original question (for metadata)
            # well_name = detect_well_from_query(query=question)

            # Count tool calls
            # tool_calls = self._count_tool_calls(result)

            # Save response
            # self.conversations.add_message(conversation.id, "assistant", answer)

            # Assess confidence
            # confidence = self._assess_confidence(answer, tool_calls)

            # logger.info(f"Agent completed: {tool_calls} tool calls, confidence={confidence}")

            return AgentResponse(
                answer=response.answer,
                sources=response.sources or [],  # Simplified for now
                # conversation_id=conversation.id,
                confidence=response.confidence,
                well_name=response.well_name,
                tool_calls=response.tool_calls
            )

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            error_msg = f"I encountered an error: {str(e)}"
            # self.conversations.add_message(conversation.id, "assistant", error_msg)

            return AgentResponse(
                answer=error_msg,
                sources=[],
                # conversation_id=conversation.id,
                confidence="low"
            )

    def _extract_answer(self, result: Dict[str, Any]) -> str:
        """Extract final answer from agent result."""
        # Agent result contains messages
        messages = result.get("messages", [])

        # Last message should be the answer
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                return last_msg.content
            elif isinstance(last_msg, dict):
                return last_msg.get("content", "No response generated")

        return "No response generated"

    def _count_tool_calls(self, result: Dict[str, Any]) -> int:
        """Count how many tools were called."""
        messages = result.get("messages", [])
        tool_calls = 0

        for msg in messages:
            if hasattr(msg, "tool_calls"):
                tool_calls += len(msg.tool_calls)
            elif isinstance(msg, dict) and msg.get("tool_calls"):
                tool_calls += len(msg["tool_calls"])

        return tool_calls

    def _assess_confidence(self, answer: str, tool_calls: int) -> str:
        """Assess confidence based on tool usage."""
        if tool_calls == 0:
            return "low"  # No tools used
        elif tool_calls >= 2:
            return "high"  # Multiple tools used
        else:
            return "medium"  # One tool used


# ==================== FastAPI Dependency ====================

def get_simple_agent(
    llm_service: LLMService = Depends(get_llm_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
    retriever: DocumentRetriever = Depends(get_retriever)
) -> SimpleAgent:
    """FastAPI dependency to get agent."""
    return SimpleAgent(
        llm_service=llm_service,
        # conversation_service=conversation_service,
        retriever=retriever
    )


# ==================== Test Function ====================

def test_agent():
    """Test agent with sample queries."""
    from app.services.llm_service import get_llm_service
    from app.services.conversation_service import get_conversation_service
    from app.rag.retriever import get_retriever
    from app.core.database import SessionLocal

    # Initialize
    db = SessionLocal()
    agent = SimpleAgent(
        llm_service=get_llm_service(),
        # conversation_service=get_conversation_service(db),
        retriever=get_retriever()
    )

    print("=" * 80)
    print("TESTING AGENT WITH TOOLS")
    print("=" * 80)

    # Test 1: Well detection + search
    print("\n📋 Test 1: Simple Query")
    print("-" * 80)
    response = agent.answer("What is the tubing depth for Well 1?")
    print(f"Answer: {response.answer}")
    print(f"Well detected: {response.well_name}")
    print(f"Tool calls: {response.tool_calls}")
    print(f"Confidence: {response.confidence}")

    # Test 2: List wells
    print("\n📋 Test 2: List Wells")
    print("-" * 80)
    response = agent.answer("What wells do we have?")
    print(f"Answer: {response.answer}")
    print(f"Tool calls: {response.tool_calls}")

    # Test 3: Summarization
    print("\n📝 Test 3: Summarization")
    print("-" * 80)
    response = agent.answer("Summarize Well 4")
    print(f"Answer: {response.answer[:300]}...")
    print(f"Tool calls: {response.tool_calls}")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_agent()