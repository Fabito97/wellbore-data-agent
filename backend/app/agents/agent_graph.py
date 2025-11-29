"""
Efficient Single-Node Agent

Key Improvement: ONE LLM call per interaction
- LLM decides everything at once
- Can answer directly OR provide search params
- No separate classification/strategy nodes
- Optimized for speed AND accuracy

Flow:
User → [LLM Agent] → Answer OR Search → Answer
     ONE call          (if needed)  ONE call
"""
from typing import TypedDict, Literal, Optional, Dict, Any, List, Sequence, Union
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.agents.tools.rag_tool import rag_query_tool
from app.core.config import settings
from app.rag.retriever import get_retriever
from app.services.llm_service import get_llm_service, LLMProvider
from app.utils.logger import get_logger

name = "Agent" if __name__ == "__main__" else __name__
logger = get_logger(name)


# ==================== Domain Knowledge ====================

DOMAIN_CONTEXT = """
SYSTEM CONTEXT:
You are analyzing well completion data from oil & gas operations.

DOCUMENT TYPES:
- WELL_REPORT: Completion specs (tubing, casing, depths, reservoir data)
- PVT: Fluid properties (API gravity, viscosity, GOR)
- PRODUCTION: Production rates and test data

SEARCH PARAMETERS:
- document_type: Use "WELL_REPORT" for completion data (default)
- chunk_type: Use "table" for technical specs/numbers, "text" for descriptions
- well_name: Always required for specific well queries (format: "well-1")
"""


# ==================== Single Response Schema ====================

class AgentAction(BaseModel):
    """LLM decides everything in ONE call."""

    action: Literal["answer", "search", "clarify"] = Field(
        description="What to do: answer directly, search documents, or ask for clarification"
    )

    # For direct answers
    direct_answer: Optional[str] = Field(
        default=None,
        description="If action='answer', provide the answer here"
    )

    # For searches
    search_query: Optional[str] = Field(
        default=None,
        description="If action='search', the search query"
    )
    well_name: Optional[str] = Field(
        default=None,
        description="If action='search', which well to search"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="If action='search', document type (WELL_REPORT, PVT_REPORT, etc.)"
    )
    chunk_type: Optional[str] = Field(
        default=None,
        description="If action='search', chunk type ('table', 'text', or null)"
    )

    # For clarifications
    clarification_question: Optional[str] = Field(
        default=None,
        description="If action='clarify', what to ask user"
    )

    reasoning: str = Field(description="Why this action?")


# ==================== State ====================

class AgentState(TypedDict):
    """Minimal state."""
    question: str
    messages: Sequence[BaseMessage]

    # Agent decision
    action: str
    search_params: Optional[Dict[str, Any]]

    # Results
    search_results: Optional[str]
    response: str


# ==================== Single Agent Node ====================

def agent_decision_node(state: AgentState) -> AgentState:
    """
    ONE LLM call that decides everything.

    Can:
    1. Answer directly (greetings, general questions, known info)
    2. Provide search parameters (technical queries)
    3. Ask for clarification (ambiguous questions)
    """
    logger.info("=== AGENT THINKING ===")

    question = state["question"]
    messages = state.get("messages", [])

    # Build conversation context
    history = ""
    if len(messages) > 1:
        history = "\nRecent conversation:\n"
        for msg in messages[-4:]:
            role = "User" if isinstance(msg, HumanMessage) else "You"
            history += f"{role}: {msg.content[:150]}\n"

    groq_chat_model = get_llm_service(
        provider=LLMProvider.GROQ,
        api_key=settings.GROQ_API_KEY
    )
    ollama_chat_model = get_llm_service()
    llm = groq_chat_model.llm

    parser = PydanticOutputParser(pydantic_object=AgentAction)

    prompt = f"""{DOMAIN_CONTEXT}

{history}

User: {question}

Decide what to do:

1. ACTION='answer' - Answer directly if:
   - Greeting (hello, hi, how are you)
   - General question (what can you do, help)
   - Example: "Hello" → direct_answer: "Hello! I can help with well completion data..."

2. ACTION='search' - Search documents if:
   - User asks for specific technical data
   - Provide complete search parameters:
     * search_query: Optimized terms (e.g., "tubing inner diameter depth") - Always required
     * well_name: Extract from question (e.g., "well-1") - Always required
     * document_type: "WELL_REPORT" (default) or "PVT_REPORT" for fluid data
     * chunk_type: "table" for specs/numbers, "text" for descriptions, null for both
   - Example: "What's the tubing depth for well-1?"
     → search_query: "tubing depth", well_name: "well-1", chunk_type: "table"

3. ACTION='clarify' - Ask for clarification if:
   - Question is ambiguous (e.g., "What's the pressure?" - which pressure?)
   - Missing well name for specific query
   - Example: "What's the tubing depth?" → "Which well would you like information about?"

CRITICAL:
- Use conversation history for context (if user mentioned well before, use it)
- Be smart about what needs search vs direct answer
- For technical specs, prioritize chunk_type="table"
- Provide ALL search parameters in one shot

{parser.get_format_instructions()}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        decision = parser.parse(content)

        logger.info(f"Decision: {decision.action} - {decision.reasoning}")

        # Route based on decision
        if decision.action == "answer":
            # Direct answer - done!
            messages_updated = list(messages)
            messages_updated.append(AIMessage(content=decision.direct_answer))

            return {
                **state,
                "action": "answer",
                "response": decision.direct_answer,
                "messages": messages_updated
            }

        elif decision.action == "search":
            # Need to search - save params
            search_params = {
                "query": decision.search_query,
                "well_name": decision.well_name,
                "document_type": decision.document_type or "WELL_REPORT",
                "chunk_type": decision.chunk_type,
                "top_k": 5
            }

            logger.info(
                f"Search: query='{decision.search_query}', "
                f"well={decision.well_name}, chunk_type={decision.chunk_type}"
            )

            return {
                **state,
                "action": "search",
                "search_params": search_params
            }

        else:  # clarify
            messages_updated = list(messages)
            messages_updated.append(AIMessage(content=decision.clarification_question))

            return {
                **state,
                "action": "clarify",
                "response": decision.clarification_question,
                "messages": messages_updated
            }

    except Exception as e:
        logger.error(f"Agent decision failed: {e}", exc_info=True)

        # Fallback: Try to extract well name and search
        import re
        match = re.search(r'well[\s\-_]*(\d+)', question.lower())

        if match:
            num = match.group(1).lstrip('0') or '0'
            well_name = f"well-{num}"

            return {
                **state,
                "action": "search",
                "search_params": {
                    "query": question,
                    "well_name": well_name,
                    "document_type": "WELL_REPORT",
                    "chunk_type": None,
                    "top_k": 5
                }
            }
        else:
            # No well found - ask
            fallback = "Which well would you like information about?"
            messages_updated = list(messages)
            messages_updated.append(AIMessage(content=fallback))

            return {
                **state,
                "action": "clarify",
                "response": fallback,
                "messages": messages_updated
            }


def search_node(state: AgentState) -> AgentState:
    """Execute search (only called if action='search')."""
    logger.info("=== SEARCHING ===")

    params = state.get("search_params")

    try:
        result = rag_query_tool.invoke(params)
        results_list = result.get("results", [])

        if not results_list:
            return {
                **state,
                "search_results": f"No results found for '{params['query']}' in {params['well_name']}"
            }

        retriever = get_retriever()
        formatted = retriever.format_context_for_llm(results_list, max_tokens=2000)

        logger.info(f"Found {len(results_list)} results")

        return {
            **state,
            "search_results": formatted
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {
            **state,
            "search_results": f"Search error: {str(e)}"
        }


def answer_node(state: AgentState) -> AgentState:
    """
    Generate answer from search results.
    ONE LLM call.
    """
    logger.info("=== ANSWERING ===")

    question = state["question"]
    search_results = state.get("search_results", "")
    messages = state.get("messages", [])

    llm = get_llm_service().llm

    prompt = f"""{DOMAIN_CONTEXT}

User asked: "{question}"

Retrieved information:
{search_results}

Provide a clear, accurate answer:
- Extract the specific value/information requested
- Include units and values
- Cite sources
- Use proper oil & gas terminology
- If data missing, say so honestly

Answer:"""

    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        messages_updated = list(messages)
        messages_updated.append(AIMessage(content=answer))

        return {
            **state,
            "response": answer.strip(),
            "messages": messages_updated
        }

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")

        error_msg = f"I found information but couldn't process it: {str(e)}"
        messages_updated = list(messages)
        messages_updated.append(AIMessage(content=error_msg))

        return {
            **state,
            "response": error_msg,
            "messages": messages_updated
        }


# ==================== Routing ====================

def route_action(state: AgentState) -> Literal["done", "search"]:
    """Simple routing based on agent's decision."""
    action = state.get("action", "done")

    if action == "search":
        return "search"
    else:  # "answer" or "clarify"
        return "done"


# ==================== Build Graph ====================

def create_efficient_agent():
    """
    Minimal graph:
    - 1 LLM call if answering directly
    - 2 LLM calls if search needed
    """
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("agent", agent_decision_node)
    workflow.add_node("search", search_node)
    workflow.add_node("answer", answer_node)

    # Entry
    workflow.set_entry_point("agent")

    # Route based on agent decision
    workflow.add_conditional_edges(
        "agent",
        route_action,
        {
            "search": "search",
            "done": END
        }
    )

    # Search → Answer → Done
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()


# ==================== Agent ====================

class EfficientAgent:
    """
    Efficient agent - minimal LLM calls.

    Calls:
    - Greeting/General: 1 LLM call
    - Technical query: 2 LLM calls (decision + answer)
    """

    def __init__(self):
        self.graph = create_efficient_agent()
        self.conversation_history: List[BaseMessage] = []
        logger.info("✓ Agent initialized")

    def ask(self, question: str, reset: bool = False) -> str:
        """Ask a question."""

        if reset:
            self.conversation_history = []

        logger.info("=" * 80)
        logger.info(f"Q: {question}")
        logger.info("=" * 80)

        self.conversation_history.append(HumanMessage(content=question))

        initial_state: AgentState = {
            "question": question,
            "messages": self.conversation_history,
            "action": "",
            "search_params": None,
            "search_results": None,
            "response": ""
        }

        try:
            final_state = self.graph.invoke(initial_state)

            self.conversation_history = list(final_state.get("messages", []))

            response = final_state.get("response", "No response generated")

            logger.info(f"✓ Complete")
            return response

        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)
            error_msg = f"Error: {str(e)}"
            self.conversation_history.append(AIMessage(content=error_msg))
            return error_msg

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("History cleared")


# ==================== Interactive ====================

def interactive_mode():
    """Interactive mode."""
    print("\n" + "=" * 80)
    print("EFFICIENT WELL ANALYSIS AGENT")
    print("=" * 80)
    print("\nOptimized for speed and accuracy!")
    print("Commands: 'reset', 'exit'\n")

    agent = EfficientAgent()

    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break

            if question.lower() == 'reset':
                agent.clear_history()
                print("\n✓ Conversation reset")
                continue

            print("\nAgent: ", end="", flush=True)
            answer = agent.ask(question)
            print()
            print(answer)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    interactive_mode()