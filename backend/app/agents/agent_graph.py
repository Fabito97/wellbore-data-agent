"""
Mini LangGraph Agent - Simple Q&A Only

Flow:
1. LLM decides: list wells, get well info, or search?
2. If search: LLM provides search params (query, filters)
3. Execute search
4. LLM generates answer
"""
from typing import TypedDict, Literal, Optional, Dict, Any, List
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import json

from app.agents.tools.rag_tool import list_available_wells, get_well_by_name, rag_query_tool
from app.rag.retriever import get_retriever
from app.services.llm_service import get_llm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== Pydantic Schemas ====================

class AgentDecision(BaseModel):
    """LLM decides what action to take."""
    action: Literal["list_wells", "get_well", "search", "answer"] = Field(
        description="What to do next"
    )
    well_name: Optional[str] = Field(description="Well name if needed (e.g., 'well-1')")
    reasoning: str = Field(description="Why this action?")


class SearchParameters(BaseModel):
    """LLM provides search parameters."""
    query: str = Field(description="Optimized search query")
    well_name: Optional[str] = Field(description="Filter by well (e.g., 'well-1')")
    document_id: Optional[str] = Field(description="Filter by specific document ID")
    document_type: Optional[str] = Field(description="Filter by type (e.g., 'WELL_REPORT')")
    chunk_type: Optional[str] = Field(description="Filter by chunk type ('text' or 'table')")
    top_k: int = Field(description="Number of results", default=5)


# ==================== State ====================

class MiniAgentState(TypedDict):
    """State flowing through graph."""
    # User input
    question: str

    # LLM decisions
    action: str  # "list_wells", "get_well", "search", "answer"
    well_name: Optional[str]

    # Search params (if searching)
    search_params: Optional[Dict[str, Any]]

    # Retrieved data
    search_results: Optional[str]
    well_info: Optional[str]
    wells_list: Optional[str]

    # Output
    answer: str

    # Control
    iteration: int


# ==================== Nodes ====================

def llm_decision_node(state: MiniAgentState) -> MiniAgentState:
    """
    LLM decides what to do based on question and current state.
    """
    logger.info("=== LLM DECISION NODE ===")

    question = state["question"]
    iteration = state.get("iteration", 0)

    # Build context from what we know
    context = []
    if state.get("wells_list"):
        context.append(f"Available wells: {state['wells_list']}")
    if state.get("well_info"):
        context.append(f"Well info: {state['well_info']}")
    if state.get("search_results"):
        context.append(f"Search results: {state['search_results'][:500]}...")

    context_str = "\n".join(context) if context else "No information gathered yet."

    # Create prompt
    llm = get_llm_service().llm
    parser = PydanticOutputParser(pydantic_object=AgentDecision)

    prompt = f"""You are helping answer a user's question about well data.

User Question: {question}

Current Information:
{context_str}

Decide what to do next:

1. "list_wells" - If user didn't specify a well OR you need to see available wells
2. "get_well" - If you have a well name but need to verify it exists
3. "search" - If you have a well name and are ready to search documents
4. "answer" - If you have enough information to answer the question

CRITICAL: 
- If no well mentioned in question, you MUST list_wells first
- If well mentioned but not verified, you MUST get_well first
- Only search when you're sure which well to query

{parser.get_format_instructions()}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        decision = parser.parse(content)

        logger.info(f"LLM Decision: {decision.action} (well: {decision.well_name})")
        logger.info(f"Reasoning: {decision.reasoning}")

        return {
            **state,
            "action": decision.action,
            "well_name": decision.well_name,
            "iteration": iteration + 1
        }

    except Exception as e:
        logger.error(f"LLM decision failed: {e}")

        # Fallback: Simple logic
        if not state.get("wells_list"):
            action = "list_wells"
        elif state.get("search_results"):
            action = "answer"
        else:
            action = "search"

        return {
            **state,
            "action": action,
            "iteration": iteration + 1
        }


def list_wells_node(state: MiniAgentState) -> MiniAgentState:
    """List all available wells."""
    logger.info("=== LIST WELLS NODE ===")

    try:
        wells = list_available_wells.invoke({})

        wells_text = ""
        for well in wells:
            name = well.get('name', 'unknown')
            count = well.get('document_count', 0)
            wells_text += f"{name} ({count} docs), "

        wells_text = wells_text.rstrip(", ")

        logger.info(f"Listed {len(wells)} wells")

        return {
            **state,
            "wells_list": wells_text
        }
    except Exception as e:
        logger.error(f"List wells failed: {e}")
        return {
            **state,
            "wells_list": f"Error: {str(e)}"
        }


def get_well_node(state: MiniAgentState) -> MiniAgentState:
    """Get info about specific well."""
    logger.info("=== GET WELL NODE ===")

    well_name = state.get("well_name")

    if not well_name:
        return {
            **state,
            "well_info": "Error: No well name provided"
        }

    try:
        well_data = get_well_by_name.invoke({"well_name": well_name})

        if well_data:
            docs = well_data.get("documents", [])
            doc_types = [d.get("document_type", "unknown") for d in docs]

            well_text = f"Well {well_name}: {len(docs)} documents"
            well_text += f" (types: {', '.join(set(doc_types))})"

            logger.info(f"Got well info: {well_name}")

            return {
                **state,
                "well_info": well_text
            }
        else:
            return {
                **state,
                "well_info": f"Well '{well_name}' not found"
            }
    except Exception as e:
        logger.error(f"Get well failed: {e}")
        return {
            **state,
            "well_info": f"Error: {str(e)}"
        }


def llm_search_params_node(state: MiniAgentState) -> MiniAgentState:
    """
    LLM provides search parameters.
    """
    logger.info("=== LLM SEARCH PARAMS NODE ===")

    question = state["question"]
    well_name = state.get("well_name")

    llm = get_llm_service().llm
    parser = PydanticOutputParser(pydantic_object=SearchParameters)

    prompt = f"""Generate search parameters to answer this question.

User Question: {question}
Well Name: {well_name or "Not specified"}

Provide optimized search parameters:

1. query: Core search terms (optimize for retrieval)
2. well_name: Which well to search (use '{well_name}' if known)
3. document_id: Specific doc ID (usually null unless user mentions it)
4. document_type: Use 'WELL_REPORT' for completion data, or null for all types
5. chunk_type: Use 'table' for technical/numerical queries, 'text' for descriptive, or null for both
6. top_k: Usually 5, increase to 10 for broad questions

Examples:
- "What's the tubing depth?" → query: "tubing depth", chunk_type: "table"
- "Tell me about the reservoir" → query: "reservoir properties", chunk_type: null
- "What's in the completion report?" → query: "completion", document_type: "WELL_REPORT"

{parser.get_format_instructions()}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        params = parser.parse(content)

        # Convert to dict
        params_dict = {
            "query": params.query,
            "well_name": params.well_name or well_name,
            "document_id": params.document_id,
            "document_type": params.document_type,
            "chunk_type": params.chunk_type,
            "top_k": params.top_k
        }

        logger.info(f"Search params: query='{params.query}', chunk_type={params.chunk_type}")

        return {
            **state,
            "search_params": params_dict
        }

    except Exception as e:
        logger.error(f"Search params generation failed: {e}")

        # Fallback: Basic params
        return {
            **state,
            "search_params": {
                "query": question,
                "well_name": well_name,
                "document_id": None,
                "document_type": "WELL_REPORT",
                "chunk_type": None,
                "top_k": 5
            }
        }


def search_node(state: MiniAgentState) -> MiniAgentState:
    """Execute search with provided parameters."""
    logger.info("=== SEARCH NODE ===")

    params = state.get("search_params")

    if not params:
        return {
            **state,
            "search_results": "Error: No search parameters"
        }

    try:
        # Call RAG tool
        result = rag_query_tool.invoke(params)

        results_list = result.get("results", [])

        if not results_list:
            return {
                **state,
                "search_results": f"No results found for query '{params.get('query')}'"
            }

        # Format with retriever
        retriever = get_retriever()
        formatted = retriever.format_context_for_llm(results_list, max_tokens=2000)

        logger.info(f"Search returned {len(results_list)} results")

        return {
            **state,
            "search_results": formatted
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {
            **state,
            "search_results": f"Error: {str(e)}"
        }


def answer_node(state: MiniAgentState) -> MiniAgentState:
    """LLM generates final answer."""
    logger.info("=== ANSWER NODE ===")

    question = state["question"]
    search_results = state.get("search_results", "")
    wells_list = state.get("wells_list", "")

    llm = get_llm_service().llm

    # Build context
    context = []
    if wells_list:
        context.append(f"Available wells: {wells_list}")
    if search_results:
        context.append(f"Retrieved information:\n{search_results}")

    context_str = "\n\n".join(context) if context else "No information available."

    prompt = f"""Answer the user's question based on the information below.

User Question: {question}

Information:
{context_str}

Provide a clear, direct answer. If information is missing, say so clearly.
Include specific values, numbers, and cite sources when possible.

Answer:"""

    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        logger.info("Generated answer")

        return {
            **state,
            "answer": answer.strip()
        }

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return {
            **state,
            "answer": f"Error generating answer: {str(e)}"
        }


# ==================== Routing ====================

def route_action(state: MiniAgentState) -> Literal["list_wells", "get_well", "search_params", "answer", "end"]:
    """Route based on LLM's decision."""

    action = state.get("action", "list_wells")
    iteration = state.get("iteration", 0)

    # Safety: prevent infinite loops
    if iteration >= 10:
        logger.warning("Max iterations reached")
        return "end"

    logger.info(f"Routing to: {action}")

    if action == "list_wells":
        return "list_wells"
    elif action == "get_well":
        return "get_well"
    elif action == "search":
        return "search_params"
    elif action == "answer":
        return "answer"
    else:
        return "end"


# ==================== Build Graph ====================

def create_mini_agent():
    """
    Create mini LangGraph agent.

    Flow:
        START
          ↓
        llm_decision ←─────────┐
          ↓                    │
        [route by action]      │
          ├─ list_wells ───────┘ (loop back)
          ├─ get_well ─────────┘ (loop back)
          ├─ search_params     │
          │    ↓               │
          │  search ───────────┘ (loop back)
          ├─ answer → END
          └─ end → END
    """
    workflow = StateGraph(MiniAgentState)

    # Add nodes
    workflow.add_node("llm_decision", llm_decision_node)
    workflow.add_node("list_wells", list_wells_node)
    workflow.add_node("get_well", get_well_node)
    workflow.add_node("search_params", llm_search_params_node)
    workflow.add_node("search", search_node)
    workflow.add_node("answer", answer_node)

    # Entry point
    workflow.set_entry_point("llm_decision")

    # Route based on LLM decision
    workflow.add_conditional_edges(
        "llm_decision",
        route_action,
        {
            "list_wells": "list_wells",
            "get_well": "get_well",
            "search_params": "search_params",
            "answer": "answer",
            "end": END
        }
    )

    # After actions, go back to LLM decision
    workflow.add_edge("list_wells", "llm_decision")
    workflow.add_edge("get_well", "llm_decision")

    # Search flow
    workflow.add_edge("search_params", "search")
    workflow.add_edge("search", "llm_decision")

    # Answer ends
    workflow.add_edge("answer", END)

    return workflow.compile()


# ==================== Agent Wrapper ====================

class MiniLangGraphAgent:
    """
    Mini LangGraph agent for simple Q&A.

    LLM decides:
    - When to list wells
    - When to get well info
    - What search parameters to use
    - When to provide final answer
    """

    def __init__(self):
        self.graph = create_mini_agent()
        logger.info("✓ Mini LangGraph Agent initialized")

    def ask(self, question: str) -> str:
        """
        Ask a question.

        Args:
            question: User's question

        Returns:
            Answer string
        """
        logger.info("=" * 80)
        logger.info(f"Question: {question}")
        logger.info("=" * 80)

        initial_state: MiniAgentState = {
            "question": question,
            "action": "",
            "well_name": None,
            "search_params": None,
            "search_results": None,
            "well_info": None,
            "wells_list": None,
            "answer": "",
            "iteration": 0
        }

        try:
            final_state = self.graph.invoke(initial_state)

            answer = final_state.get("answer", "No answer generated")

            logger.info(f"✓ Complete after {final_state.get('iteration', 0)} iterations")

            return answer

        except Exception as e:
            logger.error(f"✗ Failed: {e}", exc_info=True)
            return f"Error: {str(e)}"

    def stream_ask(self, question: str):
        """
        Ask question with streaming (see each step).

        Usage:
            for step in agent.stream_ask("What's the tubing depth?"):
                print(step)
        """
        initial_state: MiniAgentState = {
            "question": question,
            "action": "",
            "well_name": None,
            "search_params": None,
            "search_results": None,
            "well_info": None,
            "wells_list": None,
            "answer": "",
            "iteration": 0
        }

        for output in self.graph.stream(initial_state):
            yield output


# ==================== Interactive Terminal ====================

def interactive_mode():
    """Run agent in interactive terminal."""
    print("\n" + "=" * 80)
    print("MINI LANGGRAPH AGENT - Interactive Mode")
    print("=" * 80)
    print("\nLLM-driven Q&A agent. Type 'exit' to quit.\n")

    agent = MiniLangGraphAgent()

    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break

            print("\nAgent: ", end="", flush=True)
            answer = agent.ask(question)
            print(answer)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def test_mode():
    """Test with sample questions."""
    print("\n" + "=" * 80)
    print("Testing Mini LangGraph Agent")
    print("=" * 80)

    agent = MiniLangGraphAgent()

    test_questions = [
        "List all wells",
        "What is the tubing depth for well-1?",
        "What's the API gravity?",  # No well specified
        "Tell me about the reservoir in well-1",
    ]

    for question in test_questions:
        print("\n" + "-" * 80)
        print(f"Q: {question}")
        print("-" * 80)

        answer = agent.ask(question)
        print(f"A: {answer}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_mode()
    else:
        interactive_mode()