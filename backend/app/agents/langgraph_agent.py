"""
LangGraph Master Agent - Primary Version

Uses existing agents/tools with direct function calls.
No JSON parsing needed - fast and reliable.
"""
from typing import TypedDict, Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, END
import json

from app.agents.tools.rag_tool import list_available_wells, get_well_by_name, rag_query_tool
from app.agents.tools.summarization_tool import summarize_well
from app.agents.tools.extraction_tool import extract_parameters
from app.rag.retriever import get_retriever
from app.services.llm_service import get_llm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== State Definition ====================

class AgentState(TypedDict):
    """State that flows through the graph."""
    # Input
    query: str
    original_query: str  # Preserve for sub-agents

    # Intent classification
    intent: str  # "list", "info", "search", "summarize", "extract"

    # Extracted entities
    well_name: Optional[str]
    search_query: Optional[str]

    # Data
    well_info: Optional[str]

    # Output
    output: str
    error: Optional[str]


# ==================== Helper: Extract Well Name ====================

def extract_well_name_from_query(query: str) -> Optional[str]:
    """Extract well name from query using simple regex."""
    import re

    match = re.search(r'well[\s\-_]*(\d+)', query.lower())
    if match:
        num = match.group(1).lstrip('0') or '0'
        return f"well-{num}"
    return None


# ==================== Nodes ====================

def classify_intent_node(state: AgentState) -> AgentState:
    """
    Classify user intent using LLM (simple, no JSON).
    """
    logger.info("=== CLASSIFY INTENT ===")

    query = state["query"].lower()

    # Simple LLM prompt - no JSON, just return intent word
    llm = get_llm_service().llm

    prompt = f"""Classify this query into ONE of these intents:
- list: User wants to see available wells
- info: User wants information about a specific well
- search: User has a specific question about well data
- summarize: User wants a summary of well documents
- extract: User wants to extract parameters for nodal analysis
- analyze: User wants to extract parameters for nodal analysis

Query: {state["query"]}

Intent (just the word):"""

    response = llm.invoke(prompt)
    intent = response.content if hasattr(response, 'content') else str(response)
    intent = intent.strip().lower()

    # Validate intent
    valid_intents = ["list", "info", "search", "summarize", "extract"]
    if intent not in valid_intents:
        # Fallback to keyword matching
        if "list" in query or "available" in query:
            intent = "list"
        elif "summarize" in query or "summary" in query:
            intent = "summarize"
        elif "extract" in query or "parameter" in query:
            intent = "extract"
        else:
            intent = "search"

    # Extract well name if present
    well_name = extract_well_name_from_query(query)

    logger.info(f"Intent: {intent}, Well: {well_name}")

    return {
        **state,
        "intent": intent,
        "well_name": well_name,
        "original_query": state["query"]
    }


def list_wells_node(state: AgentState) -> AgentState:
    """List all available wells."""
    logger.info("=== LIST WELLS ===")

    try:
        result = list_available_wells.invoke({})

        if isinstance(result, list):
            output = "Available wells:\n"
            for well in result:
                name = well.get('name', 'unknown')
                count = well.get('document_count', 0)
                output += f"  • {name} ({count} documents)\n"
        else:
            output = str(result)

        return {**state, "output": output}
    except Exception as e:
        return {**state, "output": f"Error: {str(e)}", "error": str(e)}


def get_well_info_node(state: AgentState) -> AgentState:
    """Get info about specific well."""
    logger.info("=== GET WELL INFO ===")

    well_name = state.get("well_name")

    if not well_name:
        return {
            **state,
            "output": "Please specify which well you'd like information about.",
            "error": "No well name"
        }

    try:
        result = get_well_by_name.invoke({"well_name": well_name})

        if result:
            name = result.get('name', 'unknown')
            count = result.get('document_count', 0)
            docs = result.get('documents', [])

            output = f"Well: {name}\n"
            output += f"Documents: {count}\n\n"
            output += "Document types:\n"
            for doc in docs:
                output += f"  • {doc.get('document_type', 'unknown')}: {doc.get('filename', 'unknown')}\n"
        else:
            output = f"Well '{well_name}' not found."

        return {**state, "output": output, "well_info": output}
    except Exception as e:
        return {**state, "output": f"Error: {str(e)}", "error": str(e)}


def search_documents_node(state: AgentState) -> AgentState:
    """Search documents for specific information."""
    logger.info("=== SEARCH DOCUMENTS ===")

    well_name = state.get("well_name")
    query = state.get("original_query")

    if not well_name:
        return {
            **state,
            "output": "Please specify which well you'd like to search.",
            "error": "No well name"
        }

    try:
        # Extract search query from original query
        import re
        search_query = re.sub(r'well[\s\-_]*\d+', '', query, flags=re.IGNORECASE)
        search_query = re.sub(r'for|about|in|the', '', search_query, flags=re.IGNORECASE).strip()

        result = rag_query_tool.invoke({
            "query": search_query,
            "well_name": well_name,
            "top_k": 5
        })

        results_list = result.get("results", [])

        if not results_list:
            output = f"No results found for your query in {well_name}"
        else:
            # Format with retriever
            retriever = get_retriever()
            context = retriever.format_context_for_llm(results_list, max_tokens=1500)

            # Use LLM to extract answer from context
            llm = get_llm_service().llm

            prompt = f"""Answer this question based on the information provided.

Question: {query}

Information:
{context}

Answer (be specific and concise):"""

            response = llm.invoke(prompt)
            output = response.content if hasattr(response, 'content') else str(response)

        return {**state, "output": output}
    except Exception as e:
        return {**state, "output": f"Error: {str(e)}", "error": str(e)}


def summarize_node(state: AgentState) -> AgentState:
    """Delegate to summarization agent."""
    logger.info("=== SUMMARIZE ===")

    well_name = state.get("well_name")

    if not well_name:
        return {
            **state,
            "output": "Please specify which well to summarize.",
            "error": "No well name"
        }

    try:
        # Call summarization agent
        output = summarize_well(
            query=state["original_query"],
            well_name=well_name,
            max_words=200
        )

        return {**state, "output": output}
    except Exception as e:
        return {**state, "output": f"Error: {str(e)}", "error": str(e)}


def extract_parameters_node(state: AgentState) -> AgentState:
    """Delegate to parameter extraction agent."""
    logger.info("=== EXTRACT PARAMETERS ===")

    well_name = state.get("well_name")

    if not well_name:
        return {
            **state,
            "output": "Please specify which well to extract parameters from.",
            "error": "No well name"
        }

    try:
        # Call parameter extraction agent
        params_json = extract_parameters(
            query=state["original_query"],
            well_name=well_name
        )

        # Parse and format nicely
        params = json.loads(params_json)

        output = f"=== Extracted Parameters for {well_name} ===\n\n"
        output += f"Found: {params.get('parameters_found', 0)}/5 required parameters\n\n"

        # Show each parameter
        for param_name in ["tubing_id", "tubing_depth", "reservoir_pressure", "oil_gravity", "productivity_index"]:
            param = params.get(param_name, {})
            value = param.get('value', 'Not found')
            source = param.get('source_document', 'N/A')
            confidence = param.get('confidence', 'unknown')

            status = "✓" if confidence == "found" else "✗"
            output += f"{status} {param_name}: {value}\n"
            if confidence == "found":
                output += f"   Source: {source}\n"

        output += f"\nDocuments searched: {', '.join(params.get('documents_searched', []))}"

        return {**state, "output": output}
    except Exception as e:
        return {**state, "output": f"Error: {str(e)}", "error": str(e)}


# ==================== Routing ====================

def route_by_intent(state: AgentState) -> Literal["list", "info", "search", "summarize", "extract"]:
    """Route based on classified intent."""
    intent = state.get("intent", "search")
    logger.info(f"Routing to: {intent}")
    return intent


# ==================== Build Graph ====================

def create_langgraph_agent():
    """
    Create LangGraph agent that uses existing tools/agents.

    Flow:
        START
          ↓
        classify_intent
          ↓
        [route by intent]
          ├─ list → list_wells
          ├─ info → get_well_info
          ├─ search → search_documents
          ├─ summarize → summarize (uses sub-agent)
          └─ extract → extract_parameters (uses sub-agent)
          ↓
        END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("list", list_wells_node)
    workflow.add_node("info", get_well_info_node)
    workflow.add_node("search", search_documents_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("extract", extract_parameters_node)

    # Entry point
    workflow.set_entry_point("classify_intent")

    # Route based on intent
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "list": "list",
            "info": "info",
            "search": "search",
            "summarize": "summarize",
            "extract": "extract"
        }
    )

    # All nodes end
    workflow.add_edge("list", END)
    workflow.add_edge("info", END)
    workflow.add_edge("search", END)
    workflow.add_edge("summarize", END)
    workflow.add_edge("extract", END)

    return workflow.compile()


# ==================== Agent Wrapper ====================

class LangGraphMasterAgent:
    """
    LangGraph master agent - primary version.
    """

    def __init__(self):
        self.graph = create_langgraph_agent()
        logger.info("✓ LangGraph Master Agent initialized")

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run agent on query.

        Args:
            query: User's question

        Returns:
            {"output": str, "success": bool, "state": dict}
        """
        logger.info("=" * 80)
        logger.info(f"LangGraph Agent: '{query}'")
        logger.info("=" * 80)

        initial_state: AgentState = {
            "query": query,
            "original_query": query,
            "intent": "",
            "well_name": None,
            "search_query": None,
            "well_info": None,
            "output": "",
            "error": None
        }

        try:
            final_state = self.graph.invoke(initial_state)

            output = final_state.get("output", "No output generated")
            error = final_state.get("error")

            logger.info(f"{'✓' if not error else '✗'} Execution complete")

            return {
                "output": output,
                "success": not error,
                "state": final_state
            }

        except Exception as e:
            logger.error(f"✗ Execution failed: {e}", exc_info=True)

            return {
                "output": f"Error: {str(e)}",
                "success": False,
                "error": str(e)
            }


# ==================== Testing ====================

def test_langgraph_agent():
    """Test LangGraph master agent."""
    agent = LangGraphMasterAgent()

    test_queries = [
        "List all available wells",
        "Tell me about well-1",
        "Summarize well-1",
        "Extract parameters for well-1",
        "What is the tubing depth for well-1?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        result = agent.run(query)

        print(f"\nINTENT: {result['state'].get('intent')}")
        print(f"\nOUTPUT:")
        print(result["output"][:500])
        print(f"\nSTATUS: {'✓ Success' if result['success'] else '✗ Failed'}")

        if not result['success']:
            print(f"ERROR: {result.get('error')}")


if __name__ == "__main__":
    test_langgraph_agent()