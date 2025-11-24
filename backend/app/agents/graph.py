"""
Fixed Agent - Following Organizer's Working Pattern

Key changes:
1. Use @tool decorated functions DIRECTLY (not wrapped in lambdas)
2. Simple tool descriptions like organizer's example
3. Use STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION agent type
4. Clear system message instead of complex prompt template
"""
from typing import List, Dict, Any
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.tools import tool

from app.agents.tools.rag_tool import (
    list_available_wells,
    list_available_wells_with_documents,
    get_well_by_name,
    rag_query_tool,
    summarize_well_report_tool,
    extract_parameters_tool,
    query_tables_tool
)
from app.services.llm_service import get_llm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== Create Wrapper Tools (because original tools need specific signatures) ====================

@tool("list_wells")
def list_wells_wrapper() -> str:
    """
    Lists all available wells in the system with document counts.
    Use when user asks 'what wells are available' or doesn't specify a well.
    Input: any string (ignored).
    """
    try:
        wells = list_available_wells.invoke({})
        return str(wells)
    except Exception as e:
        return f"Error listing wells: {e}"


@tool("get_well")
def get_well_wrapper(well_name: str) -> str:
    """
    Gets details about a specific well to confirm it exists.
    Use FIRST when user mentions a well name (e.g., 'Well 4', 'well-4').
    Input: well name like 'well-4' or 'Well 4'.
    Returns: Well details with exact well_name to use in other tools.
    """
    try:
        result = get_well_by_name.invoke({"well_name": well_name})
        return str(result)
    except Exception as e:
        return f"Error getting well '{well_name}': {e}"


@tool("search_well_documents")
def search_documents_wrapper(query: str, well_name: str = None, top_k: int = 5) -> str:
    """
    Searches well documents. Must use exact well_name from get_well tool.
    Use for general questions about well data.
    Inputs:
        - query: what to search for (e.g., 'tubing depth', 'reservoir pressure')
        - well_name: exact well name from get_well (e.g., 'well-4')
        - top_k: number of results (default 5)
    """
    try:
        result = rag_query_tool.invoke({
            "query": query,
            "well_name": well_name,
            "top_k": top_k
        })

        # Format nicely for agent
        results_list = result.get("results", [])
        if not results_list:
            return f"No results found for query '{query}' in {well_name}"

        formatted = f"Found {len(results_list)} results:\n\n"
        for i, r in enumerate(results_list[:3], 1):  # Show top 3
            formatted += f"{i}. {r.text[:200]}...\n"

        return formatted
    except Exception as e:
        return f"Error searching documents: {e}"


@tool("get_summary_context")
def summary_wrapper(well_name: str) -> str:
    """
    Gets comprehensive document context to summarize a well report.
    Use when user asks to 'summarize' or 'describe' a well.
    Input: exact well name from get_well (e.g., 'well-4').
    """
    try:
        result = summarize_well_report_tool.invoke({
            "well_name": well_name,
            "max_words": 200
        })

        context = result.get("context", "")
        chunks = result.get("chunks_used", 0)

        return f"Retrieved {chunks} document chunks for {well_name}:\n\n{context[:1000]}..."
    except Exception as e:
        return f"Error getting summary context: {e}"


@tool("extract_parameters")
def parameters_wrapper(well_name: str) -> str:
    """
    Extracts nodal analysis parameters from well documents.
    Use when user asks to 'extract parameters' or needs data for nodal analysis.
    Input: exact well name from get_well (e.g., 'well-4').
    Returns: Tables and text containing tubing specs, reservoir pressure, fluid properties, etc.
    """
    try:
        result = extract_parameters_tool.invoke({"well_name": well_name})

        context = result.get("context", "")
        tables = result.get("table_chunks", [])

        return f"Extracted parameter data for {well_name}:\n{len(tables)} tables found\n\n{context[:1500]}..."
    except Exception as e:
        return f"Error extracting parameters: {e}"


@tool("validate_nodal_parameters")
def validate_wrapper(parameters: str) -> str:
    """
    Validates if extracted parameters are sufficient for nodal analysis.
    Input: JSON string of parameters like '{"tubing_id": 2.875, "reservoir_pressure": 3500, ...}'.
    Returns: validation result listing missing required parameters.
    """
    import json

    try:
        params = json.loads(parameters)
    except:
        return "Error: Invalid JSON format"

    required = [
        "tubing_id",
        "tubing_depth",
        "reservoir_pressure",
        "oil_gravity",
        "productivity_index"
    ]

    missing = [p for p in required if p not in params or params[p] is None]

    if not missing:
        return "✓ All required parameters present. Ready for nodal analysis."
    else:
        return f"✗ Missing required parameters: {', '.join(missing)}"


# ==================== Agent System Message ====================

SYSTEM_MESSAGE = """You are a wellbore analysis expert assistant with access to well completion reports.

WORKFLOW - Always follow this order:

1. IDENTIFY THE WELL:
   - If user mentions a well name (e.g., "Well 4"), use get_well tool first to confirm it exists
   - If no well mentioned, use list_wells and ask user which well
   - ALWAYS use the exact well_name returned by get_well in subsequent queries

2. ANSWER THE QUESTION:
   - For "summarize well": get_summary_context → generate summary
   - For "extract parameters": extract_parameters → parse data → validate_nodal_parameters
   - For specific questions: search_well_documents with appropriate query
   - For "what's the tubing depth": search_well_documents(query="tubing depth", well_name="well-4")

3. FORMAT YOUR RESPONSE:
   - Be concise and specific
   - Include relevant data values
   - For parameters, format as a clear list if more than one
   - If data is missing, explicitly state what's missing

IMPORTANT:
- IMPORTANT: When calling tools, always format the action as valid JSON with quoted keys and values.
Example:
{
  "action": "search_well_documents",
  "action_input": {
    "query": "tubing depth",
    "well_name": "well-1"
  }
}
- Use exact well names from get_well tool (e.g., "well-4" not "Well 4")
- Don't make up data - only use what the tools return
- If a tool returns an error, explain it to the user clearly

Available tools will help you query well documents, extract parameters, and validate data completeness.
"""


# ==================== Wellbore Agent Class ====================

class WellboreAgent:
    """
    Wellbore analysis agent using LangChain's structured chat agent.

    This follows the organizer's working pattern from the hackathon example.
    """

    def __init__(self):
        """Initialize agent with tools and LLM."""

        # Get LLM from service
        self.llm = get_llm_service().llm

        # Create tools list
        self.tools = [
            list_wells_wrapper,
            get_well_wrapper,
            search_documents_wrapper,
            summary_wrapper,
            parameters_wrapper,
            # validate_wrapper
        ]

        # Initialize agent using organizer's pattern
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={"system_message": SYSTEM_MESSAGE},
            max_iterations=15,
            max_execution_time=300
        )

        logger.info(f"✓ WellboreAgent initialized with {len(self.tools)} tools")
        logger.info(f"  Tools: {[t.name for t in self.tools]}")

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run agent on user query.

        Args:
            query: User's question

        Returns:
            {
                "output": str,
                "success": bool,
                "error": str (if failed)
            }
        """
        logger.info("=" * 80)
        logger.info(f"Processing: '{query}'")
        logger.info("=" * 80)

        try:
            # Invoke agent (organizer's pattern)
            result = self.agent.invoke(query)

            # Extract output
            output = result.get("output", "No response generated")

            logger.info("✓ Agent completed successfully")

            return {
                "output": output,
                "success": True
            }

        except Exception as e:
            logger.error(f"✗ Agent failed: {e}", exc_info=True)

            return {
                "output": f"I encountered an error: {str(e)}",
                "success": False,
                "error": str(e)
            }


# ==================== Example Usage ====================

def test_agent():
    """Test the agent with various queries."""

    agent = WellboreAgent()

    test_queries = [
        # "List all available wells",
        "What's the reservoir pressure?",  # No well specified
        "What is the tubing depth for Well 1?",
        # "Summarize the completion report for well-1",
        # "Extract nodal analysis parameters for Well 1",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        result = agent.run(query)

        print("\nRESULT:")
        print(result["output"])
        print("\nSTATUS:", "✓ Success" if result["success"] else "✗ Failed")

        if not result["success"]:
            print("ERROR:", result.get("error"))


if __name__ == "__main__":
    try:
        test_agent()
    except KeyboardInterrupt:
        logger.info(f"<UNK> Agent interrupted")
