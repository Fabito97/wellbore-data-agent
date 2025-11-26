"""
LangChain Master Agent - Refined Version (BACKUP)

Simple, functional, with Pydantic parsing to reduce JSON errors.
"""
from typing import Dict, Any
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.tools import tool

from app.agents.tools.rag_tool import list_available_wells, get_well_by_name, rag_query_tool
from app.agents.tools.summarization_tool import summarize_well
from app.agents.tools.extraction_tool import extract_parameters
from app.services.llm_service import get_llm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== Simplified Tools ====================

@tool
def list_wells_tool() -> str:
    """
    List all available wells in the system.
    Use when user asks 'what wells are available' or to show options.
    """
    try:
        result = list_available_wells.invoke({})

        if isinstance(result, list):
            output = "Available wells:\n"
            for well in result:
                name = well.get('name', 'unknown')
                count = well.get('document_count', 0)
                output += f"  • {name} ({count} documents)\n"
            return output
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_well_info_tool(well_name: str) -> str:
    """
    Get information about a specific well.
    Use FIRST to validate well exists before other operations.

    Args:
        well_name: Well identifier (e.g., "well-1")
    """
    try:
        result = get_well_by_name.invoke({"well_name": well_name})

        if result:
            name = result.get('name', 'unknown')
            count = result.get('document_count', 0)
            return f"Well '{name}' found with {count} documents."
        return f"Well '{well_name}' not found."
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_well_documents_tool(query: str, well_name: str) -> str:
    """
    Search well documents for specific information.

    Args:
        query: What to search for (e.g., "tubing depth")
        well_name: Well to search in (e.g., "well-1")
    """
    try:
        result = rag_query_tool.invoke({
            "query": query,
            "well_name": well_name,
            "top_k": 5
        })

        results_list = result.get("results", [])

        if not results_list:
            return f"No results found for '{query}' in {well_name}"

        # Format with retriever
        from app.rag.retriever import get_retriever
        retriever = get_retriever()
        formatted = retriever.format_context_for_llm(results_list, max_tokens=1500)

        return formatted
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def summarize_well_tool(well_name: str, query: str) -> str:
    """
    Generate comprehensive summary of well documents.

    Args:
        well_name: Well to summarize (e.g., "well-1")
        query: User's original question for context
    """
    try:
        return summarize_well(query=query, well_name=well_name, max_words=200)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def extract_parameters_tool(well_name: str, query: str, document_type: str = "WELL_REPORT") -> str:
    """
    Extract nodal analysis parameters from well documents.

    Args:
        well_name: Well to extract from (e.g., "well-1")
        query: User's original question for context
    """
    try:
        return extract_parameters(query=query, well_name=well_name, document_type=document_type)
    except Exception as e:
        return f"Error: {str(e)}"


# ==================== System Message with Better JSON Guidance ====================

SYSTEM_MESSAGE = """You are a wellbore analysis expert assistant. Your job is to provide accurate and correct information on users query using available documents.

Available tools:
- get_well_info_tool: Validate well exists
- search_well_documents_tool: Search for specific data
- summarize_well_tool: Summarize well documents
- extract_parameters_tool: Extract nodal parameters

WORKFLOW:
1. If user asks about wells without specifying: ask him for the well
2. If user mentions a well: use get_well_info_tool first to validate
3. For summaries: use summarize_well_tool
4. For parameters: use extract_parameters_tool
5. For specific questions: use search_well_documents_tool

CRITICAL JSON RULES:
- ALL keys need quotes: "action" not action
- ALL string values need quotes: "tool_name" not tool_name
- Use this exact format:

{
  "action": "tool_name",
  "action_input": {
    "param": "value"
  }
}

CORRECT:
{"action": "get_well_info_tool", "action_input": {"well_name": "well-1"}}

WRONG:
{action: get_well_info_tool, action_input: {well_name: "well-1"}}

Always pass the original user query to summarize_well_tool and extract_parameters_tool.
"""


# ==================== LangChain Master Agent ====================

class LangChainMasterAgent:
    """
    Refined LangChain agent with better error handling.
    """

    def __init__(self):
        self.llm = get_llm_service().llm

        self.tools = [
            list_wells_tool,
            get_well_info_tool,
            search_well_documents_tool,
            summarize_well_tool,
            extract_parameters_tool
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=(
                "JSON PARSING ERROR!\n\n"
                "Your JSON is malformed. Rules:\n"
                "1. Put QUOTES around all keys: \"action\" not action\n"
                "2. Put QUOTES around all string values: \"tool_name\" not tool_name\n"
                "3. No trailing commas\n\n"
                "Correct format:\n"
                "{\"action\": \"tool_name\", \"action_input\": {\"param\": \"value\"}}\n\n"
                "Try again with proper JSON formatting."
            ),
            agent_kwargs={
                "system_message": SYSTEM_MESSAGE
            },
            max_iterations=10,
            max_execution_time=300
        )

        logger.info(f"✓ LangChain Master Agent initialized with {len(self.tools)} tools")

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run agent on query.

        Args:
            query: User's question

        Returns:
            {"output": str, "success": bool}
        """
        logger.info("=" * 80)
        logger.info(f"LangChain Agent: '{query}'")
        logger.info("=" * 80)

        try:
            result = self.agent.invoke(query)
            output = result.get("output", "No output generated")

            logger.info("✓ Execution complete")

            return {
                "output": output,
                "success": True
            }

        except Exception as e:
            logger.error(f"✗ Execution failed: {e}", exc_info=True)

            return {
                "output": f"Error: {str(e)}",
                "success": False,
                "error": str(e)
            }


# ==================== Testing ====================

def test_langchain_agent():
    """Test LangChain master agent."""
    agent = LangChainMasterAgent()

    test_queries = [
        "List all available wells",
        "Summarize well-1",
        "Extract parameters for well-1",
        "What is the tubing depth for well-1?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        result = agent.run(query)

        print(f"\nOUTPUT:")
        print(result["output"][:500])
        print(f"\nSTATUS: {'✓ Success' if result['success'] else '✗ Failed'}")

        if not result['success']:
            print(f"ERROR: {result.get('error')}")


if __name__ == "__main__":
    test_langchain_agent()