"""
Fixed Wellbore Agent - ReAct (structured) friendly for local LLMs.

Key design choices:
- Tools accept a single dict argument (flat). This avoids nested arg parsing issues.
- LLM MUST emit JSON like: {"action": "tool_name", "action_input": {"k": 5, ...}}
- We validate JSON and tool existence before execution.
- Minimal system prompt to reduce hallucination.
- Works with local HF/ollama/llama backends (LangChain agent API).
"""

import json
import logging
from typing import Any, Dict, Callable, List, Optional

from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.tools import Tool   # adjust if your env uses a different path
from langchain_classic import LLMChain
# NOTE: in some langchain versions Tool is in langchain.tools.tool import Tool
# adjust the import to match your installed version.

# Replace with your LLM provider wrapper or service getter
from app.services.llm_service import get_llm_service  # returns an object with .llm compatible with langchain
from app.services.document_service import get_document_service

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# -------------------------
# ----- Helper helpers -----
# -------------------------
def safe_json_parse(s: str) -> Optional[Dict[str, Any]]:
    """Parse JSON robustly (strip wrapper text) and return dict or None."""
    if not s:
        return None
    s = s.strip()
    # Try to find a JSON object inside string
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except Exception:
        # last resort: allow single quotes -> double quotes
        try:
            return json.loads(s[start:end + 1].replace("'", '"'))
        except Exception:
            return None


def validate_action_payload(payload: Dict[str, Any], tools: Dict[str, Callable]) -> Optional[str]:
    """
    Validate that payload has "action" and "action_input", and action exists.
    Returns error message string if invalid, otherwise None.
    """
    if not isinstance(payload, dict):
        return "Action must be a JSON object."
    if "action" not in payload:
        return "Missing 'action' field."
    if "action_input" not in payload:
        return "Missing 'action_input' field."
    action = payload["action"]
    if action not in tools:
        return f"Unknown action '{action}'. Available: {list(tools.keys())}"
    if not isinstance(payload["action_input"], dict):
        return "action_input must be a JSON object (dictionary)."
    return None


# -------------------------
# ----- Tool wrappers -----
# -------------------------
# All tools accept a single dict argument and return a string (safe for LLM).
doc_service = get_document_service()  # singleton in your app


def _list_wells_tool(args: Dict[str, Any]) -> str:
    """
    No required args. Returns a compact JSON-ish string listing wells.
    Example args: {}
    """
    try:
        wells = doc_service.list_wells()  # returns list of wells (id, name, count)
        # produce compact line list
        lines = []
        for w in wells:
            lines.append(f"- {w.get('name')} (id={w.get('id')}, docs={w.get('document_count')})")
        return "WELLS:\n" + ("\n".join(lines) if lines else "No wells found.")
    except Exception as e:
        logger.exception("list_wells error")
        return f"Error listing wells: {e}"


def _get_well_tool(args: Dict[str, Any]) -> str:
    """
    args: {"well_name": "well-1"} or {"well_id": "..."}
    Returns well metadata JSON string.
    """
    try:
        well_name = args.get("well_name")
        well_id = args.get("well_id")

        if well_id:
            wells = doc_service.list_wells()
            found = next((w for w in wells if w["id"] == well_id), None)
            if not found:
                return f"Well with id {well_id} not found."
            return json.dumps(found, default=str)
        if well_name:
            # try normalization and match
            normalized = well_name.strip().lower().replace(" ", "-")
            wells = doc_service.list_wells()
            found = next((w for w in wells if w["name"].lower() == normalized or w["name"].lower() == well_name.lower()), None)
            if not found:
                return f"Well with name '{well_name}' not found. Use list_wells to see available wells."
            return json.dumps(found, default=str)

        return "Provide 'well_name' or 'well_id' in action_input."
    except Exception as e:
        logger.exception("get_well error")
        return f"Error getting well: {e}"


def _search_documents_tool(args: Dict[str, Any]) -> str:
    """
    args: {"query": "tubing spec", "well_name": "well-1", "top_k": 5}
    Returns a short summary of top results (text + doc id).
    """
    try:
        q = args.get("query", "").strip()
        if not q:
            return "Provide non-empty 'query' in action_input."

        well_name = args.get("well_name")
        top_k = int(args.get("top_k", 5))

        # We will call the document service retriever (assumes method exists)
        # You may need to adapt to your retriever naming: e.g., rag_query_tool or vector_store.search_documents
        # Attempt to call service.search or service.retrieve_by_well
        try:
            results = doc_service.vector_store_vector_search_for_well(q, well_name, top_k)  # placeholder; adapt
        except Exception:
            # fallback to generic store search function (adjust to your API)
            try:
                results = doc_service.vector_store.query_similar_chunks(q, top_k=top_k, filters={"well_name": well_name})
            except Exception as e:
                logger.exception("search_documents fallback error")
                return f"Search failed: {e}"

        # results expected: list of dicts with 'content', 'chunk_id', 'metadata', 'similarity_score'
        lines = []
        for r in results[:top_k]:
            meta = r.get("metadata", {})
            filename = meta.get("filename") or meta.get("file_name") or "unknown"
            snippet = (r.get("content") or "")[:300].replace("\n", " ")
            lines.append(json.dumps({
                "doc_id": meta.get("document_id", meta.get("documentId", "")),
                "filename": filename,
                "page": meta.get("page_number"),
                "score": r.get("similarity_score"),
                "snippet": snippet
            }, default=str))
        return "RESULTS:\n" + ("\n".join(lines) if lines else "No results")
    except Exception as e:
        logger.exception("search_documents error")
        return f"Error searching documents: {e}"


def _summarize_well_tool(args: Dict[str, Any]) -> str:
    """
    args: {"well_name": "well-1", "max_words": 200}
    Returns a short combined context string (we keep it sized).
    """
    try:
        well_name = args.get("well_name")
        if not well_name:
            return "Provide well_name"

        max_words = int(args.get("max_words", 200))

        # Use your existing summarize function if present:
        try:
            summary_resp = doc_service.summarize_well_reports(well_name=well_name, max_words=max_words)
            # assume returns dict with 'summary' or similar
            if isinstance(summary_resp, dict):
                return summary_resp.get("summary", str(summary_resp))
            return str(summary_resp)
        except Exception:
            # fallback: do a small search for "summary" kind of results
            results = doc_service.vector_store.query_similar_chunks("summary " + well_name, top_k=5, filters={"well_name": well_name})
            combined = " ".join((r.get("content") or "") for r in results[:5])
            words = combined.split()
            return " ".join(words[:max_words]) if words else "No summarizable text found."

    except Exception as e:
        logger.exception("summarize_well error")
        return f"Error summarizing well: {e}"


def _extract_parameters_tool(args: Dict[str, Any]) -> str:
    """
    args: {"well_name": "well-1"}
    Returns extracted parameters in JSON-ish string.
    """
    try:
        well_name = args.get("well_name")
        if not well_name:
            return "Provide well_name"

        # Try to call document service's parameter extraction
        try:
            params = doc_service.extract_parameters_for_well(well_name=well_name)
            if isinstance(params, dict):
                return json.dumps(params, default=str)
            return str(params)
        except Exception:
            # fallback: run targeted searches for common parameters and assemble approximate JSON
            searches = ["tubing", "reservoir pressure", "fluid gravity", "productivity index"]
            out = {}
            for s in searches:
                results = doc_service.vector_store.query_similar_chunks(s + " " + well_name, top_k=3, filters={"well_name": well_name})
                out[s] = [(r.get("metadata", {}).get("document_id"), (r.get("content") or "")[:200]) for r in results]
            return json.dumps(out, default=str)

    except Exception as e:
        logger.exception("extract_parameters error")
        return f"Error extracting parameters: {e}"


# Map tool names -> callables
_TOOL_MAP: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "list_wells": _list_wells_tool,
    "get_well": _get_well_tool,
    "search_documents": _search_documents_tool,
    "summarize_well": _summarize_well_tool,
    "extract_parameters": _extract_parameters_tool,
}


# Build LangChain Tool objects for agent registration (descriptions short)
TOOL_REGISTRY: List[Tool] = [
    Tool(name="list_wells", func=_list_wells_tool, description="List wells. action_input: {}"),
    Tool(name="get_well", func=_get_well_tool, description="Get well metadata. action_input: {\"well_name\": \"well-1\"}"),
    Tool(name="search_documents", func=_search_documents_tool, description="Search documents for a well. action_input: {\"query\":\"...\",\"well_name\":\"well-1\",\"top_k\":5}"),
    Tool(name="summarize_well", func=_summarize_well_tool, description="Summarize well reports. action_input: {\"well_name\":\"well-1\",\"max_words\":200}"),
    Tool(name="extract_parameters", func=_extract_parameters_tool, description="Extract nodal parameters. action_input: {\"well_name\":\"well-1\"}"),
]


# -------------------------
# ----- Agent factory -----
# -------------------------
SYSTEM_PROMPT = """
You are a concise wellbore assistant. When you call a tool, you MUST output exactly ONE JSON object (no extra commentary).
The JSON must be: {"action": "<tool_name>", "action_input": { ... } }.

Rules:
- Use `get_well` if a well is mentioned, or `list_wells` if no well mentioned.
- Keep action_input flat (no nested objects).
- If unsure, ask a clarifying question (do NOT call tools).
- Do not invent numbers. Use tools to fetch data.

Examples:
{"action":"list_wells","action_input":{}}
{"action":"get_well","action_input":{"well_name":"well-1"}}
{"action":"search_documents","action_input":{"query":"tubing specification","well_name":"well-1","top_k":5}}
"""

def build_agent() -> Any:
    """
    Build and return a LangChain agent configured for structured ReAct.
    """
    llm_wrapper = get_llm_service().llm  # must be compatible with langchain LLM
    # initialize_agent expects `tools` and an LLM instance
    agent = initialize_agent(
        tools=TOOL_REGISTRY,
        llm=llm_wrapper,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": SYSTEM_PROMPT},
        max_iterations=8,
        max_execution_time=60
    )
    return agent


# -------------------------
# ----- Runner / helper ----
# -------------------------
def run_query(agent, user_question: str) -> Dict[str, Any]:
    """
    Run the agent in a defensive loop:
    1) Get agent raw output (string)
    2) Parse JSON action
    3) Validate payload
    4) Execute tool and return observation
    If the agent returns a final answer instead of an action, return it.
    """
    logger.info("User question: %s", user_question)

    # invoke agent to produce an action in a single step (agent.invoke may return dict or object)
    # We call the agent but will validate its textual output before running tools for safety.
    try:
        raw = agent.invoke(user_question)
    except Exception as e:
        logger.exception("Agent.invoke error")
        return {"success": False, "error": f"Agent invocation error: {e}"}

    # langchain agent.invoke may return a dict with "output" or similar; attempt to extract
    text_output = None
    if isinstance(raw, dict):
        text_output = raw.get("output") or raw.get("text") or str(raw)
    else:
        text_output = str(raw)

    logger.debug("Agent raw output: %s", text_output[:1000])

    # Try to parse JSON action
    payload = safe_json_parse(text_output)
    if not payload:
        # LLM didn't return action JSON - treat as final answer text
        return {"success": True, "final_answer": text_output}

    # Validate
    err = validate_action_payload(payload, _TOOL_MAP)
    if err:
        # If invalid, return message and raw LLM output so you can inspect
        return {"success": False, "error": f"Invalid tool call: {err}", "raw": text_output}

    # Execute tool
    action = payload["action"]
    args = payload["action_input"]
    func = _TOOL_MAP[action]
    logger.info("Executing tool %s with args %s", action, args)
    try:
        obs = func(args)
        # Provide the tool observation back in a safe structure for client or further chaining
        return {"success": True, "action": action, "observation": obs}
    except Exception as e:
        logger.exception("Tool execution error")
        return {"success": False, "error": f"Tool execution failed: {e}"}


# -------------------------
# ----- Quick demo/test ----
# -------------------------
if __name__ == "__main__":
    agent = build_agent()
    queries = [
        "List wells available",
        "What is the tubing specification for well-1?",
        "Summarize the well-1 completion report in 150 words"
    ]
    for q in queries:
        print("\n\n>> QUERY:", q)
        out = run_query(agent, q)
        print("OUT:", json.dumps(out, indent=2, default=str))
