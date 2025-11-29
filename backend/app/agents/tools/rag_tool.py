"""
RAG Tools for Agent - Well-scoped document retrieval.

These tools are exposed to the agent for:
1. Summarization (Sub-challenge 1)
2. Parameter extraction (Sub-challenge 2)
3. General queries
4. Well detection

All tools support well-scoped filtering.
"""
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool

from app.db.database import SessionLocal
from app.rag.retriever import get_retriever
from app.services.document_service import DocumentService, get_document_service
from app.utils.helper import normalize_well_name

from app.utils.logger import get_logger
from app.agents.tools.nodal_tool import run_nodal_analysis_tool
import re

logger = get_logger(__name__)


# Lazy singletons (initialized on first use)
_document_service: Optional[DocumentService] = None
_retriever = None  # Initialize lazily

def _get_service():
    """Get document service instance (lazily create with a real DB session).

    Uses SessionLocal() to create a SQLAlchemy Session. We intentionally keep
    a long-lived session for these tools (process-level singleton). If you
    prefer shorter-lived sessions, create and close sessions per-call.
    """
    global _document_service
    if _document_service is None:
        # Create a DB session and DocumentService once
        db = SessionLocal()
        _document_service = get_document_service(db=db)
    return _document_service

def _get_retriever():
    """Get retriever instance."""
    global _retriever
    if _retriever is None:
        # Use the project's retriever factory which returns a DocumentRetriever singleton
        _retriever = get_retriever()
    return _retriever

# ==================== Well Detection ====================
@tool('detect_well_tool')
def detect_well_from_query(query: str) -> Optional[str]:
    """
    Use this tool to extract well reference from user query.

    Input:
        query (str): The user's query string.

    Returns:
        Normalized well name (e.g., "well-4") or None
    """
    query_lower = query.lower()

    # Pattern 1: "well 4", "well-4", "well_4"
    match = re.search(r'\bwell[\s\-_]*(\d+)\b', query_lower)
    if match:
        well_num = match.group(1).lstrip('0') or '0'
        normalized = f"well-{well_num}"
        logger.info(f"Detected well from query: {normalized}")
        return normalized

    # Pattern 2: "w4", "w-4"
    match = re.search(r'\bw[\s\-_]*(\d+)\b', query_lower)
    if match:
        well_num = match.group(1).lstrip('0') or '0'
        normalized = f"well-{well_num}"
        logger.info(f"Detected well from query (short form): {normalized}")
        return normalized

    logger.debug("No well reference detected in query")
    return None


@tool('list_wells')
def list_available_wells() -> List[Dict[str, Any]]:
    """
    List all wells in the system.

    Returns:
        [{"well_name": "well-4", "document_count": 3, ...}, ...]

    LLM/Agent usage:
      - Call this tool when the user asks "what wells do we have" or to enumerate wells
        before deciding which well to query.
      - Output is JSON-serializable; the agent should parse and present the names and counts.

    Example:
      list_wells() -> [{"id":"well-1","name":"well-1","document_count":3}, ...]
    """
    service = _get_service()  # Use the singleton
    return service.list_wells()



@tool('list_wells_with_documents')
def list_available_wells_with_documents() -> List[Dict[str, Any]]:
    """
    List all wells and include basic document summaries.

    Returns:
        [
            {
                "id": "well-xxxx",
                "name": "well-4",
                "document_count": 3,
                "documents": [{"id": "doc-...", "filename": "report.pdf", "document_type": "WELL_REPORT"}, ...],
                "created_at": "2025-11-24T..."
            },
            ...
        ]

    LLM/Agent usage:
      - Use when the agent needs to show documents per well or pick a document for detailed retrieval.
      - Prefer to call this before expensive retrievals if the agent needs to present options to the user.
    """
    service = _get_service()  # Use the singleton
    return service.list_wells_with_documents()



@tool('get_well_by_name')
def get_well_by_name(well_name: str) -> Dict[str, Any]:
    """
    Use this to get a well record including its documents by (normalized) name.

    Input:
      - well_name (str): free-form well identifier (e.g. "Well 4", "W4", "well-4").
        The function normalizes the name to the internal format (e.g. "well-4").

    Output (dict):
      {
        "id": "well-xxxx",
        "name": "well-4",
        "created_at": datetime,
        "document_count": int,
        "documents": [{"document_id": "...", "filename": "...", "document_type": "..."}, ...]
      }
    """
    service = _get_service()  # Use the singleton
    return service.get_well(well_name=well_name)


# ==================== Tool 1: General Query (with well scope) ====================
@tool('rag_query')
def rag_query_tool(
    query: str,
    well_name: Optional[str] = None,
    document_id: Optional[str] = None,
    document_type: Optional[str] = None,
    top_k: int = 5,
    chunk_type: Optional[str] = None  # "text" or "table"
) -> Dict[str, Any]:
    """
    Use this to make a general RAG query with optional filtering.

    Inputs:
      - query (str): Natural language query.
      - well_name (optional str): Well identifier to scope the search (e.g. "well-4").
      - document_type (optional str): Document type filter (e.g. "WELL_REPORT").
      - top_k (int): number of results to return.
      - chunk_type (optional str): "text" or "table" to filter chunk types.

    Output (dict):
      {
        "results": [RetrievalResult, ...],  # objects with .content, .filename, .page_number, .chunk_type, .chunk_id
        "count": int,
        "well_name": str | None,
        "document_type": str | None,
        "query": str
      }
    """
    logger.info(f"RAG query: '{query}' (well: {well_name}, type: {document_type})")

    # Build filters
    filters = {}
    if well_name:
        filters["well_name"] = normalize_well_name(well_name)
    if document_id:  # ADD THIS
        filters["document_id"] = document_id
    if document_type:
        filters["document_type"] = document_type
    if chunk_type:
        filters["chunk_type"] = chunk_type

    # Retrieve
    retriever = _get_retriever()  # Use singleton
    results = retriever.retrieve(
        query=query,
        top_k=top_k,
        filters=filters if filters else None
    )

    logger.info(f"Retrieved {len(results)} results")

    if results and len(results) > 0:
        retriever.format_context_for_llm(results)
    return {
         "results": results,
         "count": len(results),
         "well_name": well_name,
         "document_type": document_type,
         "query": query
     }


# ==================== Tool 2: Summarization ====================

@tool('summarize_report')
def summarize_well_report_tool(
    well_name: str,
    document_type: str = "WELL_REPORT",
    max_words: int = 200
) -> Dict[str, Any]:
    """
    Use this to summarize a well report (Sub-challenge 1).

    Args:
        well_name: Well to summarize (e.g., "well-4")
        document_type: Type of document (default: "WELL_REPORT")
        max_words: Max words in summary

    Returns:
        {
            "well_name": str,
            "chunks_used": int,
            "context": str,  # Full text for LLM to summarize
            "document_ids": [...]
        }
    """
    logger.info(f"Summarizing well report: {well_name}")

    normalized_well = normalize_well_name(well_name)

    # Get well record (with documents) directly from the DocumentService
    service = _get_service()
    well_record = service.get_well(well_name=normalized_well)

    # get_well returns None if the well does not exist
    if not well_record or not well_record.get("documents"):
        return {
            "well_name": normalized_well,
            "chunks_used": 0,
            "context": "No " + document_type + " documents found for " + normalized_well,
            "document_ids": []
        }

    # Filter documents by document_type if needed
    well_docs = [
        d for d in well_record.get("documents", [])
        if (not document_type) or (d.get("document_type") == document_type)
    ]

    if not well_docs:
        return {
            "well_name": normalized_well,
            "chunks_used": 0,
            "context": "No " + document_type + " documents found for " + normalized_well,
            "document_ids": []
        }

    # Get chunks from these documents
    retriever = _get_retriever()  # Use singleton
    all_chunks = []

    for doc in well_docs:
        doc_id = doc.get('document_id')
        # Explicitly pass normalized well name and document id to retrieve_for_summarization
        chunks = retriever.retrieve_for_summarization(
            well_name=normalized_well,
            document_id=doc_id,
            max_chunks=20  # Get plenty of content
        )
        all_chunks.extend(chunks)

    # Format context for LLM
    context = retriever.format_context_for_llm(
        all_chunks,
        max_tokens=3000  # Allow more tokens for summarization
    )

    logger.info(f"Retrieved {len(all_chunks)} chunks for summarization")

    return {
        "well_name": normalized_well,
        "document_type": document_type,
        "chunks_used": len(all_chunks),
        "context": context,
        "document_ids": [d['document_id'] for d in well_docs],
        "max_words": max_words
    }


# ==================== Tool 3: Parameter Extraction ====================

@tool('extract_parameters')
def extract_parameters_tool(
    well_name: str,
    parameter_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Extract nodal analysis parameters from well report (Sub-challenge 2).

    Args:
        well_name: Well to extract from
        parameter_types: Specific params to look for (optional)
            ["tubing", "pressure", "fluid_properties", "production"]

    Returns:
        {
            "well_name": str,
            "table_chunks": [RetrievalResult, ...],
            "text_chunks": [RetrievalResult, ...],
            "context": str
        }

    Agent workflow:
        1. Agent calls: extract_parameters_tool("well-4")
        2. Tool returns tables + relevant text
        3. Agent extracts specific parameters
    """
    logger.info(f"Extracting parameters for: {well_name}")

    normalized_well = normalize_well_name(well_name)

    # Query for parameter-related content
    queries = [
        "tubing specifications inner diameter depth",
        "reservoir pressure productivity index",
        "oil gravity gas gravity water cut GOR",
        "production rate flowing pressure"
    ]

    retriever = _get_retriever()  # Use singleton
    all_results = []

    for query in queries:
        results = retriever.retrieve(
            query=query,
            top_k=3,
            filters={"well_name": normalized_well}
        )
        all_results.extend(results)

    # Separate tables and text
    table_chunks = [r for r in all_results if r.chunk_type == "table"]
    text_chunks = [r for r in all_results if r.chunk_type == "text"]

    # Remove duplicates (by chunk_id)
    seen = set()
    unique_results = []
    for r in all_results:
        if r.chunk_id not in seen:
            seen.add(r.chunk_id)
            unique_results.append(r)

    # Format context
    context = retriever.format_context_for_llm(unique_results, max_tokens=4000)

    logger.info(
        f"Extracted {len(table_chunks)} tables, {len(text_chunks)} text chunks "
        f"for parameter extraction"
    )

    return {
        "well_name": normalized_well,
        "table_chunks": table_chunks,
        "text_chunks": text_chunks,
        "total_chunks": len(unique_results),
        "context": context
    }


# ==================== Tool 4: Table-Only Query ====================

@tool('query_tables')
def query_tables_tool(
    query: str,
    well_name: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Query ONLY tables (useful for structured data extraction).

    Args:
        query: What to search for
        well_name: Filter by well
        top_k: Number of tables

    Returns:
        {
            "tables": [RetrievalResult, ...],
            "count": int,
            "context": str
        }
    """
    logger.info(f"Table query: '{query}' (well: {well_name})")

    filters = {"chunk_type": "table"}
    if well_name:
        filters["well_name"] = normalize_well_name(well_name)

    retriever = _get_retriever()  # Use singleton
    results = retriever.retrieve(
        query=query,
        top_k=top_k,
        filters=filters
    )

    context = retriever.format_context_for_llm(results, max_tokens=3000)

    return {
        "tables": results,
        "count": len(results),
        "context": context,
        "well_name": well_name
    }


# ==================== Tool Registry for Agent ====================

AGENT_TOOLS = {
    "detect_well": {
        "function": detect_well_from_query,
        "description": "Detect which well the user is asking about from their query",
        "parameters": {
            "query": {"type": "string", "description": "User's question"}
        }
    },

    "list_wells": {
        "function": list_available_wells,
        "description": "List all available wells in the system",
    },

    "rag_query": {
        "function": rag_query_tool,
        "description": "Search documents with optional well/document filtering",
        "parameters": {
            "query": {"type": "string", "required": True},
            "well_name": {"type": "string", "optional": True},
            "document_type": {"type": "string", "optional": True},
            "top_k": {"type": "integer", "default": 5}
        }
    },

    "summarize_report": {
        "function": summarize_well_report_tool,
        "description": "Get context to summarize a well report",
        "parameters": {
            "well_name": {"type": "string", "required": True},
            "max_words": {"type": "integer", "default": 200}
        }
    },

    "extract_parameters": {
        "function": extract_parameters_tool,
        "description": "Extract nodal analysis parameters from well report",
        "parameters": {
            "well_name": {"type": "string", "required": True}
        }
    },

    "nodal_analysis": {
        "function": run_nodal_analysis_tool,
        "description": "Run nodal analysis given explicit parameters (reservoir pressure and PI are required).",
        "parameters": {
            "reservoir_pressure": {"type": "number", "required": True, "description": "Reservoir pressure (bar or psi)"},
            "productivity_index": {"type": "number", "required": True, "description": "Productivity index (m3/hr per bar)"},
            "tubing_id": {"type": "number", "required": False, "description": "Tubing inner diameter (inches)"},
            "tubing_depth": {"type": "number", "required": False, "description": "Tubing depth (ft or m)"},
            "oil_gravity": {"type": "number", "required": False, "description": "API gravity"},
            "wellhead_pressure": {"type": "number", "required": False, "description": "Wellhead pressure (bar or psi)"},
            "pump_curve": {"type": "object", "required": False, "description": "Optional pump curve dict with 'flow' and 'head' lists"},
            "return_curves": {"type": "boolean", "required": False, "description": "Include VLP/IPR curves in response"}
        }
    },

    "query_tables": {
        "function": query_tables_tool,
        "description": "Query only table data (for structured information)",
        "parameters": {
            "query": {"type": "string", "required": True},
            "well_name": {"type": "string", "optional": True}
        }
    }
}


# ==================== Helper: Get Tool by Name ====================

def get_tool(tool_name: str):
    """Get tool function by name."""
    if tool_name not in AGENT_TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    return AGENT_TOOLS[tool_name]["function"]


def list_tool_descriptions() -> List[Dict[str, Any]]:
    """Get all tool descriptions for agent prompt."""
    return [
        {
            "name": name,
            "description": tool["description"],
            "parameters": tool["parameters"] if tool["parameters"] else None
        }
        for name, tool in AGENT_TOOLS.items()
    ]
