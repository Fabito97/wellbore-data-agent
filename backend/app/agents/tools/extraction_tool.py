"""
Parameter Extraction Agent - Refined & Simplified

Flow:
1. Get well documents by name
2. For each document: search for parameters
3. Extract and structure parameters
4. Return what was found (no validation here)

Validation happens separately before nodal analysis.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from app.agents.tools.rag_tool import get_well_by_name, rag_query_tool
from app.rag.retriever import get_retriever
from app.services.llm_service import get_llm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== Pydantic Schemas ====================

class ParameterValue(BaseModel):
    """Single parameter with metadata."""
    value: Optional[str] = Field(description="Parameter value with unit (e.g., '2.875 inches')")
    source_document: Optional[str] = Field(description="Which document it came from")
    confidence: str = Field(description="found, partial, or not_found")


class ExtractedParameters(BaseModel):
    """Schema for extracted parameters."""
    well_name: str

    # Required for nodal analysis
    tubing_id: ParameterValue = Field(description="Tubing inner diameter")
    tubing_depth: ParameterValue = Field(description="Tubing depth")
    reservoir_pressure: ParameterValue = Field(description="Reservoir pressure")
    oil_gravity: ParameterValue = Field(description="API oil gravity")
    productivity_index: ParameterValue = Field(description="Productivity index (PI)")

    # Optional but useful
    gas_gravity: Optional[ParameterValue] = None
    water_cut: Optional[ParameterValue] = None
    gor: Optional[ParameterValue] = None
    reservoir_temperature: Optional[ParameterValue] = None
    wellhead_pressure: Optional[ParameterValue] = None

    # Summary
    documents_searched: List[str] = Field(description="List of documents searched")
    parameters_found: int = Field(description="Count of required parameters found")


# ==================== Parameter Extraction Agent ====================

class ParameterExtractionAgent:
    """
    Simplified parameter extraction agent.

    Process:
    1. Get well documents
    2. Search each document for parameters
    3. Extract and structure
    4. Return findings (no validation)
    """

    def __init__(self):
        self.llm = get_llm_service().llm
        self.retriever = get_retriever()
        self.parser = PydanticOutputParser(pydantic_object=ExtractedParameters)

        logger.info("✓ ParameterExtractionAgent initialized")

    def extract_parameters(
        self,
        query: str,
        well_name: str,
        document_type: str = None
    ) -> str:
        """
        Extract nodal analysis parameters.

        Args:
            query: User's original question for context
            well_name: Well to extract from
            document_type: Optional doc type filter

        Returns:
            JSON string of extracted parameters
        """
        logger.info(f"Extracting parameters for {well_name}")

        try:
            # Step 1: Get well documents
            well_data = self._get_well_documents(well_name, document_type)

            if not well_data["documents"]:
                return self._create_empty_result(well_name)

            # Step 2: Search for parameter data
            parameter_context = self._search_for_parameters(
                well_name,
                well_data["documents"],
                query
            )

            # Step 3: Extract parameters using LLM
            extracted = self._extract_with_llm(
                well_name=well_name,
                context=parameter_context,
                query=query,
                documents=well_data["documents"]
            )

            # Step 4: Format as JSON string
            output = extracted.model_dump_json(indent=2)

            logger.info(f"✓ Extracted {extracted.parameters_found}/5 required parameters")
            return output

        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}", exc_info=True)
            return self._create_error_result(well_name, str(e))

    def _get_well_documents(self, well_name: str, document_type: str = None) -> Dict[str, Any]:
        """Get well documents."""
        logger.info(f"Getting documents for {well_name}")

        well_data = get_well_by_name.invoke({"well_name": well_name})

        if not well_data:
            return {"well_name": well_name, "documents": []}

        docs = well_data.get("documents", [])

        # Filter by type if specified
        if document_type:
            docs = [d for d in docs if d.get("document_type") == document_type]

        logger.info(f"Found {len(docs)} documents")

        return {
            "well_name": well_name,
            "documents": docs
        }

    def _search_for_parameters(
        self,
        well_name: str,
        documents: List[Dict],
        query: str
    ) -> str:
        """Search documents for parameter-related content."""
        logger.info("Searching for parameter data")

        # Search queries optimized for parameters
        search_queries = [
            "tubing inner diameter depth specifications",
            "reservoir pressure productivity index PI",
            "oil gravity API gas gravity GOR water cut",
        ]

        all_results = []

        for search_query in search_queries:
            try:
                result = rag_query_tool.invoke({
                    "query": search_query,
                    "well_name": well_name,
                    "top_k": 5
                })

                results_list = result.get("results", [])
                all_results.extend(results_list)

            except Exception as e:
                logger.error(f"Search failed for '{search_query}': {e}")

        # Remove duplicates
        unique_results = []
        seen = set()
        for r in all_results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                unique_results.append(r)

        # Format context
        if unique_results:
            context = self.retriever.format_context_for_llm(unique_results, max_tokens=3000)
            logger.info(f"Found {len(unique_results)} relevant chunks")
            return context
        else:
            return "No parameter data found in documents."

    def _extract_with_llm(
        self,
        well_name: str,
        context: str,
        query: str,
        documents: List[Dict]
    ) -> ExtractedParameters:
        """Extract parameters using LLM with Pydantic parsing."""
        logger.info("Extracting parameters with LLM")

        # Create prompt
        prompt = f"""Extract nodal analysis parameters from the well data below.

User Query: {query}
Well: {well_name}

Well Data:
{context}

Extract these REQUIRED parameters:
1. tubing_id: Inner diameter (inches)
2. tubing_depth: Depth (ft or m)
3. reservoir_pressure: Pressure (psi or bar)
4. oil_gravity: API gravity
5. productivity_index: PI (m3/hr/bar or bbl/day/psi)

Also extract if available:
- gas_gravity
- water_cut
- gor (gas-oil ratio)
- reservoir_temperature
- wellhead_pressure

For each parameter:
- value: Include number AND unit (e.g., "2.875 inches")
- source_document: Document filename where found
- confidence: "found", "partial", or "not_found"

{self.parser.get_format_instructions()}
"""

        # Get LLM response
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        # Parse with Pydantic
        try:
            extracted = self.parser.parse(content)

            # Add metadata
            extracted.well_name = well_name
            extracted.documents_searched = [d.get("filename", "unknown") for d in documents]

            # Count found parameters
            found_count = 0
            for param in ["tubing_id", "tubing_depth", "reservoir_pressure", "oil_gravity", "productivity_index"]:
                param_obj = getattr(extracted, param)
                if param_obj.confidence == "found":
                    found_count += 1

            extracted.parameters_found = found_count

            return extracted

        except Exception as e:
            logger.error(f"Parsing failed: {e}, using fallback")

            # Fallback: Create basic structure
            return ExtractedParameters(
                well_name=well_name,
                tubing_id=ParameterValue(value=None, source_document=None, confidence="not_found"),
                tubing_depth=ParameterValue(value=None, source_document=None, confidence="not_found"),
                reservoir_pressure=ParameterValue(value=None, source_document=None, confidence="not_found"),
                oil_gravity=ParameterValue(value=None, source_document=None, confidence="not_found"),
                productivity_index=ParameterValue(value=None, source_document=None, confidence="not_found"),
                documents_searched=[d.get("filename") for d in documents],
                parameters_found=0
            )

    def _create_empty_result(self, well_name: str) -> str:
        """Create empty result when no documents found."""
        result = ExtractedParameters(
            well_name=well_name,
            tubing_id=ParameterValue(value=None, source_document=None, confidence="not_found"),
            tubing_depth=ParameterValue(value=None, source_document=None, confidence="not_found"),
            reservoir_pressure=ParameterValue(value=None, source_document=None, confidence="not_found"),
            oil_gravity=ParameterValue(value=None, source_document=None, confidence="not_found"),
            productivity_index=ParameterValue(value=None, source_document=None, confidence="not_found"),
            documents_searched=[],
            parameters_found=0
        )
        return result.model_dump_json(indent=2)

    def _create_error_result(self, well_name: str, error: str) -> str:
        """Create error result."""
        result = {
            "well_name": well_name,
            "error": error,
            "parameters_found": 0
        }
        import json
        return json.dumps(result, indent=2)


# ==================== Wrapper for Master Agent ====================

def extract_parameters(query: str, well_name: str, document_type: str = "WELL_REPORT") -> str:
    """
    Main function for master agent to call.

    Args:
        query: User's original question
        well_name: Well to extract from
        document_type: Optional doc type filter

    Returns:
        JSON string of extracted parameters
    """
    agent = ParameterExtractionAgent()
    return agent.extract_parameters(query, well_name, document_type)


# ==================== Testing ====================

if __name__ == "__main__":
    import json

    print("\n" + "=" * 80)
    print("Testing Parameter Extraction Agent")
    print("=" * 80)

    result = extract_parameters(
        query="Extract nodal analysis parameters for well-1",
        well_name="well-1"
    )

    # Pretty print JSON
    parsed = json.loads(result)
    print(json.dumps(parsed, indent=2))

    print(f"\nFound: {parsed['parameters_found']}/5 required parameters")