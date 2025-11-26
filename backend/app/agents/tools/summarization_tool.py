"""
Summarization Agent - Refined & Simplified

Flow:
1. Get well documents by name
2. For each document: scan and summarize in batches with bullet points
3. Tag each doc summary
4. Combine all into final summary

No complex tool calling - just direct flow with LLM.
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agents.tools.rag_tool import get_well_by_name
from app.rag.retriever import get_retriever
from app.services.llm_service import get_llm_service
from app.utils.folder_utils import extract_well_name
from app.utils.helper import detect_well_from_query
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== Pydantic Schemas ====================

class DocumentSummary(BaseModel):
    """Schema for single document summary."""
    document_id: str = Field(description="Document identifier")
    document_type: str = Field(description="Type of document (e.g., WELL_REPORT)")
    filename: str = Field(description="Document filename")
    key_points: List[str] = Field(description="Bullet points of key information")
    summary_text: str = Field(description="Brief summary paragraph")


class FinalSummary(BaseModel):
    """Schema for final combined summary."""
    well_name: str
    total_documents: int
    document_summaries: List[DocumentSummary]
    final_summary: str = Field(description="Overall summary combining all documents")
    word_count: int


# ==================== Summarization Agent ====================

class SummarizationAgent:
    """
    Simplified summarization agent.

    Process:
    1. Get well documents
    2. Summarize each document with bullet points
    3. Combine into final summary
    """

    def __init__(self):
        self.llm = get_llm_service().llm
        self.retriever = get_retriever()

        # Pydantic parsers
        self.doc_summary_parser = PydanticOutputParser(pydantic_object=DocumentSummary)
        self.final_summary_parser = PydanticOutputParser(pydantic_object=FinalSummary)

        logger.info("✓ SummarizationAgent initialized")

    def summarize(
        self,
        query: str,
        well_name: str,
        document_type: str = None,
        max_words: int = 200
    ) -> str:
        """
        Summarize well documents.

        Args:
            query: Original user query for context
            well_name: Well to summarize
            document_type: Optional filter (e.g., "WELL_REPORT")
            max_words: Max words for final summary

        Returns:
            Formatted summary string
        """
        logger.info(f"Starting summarization: {well_name}, max_words={max_words}")

        try:
            # Step 1: Get well documents
            well_data = self._get_well_documents(well_name, document_type)

            if not well_data["documents"]:
                return f"No documents found for {well_name}"

            # Step 2: Summarize each document
            doc_summaries = []
            for doc in well_data["documents"]:
                summary = self._summarize_document(doc, well_name, query)
                if summary:
                    doc_summaries.append(summary)

            # Step 3: Combine into final summary
            final = self._combine_summaries(
                well_name=well_name,
                doc_summaries=doc_summaries,
                query=query,
                max_words=max_words
            )

            # Step 4: Format output
            output = self._format_output(final)

            logger.info(f"✓ Summarization complete: {final.word_count} words")
            return output

        except Exception as e:
            logger.error(f"Summarization failed: {e}", exc_info=True)
            return f"Error: Could not summarize {well_name}. {str(e)}"

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

    def _summarize_document(
        self,
        doc: Dict[str, Any],
        well_name: str,
        query: str
    ) -> DocumentSummary:
        """Summarize a single document with bullet points."""
        doc_id = doc.get("document_id")
        filename = doc.get("filename", "unknown")
        doc_type = doc.get("document_type", "UNKNOWN")

        logger.info(f"Summarizing document: {filename}")

        try:
            # Get document chunks
            chunks = self.retriever.retrieve_for_summarization(
                well_name=well_name,
                document_id=doc_id,
                max_chunks=20
            )

            if not chunks:
                return None

            # Format context
            context = self.retriever.format_context_for_llm(chunks, max_tokens=2000)

            # Create prompt
            prompt = f"""Summarize this well document focusing on what the user asked about.

User Query: {query}

Document: {filename} ({doc_type})

Content:
{context}

Provide:
1. 3-5 bullet points of key information
2. A brief summary paragraph (max 150 words)

{self.doc_summary_parser.get_format_instructions()}
"""

            # Get LLM response
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse with Pydantic
            try:
                summary = self.doc_summary_parser.parse(content)
                summary.document_id = doc_id
                summary.document_type = doc_type
                summary.filename = filename
                return summary
            except:
                # Fallback: create summary manually
                logger.warning(f"Parser failed for {filename}, using fallback")
                return DocumentSummary(
                    document_id=doc_id,
                    document_type=doc_type,
                    filename=filename,
                    key_points=["Summary generation failed - see raw content"],
                    summary_text=content[:200]
                )

        except Exception as e:
            logger.error(f"Failed to summarize {filename}: {e}")
            return None

    def _combine_summaries(
        self,
        well_name: str,
        doc_summaries: List[DocumentSummary],
        query: str,
        max_words: int
    ) -> FinalSummary:
        """Combine document summaries into final summary."""
        logger.info("Combining summaries")

        # Format all document summaries
        combined_text = ""
        for ds in doc_summaries:
            combined_text += f"\n[{ds.filename}]\n"
            for point in ds.key_points:
                combined_text += f"  • {point}\n"
            combined_text += f"Summary: {ds.summary_text}\n"

        # Create final summary prompt
        prompt = f"""You are a well bore specialist provides summary insight on well reports or activities. 
        Based on the provided information, provide a comprehensive summary that addresses the user's query of this well documents.

User Query: {query}
Well: {well_name}
{f"Max:words: {max_words}" if max_words else ""}

Document Information:
{combined_text}

Create a final summary that:
1. Addresses what the user asked about
2. Combines key information from all documents
3. Is the amount of words requested by the user or less
4. Is clear and technical
5. Only answer based on the provided documents information
"""

        # Get LLM response
        response = self.llm.invoke(prompt)
        final_text = response.content if hasattr(response, 'content') else str(response)

        # Create final summary object
        final = FinalSummary(
            well_name=well_name,
            total_documents=len(doc_summaries),
            document_summaries=doc_summaries,
            final_summary=final_text.strip(),
            word_count=len(final_text.split())
        )

        return final

    def _format_output(self, final: FinalSummary) -> str:
        """Format final summary for display."""
        output = f"=== Summary of {final.well_name} ===\n\n"
        output += f"{final.final_summary}\n\n"
        output += f"--- Document Details ({final.total_documents} documents) ---\n"

        for ds in final.document_summaries:
            output += f"\n[{ds.filename}] ({ds.document_type})\n"
            for point in ds.key_points:
                output += f"  • {point}\n"

        output += f"\nWord count: {final.word_count}"

        return output


# ==================== Wrapper for Master Agent ====================

def summarize_well(query: str, well_name: str, document_type: str = "WELL_REPORT", max_words: int = 200) -> str:
    """
    Main function for master agent to call.

    Args:
        query: User's original question
        well_name: Well to summarize
        document_type: Optional doc type filter
        max_words: Max words for summary

    Returns:
        Formatted summary string
    """
    agent = SummarizationAgent()
    return agent.summarize(query, well_name, document_type, max_words)


# ==================Interactive Window=================
def interactive_mode():
    """Run agent in interactive terminal mode."""
    print("\n" + "=" * 80)
    print("QUERY AGENT - Interactive Mode")
    print("=" * 80)
    print("\nThis agent answers questions about well data.")
    print("Type 'exit' or 'quit' to stop.\n")


    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break

            # Process question
            print("\nAgent: ", end="", flush=True)

            well_name = detect_well_from_query(user_input)
            if well_name:
                answer = summarize_well(query=user_input, well_name=well_name)
                print(answer)
            else:
                print("Please provide a well name to work with.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

# ==================== Testing ====================

if __name__ == "__main__":
    import sys

    # Check if user wants interactive mode or test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("\n" + "=" * 80)
        print("Testing Summarization Agent")
        print("=" * 80)

        result = summarize_well(
            query="Give me a summary of well-1 completion",
            well_name="well-1",
            max_words=200
        )

        print(result)

    else:
        interactive_mode()