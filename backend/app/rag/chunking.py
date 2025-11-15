"""
Text Chunking - Splits documents into embedding-sized pieces.

We'll implement: Recursive Character Splitter (best balance)
"""

from app.utils.logger import get_logger
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.models.document import DocumentContent, DocumentChunk, TableData
from app.core.config import settings

logger = get_logger(__name__)

@dataclass
class ChunkingConfig:
    """
    Configuration for text chunking.
    """
    # default=1000: Fits in embedding models with room to spare
    chunk_size: int = settings.CHUNK_SIZE

    # default 200: 20% overlap prevents concept splitting
    chunk_overlap: int = settings.CHUNK_OVERLAP

    # Separator hierarchy: Prioritize semantic boundaries (paragraphs > sentences > words)
    separators: List[str] = None

    def __post_init__(self):
        """Set default separators after initialization."""
        if self.separators is None:
            self.separators = [
                 "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentences
                " ",     # Words
                ""       # Characters (last resort)
            ]


class DocumentChunker:
    """
    Chunks documents into embedding-ready pieces - uses RecursiveCharacterTextSplitter

    Others chunking strategies are: SemanticChunker, TokenBasedChunker, etc.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize chunker with configuration.

        Args:
            config: Optional custom config, uses defaults if not provided
        """
        self.config = config or ChunkingConfig()

        # LangChain's RecursiveCharacterTextSplitter
        # - Tries first separator ("\n\n")
        # - If chunk still too big, tries next separator ("\n")
        # - Continues until chunk is right size
        # - More intelligent than naive splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len, # Use character count
        )

        logger.info(
            f"DocumentChunker initialized: "
            f"chunk_size={self.config.chunk_size} "
            f"overlap={self.config.chunk_overlap}"

        )

    def chunk_document(self, document: DocumentContent):
        """
        Chunk a complete document into pieces (page by page) maintaining tables separately.

        Args:
            document: Processed document with text and tables

        Returns:
            List of DocumentChunk objects ready for embedding
        """
        logger.info(f"Chunking document: {document.document_id}")

        all_chunks = [] # Global chunk index across all pages
        chunk_index = 0

        # Process each page separately
        for page in document.pages:
            # Chunk text content
            text_chunks = self._chunk_page_text(
                document=document,
                page_number=page.page_number,
                text=page.text,
                start_index=chunk_index,
            )
            all_chunks.extend(text_chunks)
            chunk_index += len(text_chunks)

            # Chunk tables from this page
            table_chunks = self._chunk_page_tables(
                document=document,
                page_number=page.page_number,
                tables=page.tables,
                start_index=chunk_index,
            )
            all_chunks.extend(table_chunks)
            chunk_index += len(table_chunks)

        logger.info(f"Document {document.document_id} chunked into {len(all_chunks)} pieces")

        return all_chunks

    def _chunk_page_text(
            self,
            document: DocumentContent,
            page_number: int,
            text: str,
            start_index: int
    ) -> List[DocumentChunk]:
        """
         Chunk text from a single page.

         Process:
         1. Use LangChain splitter to split text
         2. Wrap each piece in DocumentChunk model
         3. Add rich metadata (page, doc ID, filename)
         """
        if not text or not text.strip():
            return []

        # Split text using recursive splitter - returns List[str]
        text_pieces = self.text_splitter.split_text(text)

        chunks = []
        for i, piece in enumerate(text_pieces):
            chunk = DocumentChunk(
                chunk_id=self._generate_chunk_id(document.document_id, start_index + i),
                document_id=document.document_id,
                content=piece,
                page_number=page_number,
                chunk_index=start_index + i,
                metadata={
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "page_number": page_number,
                    "chunk_type": "text",
                    "source": "page_content"
                }
            )
            chunks.append(chunk)

        return chunks


    def _chunk_page_tables(
            self,
            document: DocumentContent,
            page_number: int,
            tables: List[TableData],
            start_index: int,
    ) -> List[DocumentChunk]:
        """
        Create chunks from tables (Each table = one chunk) with table-specific metadata
        """
        chunks = []

        for i, table in enumerate(tables):
            # Convert table to markdown - LLMs are trained on markdown tables, Preserves structure in text format
            table_markdown = table.to_markdown()

            # Add context header - Helps LLM understand what it's looking at
            content = f"""
                    [TABLE from page {page_number}]
                    Columns: {', '.join(table.headers)}
                    Rows: {table.row_count}
                    
                    {table_markdown}
                    """
            chunk = DocumentChunk(
                chunk_id=self._generate_chunk_id(document.document_id, start_index + i),
                document_id=document.document_id,
                content=content.strip(),
                page_number=page_number,
                chunk_index=start_index + i,
                metadata={
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "page_number": page_number,
                    "chunk_type": "table",
                    "column_count": table.column_count,
                    "row_count": table.row_count,
                    "headers": ", ".join(table.headers),
                }
            )
            chunks.append(chunk)

        return chunks


    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """
        Generate unique chunk ID {document_id}_chunk_{index} - Easy to reconstruct from parts
        """
        return f"{document_id}_chunk_{chunk_index}"


    def rechunk_with_size(
            self,
            text: str,
            chunk_size: int,
            chunk_overlap: int
    ) -> List[str]:
        """
        Utility: Rechunk text with custom size - Adapt to different embedding models
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.config.separators,
            length_function=len
        )
        return splitter.split_text(text)


# ==================== Convenience Functions ====================

def chunk_document(
        document: DocumentContent,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
) -> List[DocumentChunk]:
    """
    Convenience function for quick chunking - Hides complexity of config and classes
    """
    config = ChunkingConfig(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP
    )
    chunker = DocumentChunker(config)
    return chunker.chunk_document(document)