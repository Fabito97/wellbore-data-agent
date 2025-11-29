from dataclasses import dataclass
from typing import Generic, TypeVar, Optional, Any, Dict
from pydantic import BaseModel
from pydantic.generics import GenericModel

# Define the type variable used for the generic ApiResponse
T = TypeVar("T")

class ApiResponse(GenericModel, Generic[T]):
    status: str  # "success" or "error"
    message: Optional[str] = None
    data: Optional[T] = None



@dataclass
class RetrievalResult:
    """
    Single retrieval result with all necessary information.
    """
    chunk_id: str
    content: str
    page_number: int
    document_id: str
    well_id: str
    well_name: str
    document_type: str
    filename: str
    similarity_score: float
    chunk_type: str  # "text" or "table"
    metadata: Dict[str, Any]


    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"[{self.filename} - Page {self.page_number}]"
            f"(score: {self.similarity_score:.3f})\n"
            f"{self.content[:100]}..."
        )

    @property
    def citation(self) -> str:
        return (
            f"Well:{self.well_name} | document type: {self.document_type} | filename:{self.filename}, "
            f"page: {self.page_number}, chunk_type: {self.chunk_type}"
        )


