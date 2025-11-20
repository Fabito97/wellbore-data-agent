from typing import Generic, TypeVar, Optional
from pydantic import BaseModel
from pydantic.generics import GenericModel

# Define the type variable used for the generic ApiResponse
T = TypeVar("T")

class ApiResponse(GenericModel, Generic[T]):
    status: str  # "success" or "error"
    message: Optional[str] = None
    data: Optional[T] = None