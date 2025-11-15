"""
API Dependencies - Shared dependencies for FastAPI routes.

Teaching: Dependency Injection in FastAPI
- Dependencies are reusable functions
- Injected into route handlers automatically
- Enable clean separation of concerns
- Easy to mock for testing

Common uses:
- Database sessions
- Authentication
- Service instances
- Configuration
"""

from typing import Generator
from fastapi import Depends, HTTPException, status

from app.core.config import settings, Settings
from app.services.document_service import get_document_service, DocumentService
from app.services.llm_service import get_llm_service, LLMService
from app.agents.simple_agent import get_simple_agent, SimpleAgent


# ==================== Configuration ====================

def get_settings() -> Settings:
    """
    Get application settings.

    Teaching: Settings dependency
    - Centralized configuration
    - Easy to override in tests
    - Type-safe access

    Usage:
        @app.get("/config")
        def show_config(settings: Settings = Depends(get_settings)):
            return {"model": settings.OLLAMA_MODEL}
    """
    return settings


# ==================== Services ====================

def get_document_service_dep() -> DocumentService:
    """
    Get document service instance.

    Teaching: Service dependency
    - Returns singleton instance
    - All routes share same service
    - Consistent state across requests

    Usage:
        @app.post("/documents")
        def upload(
            file: UploadFile,
            service: DocumentService = Depends(get_document_service_dep)
        ):
            return service.ingest_document(file)
    """
    return get_document_service()


def get_llm_service_dep() -> LLMService:
    """
    Get LLM service instance.

    Usage:
        @app.post("/generate")
        def generate(
            prompt: str,
            llm: LLMService = Depends(get_llm_service_dep)
        ):
            return llm.generate(prompt)
    """
    return get_llm_service()


def get_agent_dep() -> SimpleAgent:
    """
    Get agent instance.

    Teaching: Agent dependency
    - Agent already has LLM and retriever
    - Routes just call agent methods
    - Clean separation: routes handle HTTP, agent handles logic

    Usage:
        @app.post("/ask")
        def ask_question(
            question: str,
            agent: SimpleAgent = Depends(get_agent_dep)
        ):
            response = agent.answer(question)
            return {"answer": response.answer}
    """
    return get_simple_agent()


# ==================== Validation ====================

def validate_file_upload(
        content_type: str,
        content_length: int
) -> None:
    """
    Validate uploaded file.

    Teaching: Validation as dependency
    - Checks before processing
    - Raises HTTPException if invalid
    - Reusable across routes

    Args:
        content_type: MIME type of file
        content_length: Size in bytes

    Raises:
        HTTPException: If validation fails

    Usage:
        @app.post("/upload")
        def upload(
            file: UploadFile,
            _: None = Depends(validate_file_upload)
        ):
            # File is already validated here
            ...
    """
    # Check file type
    allowed_types = ["application/pdf"]
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Allowed: {', '.join(allowed_types)}"
        )

    # Check file size
    if content_length > settings.MAX_UPLOAD_SIZE:
        max_mb = settings.MAX_UPLOAD_SIZE / 1024 / 1024
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_mb:.0f}MB"
        )


# ==================== Teaching Notes ====================
"""
KEY CONCEPTS:

1. **Dependency Injection Benefits**:

   Without DI:
   ```python
   @app.get("/data")
   def get_data():
       service = DocumentService()  # Creates new instance
       return service.get_stats()    # Each request = new service
   ```

   With DI:
   ```python
   @app.get("/data")
   def get_data(service: DocumentService = Depends(get_document_service_dep)):
       return service.get_stats()    # Reuses singleton
   ```

2. **Testing with DI**:

   ```python
   # In tests, override dependency
   def mock_service():
       return MockDocumentService()

   app.dependency_overrides[get_document_service_dep] = mock_service

   # Now all routes use mock service!
   ```

3. **Dependency Chain**:

   FastAPI resolves dependencies recursively:
   ```python
   def get_db():
       return Database()

   def get_user(db: Database = Depends(get_db)):
       return db.get_user()

   @app.get("/profile")
   def profile(user = Depends(get_user)):
       # FastAPI calls: get_db() → get_user() → profile()
       return user
   ```

4. **Validation Dependencies**:

   Can be used for:
   - Authentication
   - Rate limiting
   - Input validation
   - Feature flags

   Example:
   ```python
   def verify_auth(token: str = Header()):
       if not valid(token):
           raise HTTPException(401)

   @app.get("/protected")
   def protected(user = Depends(verify_auth)):
       # Only called if auth succeeds
       return data
   ```

5. **Performance**:

   Dependencies evaluated once per request:
   ```python
   @app.get("/data")
   def handler(
       service1 = Depends(get_service),  # Called once
       service2 = Depends(get_service),  # Reuses same instance!
   ):
       ...
   ```

This pattern makes routes thin and testable!
"""