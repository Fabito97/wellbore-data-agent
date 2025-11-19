
# Wellbore AI Agent

**Version:** 1.0.0

An AI-powered wellbore analysis system using Retrieval-Augmented Generation (RAG) to answer questions and perform tasks based on uploaded technical documents.

## Features

- **Document Management**: Upload, process, and manage PDF documents.
- **AI-Powered Chat**: Interact with an AI agent to get insights from your documents.
- **Real-time Interaction**: WebSocket support for real-time, interactive conversations.
- **Question & Answering**: Ask questions about your documents and get answers with source citations.
- **Document Summarization**: Generate summaries of uploaded documents.
- **Table Extraction**: Extract tables from documents based on a query.
- **Nodal Analysis**: Perform nodal analysis calculations (currently mocked).
- **Health Check**: Endpoint to monitor the status of the application and its services.

## High-Level Structure

The project is a FastAPI application with a clean and modular architecture.

- `app/`: Main application directory.
  - `main.py`: FastAPI application entry point.
  - `api/`: API endpoints, middleware, and routes.
    - `routes/`: Defines the API endpoints for different functionalities (chat, documents, etc.).
  - `agents/`: Contains the AI agent logic and tools.
  - `core/`: Core application settings and configuration.
  - `models/`: Pydantic models defining the data structures for API requests and responses.
  - `rag/`: Components for the Retrieval-Augmented Generation (RAG) pipeline (chunking, embeddings, vector store).
  - `services/`: Services for interacting with external systems like the LLM and document storage.
  - `utils/`: Utility functions like logging.
- `data/`: Directory for storing uploaded documents, processed data, and the vector database.
- `scripts/`: Utility scripts for testing and managing the application.

## API Endpoints

The API is versioned and all endpoints are prefixed with `/api/v1`.

### Health

- **`GET /health`**
  - **Description**: Checks the health of the application and its connected services (LLM, Vector Store).
  - **Success Response (200)**:
    ```json
    {
      "status": "healthy",
      "services": {
        "llm": "healthy",
        "vector_store": "healthy"
      }
    }
    ```
  - **Failure Response (503)**:
    ```json
    {
      "status": "degraded",
      "services": {
        "llm": "unhealthy",
        "vector_store": "healthy"
      }
    }
    ```

### Documents

- **`POST /documents/upload`**
  - **Description**: Uploads a PDF document for processing and ingestion into the RAG system.
  - **Request**: `multipart/form-data` with a `file` field containing the PDF.
  - **Success Response (200)**: `DocumentUploadResponse` model.
    ```json
    {
      "status": "success",
      "message": "Document uploaded and processed successfully",
      "data": {
        "document_id": "string",
        "filename": "string",
        "status": "indexed",
        "page_count": 0,
        "word_count": 0,
        "table_count": 0,
        "chunk_count": 0,
        "uploaded_at": "string"      
      }
    }
    ```

- **`GET /documents/`**
  - **Description**: Lists all documents in the system.
  - **Success Response (200)**:
    ```json
    {
      "status": "success",
      "total": 0,
      "documents": []
    }
    ```

- **`GET /documents/{document_id}`**
  - **Description**: Retrieves detailed metadata for a specific document.
  - **Success Response (200)**: `DocumentContent` model.

- **`DELETE /documents/{document_id}`**
  - **Description**: Deletes a document and all its associated data.
  - **Success Response (200)**:
    ```json
    {
      "status": "success",
      "message": "Document {document_id} deleted successfully"
    }
    ```

### Chat

- **`POST /chat/`**
  - **Description**: A simple, non-streaming chat endpoint.
  - **Request**: `application/x-www-form-urlencoded` with a `query` field.
  - **Success Response (200)**:
    ```json
    {
      "status": "success",
      "message": "Chat generated successfully",
      "data": "string"
    }
    ```

- **`POST /chat/ask`**
  - **Description**: A question-answering endpoint that returns an answer with confidence and optional sources.
  - **Request**: `application/json` with `question` and optional `include_sources`.
  - **Success Response (200)**:
    ```json
    {
      "status": "success",
      "content": {
        "answer": "string",
        "confidence": 0.0,
        "sources": []
      }
    }
    ```

- **`POST /chat/stream`**
  - **Description**: Streams the response from the LLM for a given query.
  - **Request**: `application/x-www-form-urlencoded` with a `query` field.
  - **Success Response (200)**: A streaming response of `text/plain`.

### WebSocket

- **`WS /ws/`**
  - **Description**: A WebSocket endpoint for real-time, interactive chat.
  - **Message Protocol**:
    - **Client to Server**:
      - Ask a question: `{"type": "question", "content": "...", "options": {...}}`
      - Summarize a document: `{"type": "summarize", "document_id": "...", "max_words": 200}`
      - Extract tables: `{"type": "extract_tables", "query": "...", "top_k": 3}`
    - **Server to Client**:
      - On connection: `{"type": "connected", ...}`
      - Status update: `{"type": "status", "message": "..."}`
      - Answer: `{"type": "answer", ...}`
      - Summary: `{"type": "summary", ...}`
      - Tables: `{"type": "tables", ...}`
      - Error: `{"type": "error", "message": "..."}`

## Response Models

The API uses Pydantic models to ensure type-safe and consistent responses. Key models include:

- **`DocumentUploadResponse`**: Response for a successful document upload.
- **`DocumentContent`**: Detailed information about a processed document.
- **`Message`**: Represents a message in a chat session.
- **`NodalAnalysisInput` / `NodalAnalysisOutput`**: Defines the schema for nodal analysis calculations.

For detailed information on all models, refer to the files in the `app/models/` directory.

## Getting Started

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set up environment variables**:
    - Copy `.env.example` to `.env` and fill in the required values (e.g., `OLLAMA_BASE_URL`, `OLLAMA_MODEL`).
3.  **Run the application**:
    ```bash
    uvicorn app.main:app --reload
    ```
4.  **Access the API documentation**:
    - Once the server is running, navigate to `http://127.0.0.1:8000/docs` for the interactive Swagger UI.
