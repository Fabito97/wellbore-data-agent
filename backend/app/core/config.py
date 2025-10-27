# from pydantic_settings import BaseSettings
# from typing import List

# class Settings(BaseSettings):
#     # API Settings
#     API_V1_STR: str = "/api"
#     PROJECT_NAME: str = "Wellbore AI Agent"
    
#     # CORS
#     CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
#     # Ollama
#     OLLAMA_BASE_URL: str = "http://localhost:11434"
#     OLLAMA_MODEL: str = "phi3:mini"
    
#     # ChromaDB
#     CHROMA_PERSIST_DIR: str = "./data/vector_db"
    
#     # File Upload
#     UPLOAD_DIR: str = "./data/uploads"
#     MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    
#     # Embeddings
#     EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
#     class Config:
#         env_file = ".env"
#         case_sensitive = True

# settings = Settings()