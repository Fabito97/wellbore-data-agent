from chromadb import Client
from chromadb.config import Settings
from langchain_chroma import Chroma

from app.core.config import settings


def get_chroma_client(embedding_func, persist_dir: str = settings.VECTOR_DB_DIR):
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_func,
        persist_directory=persist_dir,
        collection_metadata={
            "hnsw:space": settings.CHROMA_DISTANCE_METRIC # cosine defaults
        }
    )