from chromadb import Client
from chromadb.config import Settings


def get_chroma_client(persist_dir: str = "./data/vector_db"):
    return Client(
        Settings(
            persist_directory=persist_dir,
            # chroma_db_impl="duckdb+parquet
            anonymized_telemetry=False,
            allow_reset=True  # Useful for testing"
        )
    )