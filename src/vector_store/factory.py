from config.settings import VECTOR_BACKEND
from .qdrant_store import QdrantVectorStore
from .pgvector_store import PgVectorStore


def get_vector_store():
    backend = VECTOR_BACKEND

    if backend == "qdrant":
        return QdrantVectorStore()

    if backend == "pgvector":
        return PgVectorStore()

    raise ValueError(f"Unsupported vector backend: {backend}")