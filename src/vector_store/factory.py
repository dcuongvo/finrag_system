"""
File: factory.py

Purpose:
Creates the configured vector store backend.

Role in Pipeline:
Retrieval Layer – Centralizes vector backend selection (Qdrant, pgvector).
"""

from config.settings import VECTOR_BACKEND
from .qdrant_store import QdrantVectorStore
from .pgvector_store import PgVectorStore


def get_vector_store():
    backend = VECTOR_BACKEND.lower()

    if backend == "qdrant":
        return QdrantVectorStore()

    if backend == "pgvector":
        return PgVectorStore()

    raise ValueError(f"Unsupported vector backend: {backend}")