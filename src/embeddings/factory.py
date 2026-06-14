"""
File: factory.py

Purpose:
Creates the configured embedding provider.

Role in Pipeline:
Embedding Layer – Centralizes embedder selection so the backend can
be changed through configuration.
"""

from config.settings import EMBEDDING_BACKEND, EMBEDDING_MODEL


def get_embedder():
    backend = EMBEDDING_BACKEND.lower()

    if backend == "bge":
        from src.embeddings.bge_embedder import BGEEmbedder
        return BGEEmbedder(model_name=EMBEDDING_MODEL)

    raise ValueError(f"Unsupported embedding backend: {backend}")
