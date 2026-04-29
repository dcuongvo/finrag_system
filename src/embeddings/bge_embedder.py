"""
File: bge_embedder.py

Purpose:
Generates document and query embeddings using the BGE embedding model.

Role in Pipeline:
Embedding Layer – Converts financial news documents and user questions
into vectors for semantic search.

Notes:
- Uses a special query instruction for BGE retrieval performance
- Returns NumPy arrays internally; convert to lists before storing in Qdrant
"""

from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.embeddings.base import BaseEmbedder


class BGEEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def _format_query(self, query: str) -> str:
        return f"Represent this sentence for searching relevant passages: {query}"

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def embed_query(self, query: str) -> np.ndarray:
        formatted_query = self._format_query(query)

        return self.model.encode(
            formatted_query,
            convert_to_numpy=True
        )

    def get_model_name(self) -> str:
        return self.model_name