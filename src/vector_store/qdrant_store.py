"""
File: qdrant_store.py

Purpose:
Handles storage and retrieval of embeddings using Qdrant vector database.

Role in Pipeline:
Retrieval Layer – Stores vector embeddings and performs similarity search.

Notes:
- Supports metadata filtering (e.g., ticker, category)
- Designed to be swappable with other vector backends (e.g., pgvector)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from .base import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(self, collection_name="finrag_news", vector_size=384):
        self.collection_name = collection_name
        self.client = QdrantClient(path="./qdrant_data")
        # Create collection (safe recreate for demo)
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    def upsert(self, documents):
        points = []

        for i, doc in enumerate(documents):
            points.append(
                PointStruct(
                    id=i,
                    vector=doc["embedding"],
                    payload={
                        **doc["metadata"],
                        "text": doc["text"]  # keep full text for LLM later
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector, filters=None, top_k=5):
        qdrant_filter = None

        if filters:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                for key, value in filters.items()
            ]

            qdrant_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k
        )

        return results.points