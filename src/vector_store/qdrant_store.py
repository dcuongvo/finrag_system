"""
File: qdrant_store.py

Purpose:
Handles storage and retrieval of embeddings using Qdrant vector database.

Role in Pipeline:
Retrieval Layer – Stores vector embeddings and performs similarity search.

Notes:
- Supports metadata filtering (e.g., ticker, category)
- Designed to be swappable with other vector backends (e.g., pgvector)
- Uses URL-derived UUIDs as point IDs so upsert is idempotent
- Collection persists between restarts; only new documents are added
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from config.settings import COLLECTION_NAME, QDRANT_PATH, VECTOR_SIZE
from .base import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str | None = None,
        vector_size: int | None = None,
        qdrant_path: str | None = None,
    ):
        self.collection_name = collection_name or COLLECTION_NAME
        self.client = QdrantClient(path=qdrant_path or QDRANT_PATH)
        self._ensure_collection(vector_size or VECTOR_SIZE)

    def _ensure_collection(self, vector_size: int):
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    @staticmethod
    def _point_id(url: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, url))

    def get_existing_urls(self) -> set[str]:
        urls = set()
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=next_offset,
                with_payload=["url"],
                with_vectors=False,
            )
            for p in points:
                url = p.payload.get("url")
                if url:
                    urls.add(url)
            if next_offset is None:
                break
        return urls

    def upsert(self, documents):
        points = []
        for doc in documents:
            url = doc["metadata"].get("url") or doc["text"]
            points.append(
                PointStruct(
                    id=self._point_id(url),
                    vector=doc["embedding"],
                    payload={
                        **doc["metadata"],
                        "text": doc["text"]
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
                FieldCondition(key=key, match=MatchValue(value=value))
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

    def export_documents(self) -> list[dict]:
        documents = []
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                payload = dict(point.payload)
                text = payload.pop("text", "")
                documents.append({
                    "text": text,
                    "metadata": payload,
                    "embedding": point.vector,
                })
            if next_offset is None:
                break
        return documents
