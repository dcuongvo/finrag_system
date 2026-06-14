"""
File: pgvector_store.py

Purpose:
Stores and searches embeddings in PostgreSQL with the pgvector extension.

Role in Pipeline:
Retrieval Layer – AWS RDS–compatible vector backend; swap in via
VECTOR_BACKEND=pgvector and POSTGRES_URL in .env.
"""

import json
import uuid
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb

from config.settings import COLLECTION_NAME, POSTGRES_URL, VECTOR_SIZE
from .base import VectorStore


@dataclass
class VectorSearchResult:
    score: float
    payload: dict


class PgVectorStore(VectorStore):
    def __init__(
        self,
        postgres_url: str | None = None,
        table_name: str | None = None,
        vector_size: int | None = None,
    ):
        if not (postgres_url or POSTGRES_URL):
            raise ValueError(
                "Missing POSTGRES_URL in .env (required when VECTOR_BACKEND=pgvector)"
            )

        self.table_name = table_name or COLLECTION_NAME
        self.vector_size = vector_size or VECTOR_SIZE
        self.conn = psycopg.connect(postgres_url or POSTGRES_URL)
        register_vector(self.conn)
        self._ensure_schema()

    def _ensure_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        id UUID PRIMARY KEY,
                        url TEXT UNIQUE NOT NULL,
                        text TEXT NOT NULL,
                        ticker TEXT,
                        category TEXT,
                        headline TEXT,
                        source TEXT,
                        published_at BIGINT,
                        related TEXT,
                        event_type TEXT,
                        impact_level TEXT,
                        topic_tags JSONB,
                        embedding vector({})
                    )
                    """
                ).format(
                    sql.Identifier(self.table_name),
                    sql.Literal(self.vector_size),
                )
            )
        self.conn.commit()

    @staticmethod
    def _point_id(url: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, url))

    def get_existing_urls(self) -> set[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT url FROM {} WHERE url IS NOT NULL"
                ).format(sql.Identifier(self.table_name))
            )
            return {row[0] for row in cur.fetchall()}

    def upsert(self, documents):
        insert_sql = sql.SQL(
            """
            INSERT INTO {} (
                id, url, text, ticker, category, headline, source,
                published_at, related, event_type, impact_level, topic_tags, embedding
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (url) DO UPDATE SET
                text = EXCLUDED.text,
                ticker = EXCLUDED.ticker,
                category = EXCLUDED.category,
                headline = EXCLUDED.headline,
                source = EXCLUDED.source,
                published_at = EXCLUDED.published_at,
                related = EXCLUDED.related,
                event_type = EXCLUDED.event_type,
                impact_level = EXCLUDED.impact_level,
                topic_tags = EXCLUDED.topic_tags,
                embedding = EXCLUDED.embedding
            """
        ).format(sql.Identifier(self.table_name))

        with self.conn.cursor() as cur:
            for doc in documents:
                meta = doc["metadata"]
                url = meta.get("url") or doc["text"]
                cur.execute(
                    insert_sql,
                    (
                        self._point_id(url),
                        url,
                        doc["text"],
                        meta.get("ticker"),
                        meta.get("category"),
                        meta.get("headline"),
                        meta.get("source"),
                        meta.get("published_at"),
                        meta.get("related"),
                        meta.get("event_type"),
                        meta.get("impact_level"),
                        Jsonb(meta.get("topic_tags") or []),
                        doc["embedding"],
                    ),
                )
        self.conn.commit()

    def search(self, query_vector, filters=None, top_k=5):
        where = sql.SQL("")
        params: list = [query_vector, query_vector, top_k]

        if filters:
            conditions = []
            filter_params = []
            for key, value in filters.items():
                conditions.append(
                    sql.SQL("{} = %s").format(sql.Identifier(key))
                )
                filter_params.append(value)
            where = sql.SQL("WHERE ") + sql.SQL(" AND ").join(conditions)
            params = [query_vector] + filter_params + [query_vector, top_k]

        query = sql.SQL(
            """
            SELECT text, ticker, category, headline, source, url,
                   published_at, related, event_type, impact_level, topic_tags,
                   1 - (embedding <=> %s::vector) AS score
            FROM {}
            {}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
        ).format(sql.Identifier(self.table_name), where)

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        results = []
        for row in rows:
            (
                text,
                ticker,
                category,
                headline,
                source,
                url,
                published_at,
                related,
                event_type,
                impact_level,
                topic_tags,
                score,
            ) = row
            results.append(
                VectorSearchResult(
                    score=float(score),
                    payload={
                        "text": text,
                        "ticker": ticker,
                        "category": category,
                        "headline": headline,
                        "source": source,
                        "url": url,
                        "published_at": published_at,
                        "related": related,
                        "event_type": event_type,
                        "impact_level": impact_level,
                        "topic_tags": topic_tags,
                    },
                )
            )
        return results

    def export_documents(self) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT text, ticker, category, headline, source, url,
                           published_at, related, event_type, impact_level,
                           topic_tags, embedding
                    FROM {}
                    """
                ).format(sql.Identifier(self.table_name))
            )
            rows = cur.fetchall()

        documents = []
        for row in rows:
            (
                text,
                ticker,
                category,
                headline,
                source,
                url,
                published_at,
                related,
                event_type,
                impact_level,
                topic_tags,
                embedding,
            ) = row
            documents.append({
                "text": text,
                "metadata": {
                    "ticker": ticker,
                    "category": category,
                    "headline": headline,
                    "source": source,
                    "url": url,
                    "published_at": published_at,
                    "related": related,
                    "event_type": event_type,
                    "impact_level": impact_level,
                    "topic_tags": topic_tags,
                },
                "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
            })
        return documents

    def close(self):
        self.conn.close()
