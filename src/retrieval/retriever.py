"""
File: retriever.py

Purpose:
Retrieves relevant financial news documents from the vector store.

Role in Pipeline:
Retrieval Layer – Converts user questions into embeddings and searches
the vector database using optional metadata filters.
"""


class Retriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, question: str, ticker: str | None = None, top_k: int = 5):
        query_vector = self.embedder.embed_query(question).tolist()

        filters = None
        if ticker:
            filters = {"ticker": ticker}

        results = self.vector_store.search(
            query_vector=query_vector,
            filters=filters,
            top_k=top_k
        )

        documents = []

        for result in results:
            payload = result.payload

            documents.append({
                "score": result.score,
                "text": payload.get("text"),
                "headline": payload.get("headline"),
                "source": payload.get("source"),
                "url": payload.get("url"),
                "ticker": payload.get("ticker"),
                "published_at": payload.get("published_at"),
            })

        return documents