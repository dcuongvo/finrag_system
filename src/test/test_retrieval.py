"""
Test end-to-end:
load → embed → store → search
"""

from src.ingestion.ingest_news import load_company_news
from src.embeddings.bge_embedder import BGEEmbedder
from src.vector_store.qdrant_store import QdrantVectorStore


def main():
    print("Loading news...")
    documents = load_company_news(["NVDA"], days_back=7)

    print(f"Loaded {len(documents)} documents")

    print("\nEmbedding documents...")
    embedder = BGEEmbedder()

    texts = [doc["text"] for doc in documents]
    embeddings = embedder.embed_documents(texts)

    # attach embeddings
    for i, doc in enumerate(documents):
        doc["embedding"] = embeddings[i].tolist()

    print("Storing in Qdrant...")
    vector_store = QdrantVectorStore(vector_size=768)

    vector_store.upsert(documents)

    print("\nSearching...")
    query = "Why is Nvidia stock moving?"
    query_vector = embedder.embed_query(query).tolist()

    results = vector_store.search(
        query_vector=query_vector,
        filters={"ticker": "NVDA"},
        top_k=5
    )

    print("\nResults:\n")

    for r in results:
        payload = r.payload

        print("Score:", r.score)
        print("Headline:", payload.get("headline"))
        print("Source:", payload.get("source"))
        print("URL:", payload.get("url"))
        print("-" * 50)


if __name__ == "__main__":
    main()