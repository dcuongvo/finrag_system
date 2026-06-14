"""
Test end-to-end:
load → embed → store → search
"""

from config.settings import DAYS_BACK, TOP_K
from src.ingestion.ingest_news import load_company_news
from src.embeddings.factory import get_embedder
from src.vector_store.factory import get_vector_store


def main():
    print("Loading news...")
    documents = load_company_news(["NVDA"], days_back=DAYS_BACK)

    print(f"Loaded {len(documents)} documents")

    print("\nEmbedding documents...")
    embedder = get_embedder()

    texts = [doc["text"] for doc in documents]
    embeddings = embedder.embed_documents(texts)

    # attach embeddings
    for i, doc in enumerate(documents):
        doc["embedding"] = embeddings[i].tolist()

    print("Storing in Qdrant...")
    vector_store = get_vector_store()

    vector_store.upsert(documents)

    print("\nSearching...")
    query = "Why is Nvidia stock moving?"
    query_vector = embedder.embed_query(query).tolist()

    results = vector_store.search(
        query_vector=query_vector,
        filters={"ticker": "NVDA"},
        top_k=TOP_K
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