"""
Test end-to-end RAG pipeline:

load → embed → store → retrieve → generate answer
"""

from src.ingestion.ingest_news import load_company_news
from src.embeddings.bge_embedder import BGEEmbedder
from src.vector_store.qdrant_store import QdrantVectorStore
from src.retrieval.retriever import Retriever
from src.generation.factory import get_llm_provider
from src.generation.answer_generator import AnswerGenerator


def main():
    question = "Why is Nvidia stock moving recently?"

    print("Loading news...")
    documents = load_company_news(["NVDA"], days_back=7)
    print(f"Loaded {len(documents)} documents")

    print("\nEmbedding documents...")
    embedder = BGEEmbedder()
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.embed_documents(texts)

    for i, doc in enumerate(documents):
        doc["embedding"] = embeddings[i].tolist()

    print("\nStoring in Qdrant...")
    vector_store = QdrantVectorStore(vector_size=768)
    vector_store.upsert(documents)

    print("\nCreating retriever...")
    retriever = Retriever(embedder, vector_store)

    print("\nRetrieving relevant documents...")
    retrieved_docs = retriever.retrieve(
        question=question,
        ticker="NVDA",
        top_k=5
    )

    print(f"Retrieved {len(retrieved_docs)} documents")

    print("\nInitializing LLM...")
    llm = get_llm_provider()

    print("\nGenerating answer...")
    generator = AnswerGenerator(llm)

    answer = generator.generate_answer(
        question=question,
        documents=retrieved_docs
    )

    print("\n===== FINAL ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    main()