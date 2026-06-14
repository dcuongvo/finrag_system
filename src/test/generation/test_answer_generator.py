"""
Test end-to-end RAG pipeline:

load → embed → store → retrieve → generate answer
"""

from config.settings import DAYS_BACK, TOP_K
from src.ingestion.ingest_news import load_company_news
from src.embeddings.factory import get_embedder
from src.vector_store.factory import get_vector_store
from src.retrieval.retriever import Retriever
from src.generation.factory import get_llm_provider
from src.generation.answer_generator import AnswerGenerator


def main():
    question = "Why is Nvidia stock moving recently?"

    print("Loading news...")
    documents = load_company_news(["NVDA"], days_back=DAYS_BACK)
    print(f"Loaded {len(documents)} documents")

    print("\nEmbedding documents...")
    embedder = get_embedder()
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.embed_documents(texts)

    for i, doc in enumerate(documents):
        doc["embedding"] = embeddings[i].tolist()

    print("\nStoring in Qdrant...")
    vector_store = get_vector_store()
    vector_store.upsert(documents)

    print("\nCreating retriever...")
    retriever = Retriever(embedder, vector_store)

    print("\nRetrieving relevant documents...")
    retrieved_docs = retriever.retrieve(
        question=question,
        ticker="NVDA",
        top_k=TOP_K
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