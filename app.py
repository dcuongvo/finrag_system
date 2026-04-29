"""
File: app.py

Purpose:
Runs the Gradio chat demo for FinRAG.

Role in Pipeline:
Application Layer – Initializes the RAG components once, then lets users
ask financial questions through a chatbot interface.

Notes:
- Loads recent news at startup for demo purposes
- Uses lightweight ticker memory for follow-up questions
"""

import gradio as gr

from src.ingestion.ingest_news import load_company_news
from src.embeddings.bge_embedder import BGEEmbedder
from src.vector_store.qdrant_store import QdrantVectorStore
from src.retrieval.retriever import Retriever
from src.generation.factory import get_llm_provider
from src.generation.answer_generator import AnswerGenerator


# -------------------------
# Initialize system once
# -------------------------
print("Initializing FinRAG system...")

TICKERS = ["NVDA", "AAPL", "TSLA", "MSFT"]

embedder = BGEEmbedder()
vector_store = QdrantVectorStore(vector_size=768)

documents = load_company_news(TICKERS, days_back=7)

texts = [doc["text"] for doc in documents]
embeddings = embedder.embed_documents(texts)

for i, doc in enumerate(documents):
    doc["embedding"] = embeddings[i].tolist()

vector_store.upsert(documents)

retriever = Retriever(embedder, vector_store)
llm = get_llm_provider()
generator = AnswerGenerator(llm)

print("FinRAG system ready.")


# -------------------------
# Helper functions
# -------------------------
def detect_ticker(text: str):
    upper_text = text.upper()

    for ticker in TICKERS:
        if ticker in upper_text:
            return ticker

    return None


def find_last_ticker(history):
    if not history:
        return None

    for item in reversed(history):
        # Gradio history format can vary by version
        if isinstance(item, dict):
            content = item.get("content", "")
            combined = content.upper()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            user_msg = item[0] or ""
            assistant_msg = item[1] or ""
            combined = f"{user_msg} {assistant_msg}".upper()
        else:
            combined = str(item).upper()

        for ticker in TICKERS:
            if ticker in combined:
                return ticker

    return None


# -------------------------
# Chat function
# -------------------------
def chat_response(message, history):
    ticker = detect_ticker(message)

    if ticker is None:
        ticker = find_last_ticker(history)

    docs = retriever.retrieve(
        question=message,
        ticker=ticker,
        top_k=5,
    )

    answer = generator.generate_answer(message, docs)

    sources = "\n\n### Sources\n" + "\n".join([
        f"- [{d['headline']}]({d['url']}) ({d['source']})"
        for d in docs
    ])

    if ticker:
        ticker_note = f"\n\n**Detected ticker:** `{ticker}`"
    else:
        ticker_note = "\n\n**Detected ticker:** None. Searching across available news."

    return answer + ticker_note + sources


# -------------------------
# Chat UI
# -------------------------
demo = gr.ChatInterface(
    fn=chat_response,
    title="📊 FinRAG: Financial Intelligence Assistant",
    description=(
        "Ask questions about recent financial news. "
        "The system retrieves relevant news using vector search and generates source-grounded answers."
    ),
    examples=[
        "Why is NVDA stock moving recently?",
        "What recent news is affecting Tesla?",
        "What is happening with Apple?",
        "What are the main risks?",
    ],
)


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)