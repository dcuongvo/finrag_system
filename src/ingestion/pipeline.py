"""
File: pipeline.py

Purpose:
Orchestrates fetch → embed → store for the vector index, and sync to AWS RDS.

Role in Pipeline:
Ingestion Layer – Callable from the Gradio UI or scripts without starting chat.
"""

from dataclasses import dataclass

from src.ingestion.finnhub_loader import FinnhubLoader, fetch_candles_yf
from src.ingestion.ingest_news import load_company_news, load_market_news, load_stock_data
from src.vector_store.pgvector_store import PgVectorStore


@dataclass
class IngestionResult:
    total_fetched: int
    new_embedded: int
    already_indexed: int
    company_count: int
    market_count: int
    stock_count: int
    company_docs: list
    market_docs: list
    stock_docs: list
    raw_candles: dict
    backend: str

    @property
    def all_docs(self) -> list:
        return self.company_docs + self.market_docs + self.stock_docs

    def summary(self) -> str:
        return (
            f"**Index build complete** ({self.backend})\n\n"
            f"- Fetched: **{self.total_fetched}** documents "
            f"({self.company_count} company, {self.market_count} market, "
            f"{self.stock_count} price summaries)\n"
            f"- Newly embedded & stored: **{self.new_embedded}**\n"
            f"- Already in index: **{self.already_indexed}**"
        )


def fetch_raw_candles(tickers: list[str], days_back: int = 30) -> dict:
    loader = FinnhubLoader()
    raw_candles: dict = {}
    for ticker in tickers:
        candles = loader.fetch_stock_candles(ticker, days_back=days_back)
        if not (candles.get("s") == "ok" and candles.get("t")):
            candles = fetch_candles_yf(ticker, days_back=days_back)
        if candles.get("s") == "ok" and candles.get("t"):
            raw_candles[ticker] = candles
    return raw_candles


def fetch_documents(tickers: list[str], days_back: int) -> IngestionResult:
    company_docs = load_company_news(tickers, days_back=days_back)
    market_docs = load_market_news(["general"])
    stock_docs = load_stock_data(tickers, days_back=days_back)
    raw_candles = fetch_raw_candles(tickers)

    all_docs = company_docs + market_docs + stock_docs
    return IngestionResult(
        total_fetched=len(all_docs),
        new_embedded=0,
        already_indexed=0,
        company_count=len(company_docs),
        market_count=len(market_docs),
        stock_count=len(stock_docs),
        company_docs=company_docs,
        market_docs=market_docs,
        stock_docs=stock_docs,
        raw_candles=raw_candles,
        backend="",
    )


def index_documents(embedder, vector_store, documents: list[dict]) -> tuple[int, int]:
    existing_urls = vector_store.get_existing_urls()
    new_docs = [d for d in documents if d["metadata"].get("url") not in existing_urls]

    if new_docs:
        texts = [doc["text"] for doc in new_docs]
        embeddings = embedder.embed_documents(texts)
        for i, doc in enumerate(new_docs):
            doc["embedding"] = embeddings[i].tolist()
        vector_store.upsert(new_docs)

    return len(new_docs), len(documents) - len(new_docs)


def run_ingestion(embedder, vector_store, tickers: list[str], days_back: int, backend: str) -> IngestionResult:
    result = fetch_documents(tickers, days_back)
    new_count, existing_count = index_documents(embedder, vector_store, result.all_docs)
    result.new_embedded = new_count
    result.already_indexed = existing_count
    result.backend = backend
    return result


def sync_to_aws(vector_store, aws_postgres_url: str) -> str:
    if not aws_postgres_url:
        return (
            "**AWS sync skipped** — set `AWS_POSTGRES_URL` in `.env` "
            "(your RDS connection string)."
        )

    if not hasattr(vector_store, "export_documents"):
        return f"**AWS sync failed** — `{type(vector_store).__name__}` does not support export."

    documents = vector_store.export_documents()
    if not documents:
        return "**AWS sync skipped** — local index is empty. Click **Build local index** first."

    aws_store = PgVectorStore(postgres_url=aws_postgres_url)
    try:
        aws_store.upsert(documents)
        host_hint = aws_postgres_url.split("@")[-1].split("/")[0] if "@" in aws_postgres_url else "RDS"
        return (
            f"**AWS sync complete**\n\n"
            f"- Uploaded **{len(documents)}** documents to `{host_hint}`\n"
            f"- Table: `{aws_store.table_name}`"
        )
    finally:
        aws_store.close()
