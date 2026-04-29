"""
File: ingest_news.py

Purpose:
Transforms raw Finnhub news articles into structured documents
with text and metadata for downstream processing.

Role in Pipeline:
Ingestion Layer – Cleans and formats data before embedding and storage.

Notes:
- Does NOT handle embeddings or vector storage
- Keeps separation of concerns for flexibility
"""

from src.ingestion.finnhub_loader import FinnhubLoader


def format_article(article: dict, ticker: str) -> dict:
    headline = article.get("headline", "")
    summary = article.get("summary", "")

    text = f"{headline}. {summary}".strip()

    return {
        "text": text,
        "metadata": {
            "ticker": ticker,
            "category": "company_news",
            "headline": headline,
            "source": article.get("source"),
            "url": article.get("url"),
            "published_at": article.get("datetime"),
            "related": article.get("related"),
        },
    }


def load_company_news(tickers: list[str], days_back: int = 7) -> list[dict]:
    loader = FinnhubLoader()
    documents = []

    for ticker in tickers:
        articles = loader.fetch_company_news(ticker=ticker, days_back=days_back)

        for article in articles:
            doc = format_article(article, ticker)
            documents.append(doc)

    return documents