"""
File: ingest_news.py

Purpose:
Transforms raw Finnhub news and market data into structured documents
with text and enriched metadata for downstream processing.

Role in Pipeline:
Ingestion Layer – Cleans, formats, and enriches data before embedding and storage.

Notes:
- Does NOT handle embeddings or vector storage
- Keeps separation of concerns for flexibility
"""

from datetime import datetime

from src.ingestion.finnhub_loader import FinnhubLoader


# ---------------------------------------------------------------------------
# Keyword tables for metadata enrichment
# ---------------------------------------------------------------------------

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "earnings":   ["earnings", "revenue", "profit", "eps", "quarterly results",
                   "annual report", "beat estimates", "miss estimates", "guidance"],
    "fed":        ["federal reserve", "fomc", "interest rate", "rate hike",
                   "rate cut", "powell", "monetary policy", "basis points"],
    "inflation":  ["inflation", "cpi", "consumer price index", "pce",
                   "deflation", "price index", "core inflation"],
    "merger":     ["merger", "acquisition", "acquire", "takeover",
                   "buyout", "m&a"],
    "layoffs":    ["layoff", "job cuts", "workforce reduction",
                   "restructuring", "downsizing", "headcount reduction"],
    "product":    ["product launch", "new product", "unveil",
                   "new model", "new service"],
    "regulation": ["sec ", "regulation", "antitrust", "fine", "penalty",
                   "lawsuit", "compliance", "investigation"],
    "macro":      ["gdp", "unemployment rate", "jobs report",
                   "nonfarm payroll", "economic growth", "recession"],
}

_HIGH_IMPACT = [
    "federal reserve", "fomc", "rate hike", "rate cut",
    "earnings", "merger", "acquisition", "sec ", "recession",
    "gdp", "inflation",
]
_MEDIUM_IMPACT = [
    "quarterly results", "layoff", "regulation", "jobs report",
    "product launch", "guidance",
]


def enrich_metadata(text: str) -> dict:
    lower = text.lower()

    topic_tags = [topic for topic, kws in _TOPIC_KEYWORDS.items()
                  if any(kw in lower for kw in kws)]
    event_type = topic_tags[0] if topic_tags else "general"

    if any(kw in lower for kw in _HIGH_IMPACT):
        impact_level = "high"
    elif any(kw in lower for kw in _MEDIUM_IMPACT):
        impact_level = "medium"
    else:
        impact_level = "low"

    return {
        "event_type": event_type,
        "impact_level": impact_level,
        "topic_tags": topic_tags,
    }


# ---------------------------------------------------------------------------
# Company news
# ---------------------------------------------------------------------------

def format_article(article: dict, ticker: str) -> dict:
    headline = article.get("headline", "")
    summary = article.get("summary", "")
    text = f"{headline}. {summary}".strip()

    metadata = {
        "ticker": ticker,
        "category": "company_news",
        "headline": headline,
        "source": article.get("source"),
        "url": article.get("url"),
        "published_at": article.get("datetime"),
        "related": article.get("related"),
    }
    metadata.update(enrich_metadata(text))

    return {"text": text, "metadata": metadata}


def load_company_news(tickers: list[str], days_back: int = 7) -> list[dict]:
    loader = FinnhubLoader()
    documents = []
    for ticker in tickers:
        articles = loader.fetch_company_news(ticker=ticker, days_back=days_back)
        for article in articles:
            if not article.get("url"):
                continue
            documents.append(format_article(article, ticker))
    return documents


# ---------------------------------------------------------------------------
# Market news
# ---------------------------------------------------------------------------

def format_market_article(article: dict) -> dict:
    headline = article.get("headline", "")
    summary = article.get("summary", "")
    text = f"{headline}. {summary}".strip()

    metadata = {
        "ticker": None,
        "category": "market_news",
        "headline": headline,
        "source": article.get("source"),
        "url": article.get("url"),
        "published_at": article.get("datetime"),
        "related": article.get("related"),
    }
    metadata.update(enrich_metadata(text))

    return {"text": text, "metadata": metadata}


def load_market_news(categories: list[str] = None) -> list[dict]:
    if categories is None:
        categories = ["general"]
    loader = FinnhubLoader()
    documents = []
    for category in categories:
        articles = loader.fetch_market_news(category=category)
        for article in articles:
            if not article.get("url"):
                continue
            documents.append(format_market_article(article))
    return documents


# ---------------------------------------------------------------------------
# Stock price / OHLCV data
# ---------------------------------------------------------------------------

def format_candles(ticker: str, candles: dict) -> dict | None:
    if candles.get("s") != "ok" or not candles.get("t"):
        return None

    timestamps = candles["t"]
    opens   = candles["o"]
    highs   = candles["h"]
    lows    = candles["l"]
    closes  = candles["c"]
    volumes = candles["v"]

    lines = [f"{ticker} daily stock prices:"]
    for i in range(len(timestamps)):
        date_str = datetime.fromtimestamp(timestamps[i]).strftime("%Y-%m-%d")
        vol_m = volumes[i] / 1_000_000
        lines.append(
            f"  {date_str}: Open ${opens[i]:.2f}, High ${highs[i]:.2f}, "
            f"Low ${lows[i]:.2f}, Close ${closes[i]:.2f}, Volume {vol_m:.1f}M"
        )

    latest_close = closes[-1]
    first_close  = closes[0]
    pct_change   = ((latest_close - first_close) / first_close) * 100
    period_high  = max(highs)
    period_low   = min(lows)
    lines.append(
        f"Latest close: ${latest_close:.2f}. "
        f"Period range: ${period_low:.2f} - ${period_high:.2f}. "
        f"Period change: {pct_change:+.1f}%."
    )

    text = "\n".join(lines)
    latest_date = datetime.fromtimestamp(timestamps[-1]).strftime("%Y-%m-%d")

    return {
        "text": text,
        "metadata": {
            "ticker": ticker,
            "category": "stock_price",
            "headline": f"{ticker} stock price summary as of {latest_date}",
            "source": "Finnhub",
            "url": f"https://finnhub.io/stock-candles/{ticker}/{latest_date}",
            "published_at": timestamps[-1],
            "related": ticker,
            "event_type": "price_data",
            "impact_level": "medium",
            "topic_tags": ["price", "market_data"],
        },
    }


def load_stock_data(tickers: list[str], days_back: int = 7) -> list[dict]:
    loader = FinnhubLoader()
    documents = []
    for ticker in tickers:
        candles = loader.fetch_stock_candles(ticker, days_back=days_back)
        doc = format_candles(ticker, candles)
        if doc:
            documents.append(doc)
    return documents
