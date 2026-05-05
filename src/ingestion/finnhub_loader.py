"""
File: finnhub_loader.py

Purpose:
Handles fetching financial news data from the Finnhub API, with a Yahoo Finance
fallback for OHLCV candle data (Finnhub free tier blocks /stock/candle).

Role in Pipeline:
Ingestion Layer – Fetches raw data before cleaning and processing.

Notes:
- Requires FINNHUB_API_KEY in .env
- Returns raw API response (no formatting applied)
- fetch_candles_yf() requires no API key and uses Yahoo Finance via yfinance
"""
from datetime import date, timedelta, datetime

import finnhub
from finnhub.exceptions import FinnhubAPIException

from config.settings import FINNHUB_API_KEY


def fetch_candles_yf(ticker: str, days_back: int = 30) -> dict:
    """Return Finnhub-style candle dict using Yahoo Finance (no API key needed)."""
    try:
        import yfinance as yf
        period = f"{days_back}d"
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return {}
        hist = hist.sort_index()
        timestamps = [int(ts.timestamp()) for ts in hist.index]
        return {
            "s": "ok",
            "t": timestamps,
            "o": hist["Open"].round(2).tolist(),
            "h": hist["High"].round(2).tolist(),
            "l": hist["Low"].round(2).tolist(),
            "c": hist["Close"].round(2).tolist(),
            "v": hist["Volume"].tolist(),
        }
    except Exception as exc:
        print(f"Warning: yfinance fallback failed for {ticker}: {exc}")
        return {}


class FinnhubLoader:
    def __init__(self):
        api_key = FINNHUB_API_KEY

        if not api_key:
            raise ValueError("Missing FINNHUB_API_KEY in .env")

        self.client = finnhub.Client(api_key=api_key)

    def fetch_company_news(self, ticker="AAPL", days_back=7):
        today = date.today()
        start_date = today - timedelta(days=days_back)

        return self.client.company_news(
            ticker,
            _from=str(start_date),
            to=str(today)
        )

    def fetch_market_news(self, category="general"):
        return self.client.general_news(category)

    def fetch_stock_candles(self, ticker: str, days_back: int = 7, resolution: str = "D") -> dict:
        today = date.today()
        start_date = today - timedelta(days=days_back)
        from_ts = int(datetime(start_date.year, start_date.month, start_date.day).timestamp())
        to_ts = int(datetime(today.year, today.month, today.day, 23, 59, 59).timestamp())
        try:
            return self.client.stock_candles(ticker, resolution, from_ts, to_ts)
        except FinnhubAPIException as exc:
            # Free-tier keys often get 403 on /stock/candle; continue without OHLCV docs.
            print(
                f"Warning: Finnhub stock candles skipped for {ticker} "
                f"(HTTP {exc.status_code}): {exc.message}"
            )
            return {}