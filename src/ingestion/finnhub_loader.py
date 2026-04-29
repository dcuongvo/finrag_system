"""
File: finnhub_loader.py

Purpose:
Handles fetching financial news data from the Finnhub API.

Role in Pipeline:
Ingestion Layer – Fetches raw data before cleaning and processing.

Notes:
- Requires FINNHUB_API_KEY in .env
- Returns raw API response (no formatting applied)
"""
from datetime import date, timedelta
import finnhub
from config.settings import FINNHUB_API_KEY


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