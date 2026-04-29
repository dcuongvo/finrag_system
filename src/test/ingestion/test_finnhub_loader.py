from src.ingestion.finnhub_loader import FinnhubLoader


def main():
    loader = FinnhubLoader()

    news = loader.fetch_company_news("NVDA", days_back=7)

    print(f"Fetched {len(news)} articles")

    for article in news[:3]:
        print(article.get("headline"))
        print(article.get("source"))
        print(article.get("url"))
        print("-" * 40)


if __name__ == "__main__":
    main()