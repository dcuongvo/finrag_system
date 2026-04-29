# src/test/ingestion/test_ingest_news.py

from src.ingestion.ingest_news import load_company_news


def main():
    documents = load_company_news(["NVDA"], days_back=7)

    print(f"Loaded {len(documents)} documents\n")

    # print first 2 docs
    for doc in documents[:2]:
        print("TEXT:", doc["text"][:100])
        print("METADATA:", doc["metadata"])
        print("-" * 50)


if __name__ == "__main__":
    main()