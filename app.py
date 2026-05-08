"""
File: app.py

Purpose:
Gradio Blocks UI for FinRAG – Chat, Stocks dashboard, and News Feed tabs.

Role in Pipeline:
Application Layer – Initializes the RAG components once, then lets users
interact via chat, browse stock prices, and explore the news index.
"""

import pandas as pd
import gradio as gr
from datetime import datetime

from src.ingestion.ingest_news import load_company_news, load_market_news, load_stock_data
from src.ingestion.finnhub_loader import FinnhubLoader, fetch_candles_yf
from src.embeddings.bge_embedder import BGEEmbedder
from src.vector_store.qdrant_store import QdrantVectorStore
from src.retrieval.retriever import Retriever
from src.generation.factory import get_llm_provider
from src.generation.answer_generator import AnswerGenerator


# ── Initialization ────────────────────────────────────────────────────────────
print("Initializing FinRAG system...")

TICKERS = ["NVDA", "AAPL", "TSLA", "MSFT"]

embedder = BGEEmbedder()
vector_store = QdrantVectorStore(vector_size=768)

print("Fetching company news...")
company_docs = load_company_news(TICKERS, days_back=7)

print("Fetching market news...")
market_docs = load_market_news(["general"])

print("Fetching stock price data...")
stock_docs = load_stock_data(TICKERS, days_back=7)

# Fetch raw candle data for the price chart (30 days).
# Try Finnhub first; fall back to Yahoo Finance (free, no key needed).
_loader = FinnhubLoader()
raw_candles: dict[str, dict] = {}
for _ticker in TICKERS:
    _candles = _loader.fetch_stock_candles(_ticker, days_back=30)
    if not (_candles.get("s") == "ok" and _candles.get("t")):
        print(f"Finnhub candles unavailable for {_ticker}, trying Yahoo Finance...")
        _candles = fetch_candles_yf(_ticker, days_back=30)
    if _candles.get("s") == "ok" and _candles.get("t"):
        raw_candles[_ticker] = _candles

all_docs = company_docs + market_docs + stock_docs
print(
    f"Fetched {len(all_docs)} total documents "
    f"({len(company_docs)} company news, {len(market_docs)} market news, "
    f"{len(stock_docs)} stock price summaries)."
)

existing_urls = vector_store.get_existing_urls()
new_docs = [d for d in all_docs if d["metadata"].get("url") not in existing_urls]
print(
    f"{len(new_docs)} new documents to embed and store "
    f"({len(all_docs) - len(new_docs)} already indexed)."
)

if new_docs:
    texts = [doc["text"] for doc in new_docs]
    embeddings = embedder.embed_documents(texts)
    for i, doc in enumerate(new_docs):
        doc["embedding"] = embeddings[i].tolist()
    vector_store.upsert(new_docs)
    print(f"Stored {len(new_docs)} new documents.")
else:
    print("No new documents — index is up to date.")

retriever = Retriever(embedder, vector_store)
llm = get_llm_provider()
generator = AnswerGenerator(llm)

print("FinRAG system ready.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts_to_date(ts) -> str:
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    return str(ts) if ts else ""


def detect_ticker(text: str) -> str | None:
    upper = text.upper()
    for t in TICKERS:
        if t in upper:
            return t
    return None


def find_last_ticker(history: list) -> str | None:
    for item in reversed(history or []):
        content = item.get("content", "") if isinstance(item, dict) else str(item)
        for t in TICKERS:
            if t in content.upper():
                return t
    return None


# ── Static data for Stocks / News tabs ───────────────────────────────────────

def _build_price_df() -> pd.DataFrame:
    rows = []
    for ticker, candles in raw_candles.items():
        for i, ts in enumerate(candles["t"]):
            rows.append({
                "Date": _ts_to_date(ts),
                "Ticker": ticker,
                "Close": round(candles["c"][i], 2),
            })
    if not rows:
        return pd.DataFrame(columns=["Date", "Ticker", "Close"])
    return pd.DataFrame(rows)


def _build_stock_summary_md() -> str:
    if not raw_candles:
        return "_No stock price data available (free-tier API limit may apply)._"
    parts = []
    for ticker, candles in raw_candles.items():
        closes = candles["c"]
        highs  = candles["h"]
        lows   = candles["l"]
        latest = closes[-1]
        first  = closes[0]
        pct    = ((latest - first) / first) * 100
        arrow  = "▲" if pct >= 0 else "▼"
        badge  = "🟢" if pct >= 0 else "🔴"
        parts.append(
            f"**{ticker}** {badge} &nbsp; "
            f"Close: **${latest:.2f}** &nbsp; {arrow} {pct:+.1f}% &nbsp;|&nbsp; "
            f"30d High: ${max(highs):.2f} &nbsp; Low: ${min(lows):.2f}"
        )
    return "\n\n".join(parts)


def _build_news_df() -> pd.DataFrame:
    rows = []
    for doc in company_docs + market_docs:
        m = doc["metadata"]
        rows.append({
            "Date":     _ts_to_date(m.get("published_at")),
            "Ticker":   m.get("ticker") or "Market",
            "Headline": m.get("headline", ""),
            "Source":   m.get("source", ""),
            "Impact":   m.get("impact_level", ""),
            "Tags":     ", ".join(m.get("topic_tags") or []),
        })
    rows.sort(key=lambda x: x["Date"], reverse=True)
    cols = ["Date", "Ticker", "Headline", "Source", "Impact", "Tags"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


price_df       = _build_price_df()
stock_summary  = _build_stock_summary_md()
news_df        = _build_news_df()

_EMPTY_SOURCES = pd.DataFrame(columns=["Headline", "Source", "Date", "Ticker", "Score"])

INDEX_STATS = (
    f"**{len(all_docs)} docs indexed** &nbsp;|&nbsp; "
    f"{len(company_docs)} company news &nbsp;|&nbsp; "
    f"{len(market_docs)} market news &nbsp;|&nbsp; "
    f"{len(stock_docs)} price summaries &nbsp;|&nbsp; "
    f"Tickers: {', '.join(TICKERS)}"
)


# ── Chat logic ────────────────────────────────────────────────────────────────

def respond(message: str, history: list, selected_ticker: str):
    ticker = None if selected_ticker == "Any" else selected_ticker
    if ticker is None:
        ticker = detect_ticker(message)
    if ticker is None:
        ticker = find_last_ticker(history)

    docs = retriever.retrieve(question=message, ticker=ticker, top_k=5)
    answer = generator.generate_answer(message, docs)

    ticker_note = (
        f"\n\n**Ticker context:** `{ticker}`"
        if ticker
        else "\n\n**Ticker context:** None — searching across all news."
    )

    updated_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer + ticker_note},
    ]

    src_rows = [
        [
            d.get("headline", ""),
            d.get("source", ""),
            _ts_to_date(d.get("published_at")),
            d.get("ticker") or "Market",
            f"{d.get('score', 0):.3f}",
        ]
        for d in docs
    ]
    sources_df = pd.DataFrame(src_rows, columns=["Headline", "Source", "Date", "Ticker", "Score"])

    return "", updated_history, sources_df


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="FinRAG") as demo:

    gr.Markdown("# FinRAG: Financial Intelligence Assistant")
    gr.Markdown(INDEX_STATS)

    with gr.Tabs():

        # ── Tab 1: Chat ───────────────────────────────────────────────────────
        with gr.TabItem("💬 Chat"):
            ticker_radio = gr.Radio(
                choices=["Any"] + TICKERS,
                value="Any",
                label="Ticker context",
                info="Pin a ticker or leave as 'Any' to auto-detect from your question.",
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=500,
                        show_label=False,
                        placeholder=(
                            "Ask a financial question — e.g. *Why is NVDA moving recently?*"
                        ),
                    )
                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="Type your question and press Enter or click Send…",
                            label="Question",
                            scale=5,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear chat", size="sm")

                with gr.Column(scale=2):
                    gr.Markdown("### Retrieved Sources")
                    gr.Markdown(
                        "_Documents used to generate the answer appear here after each query._",
                        visible=True,
                    )
                    sources_table = gr.Dataframe(
                        value=_EMPTY_SOURCES,
                        headers=["Headline", "Source", "Date", "Ticker", "Score"],
                        wrap=True,
                        max_height=470,
                    )

            send_btn.click(
                respond,
                inputs=[msg_box, chatbot, ticker_radio],
                outputs=[msg_box, chatbot, sources_table],
            )
            msg_box.submit(
                respond,
                inputs=[msg_box, chatbot, ticker_radio],
                outputs=[msg_box, chatbot, sources_table],
            )
            clear_btn.click(
                lambda: ([], _EMPTY_SOURCES),
                outputs=[chatbot, sources_table],
            )

        # ── Tab 2: Stocks ─────────────────────────────────────────────────────
        with gr.TabItem("📈 Stocks"):
            gr.Markdown("### Price Summary (30-day window)")
            gr.Markdown(stock_summary)
            gr.Markdown("---")
            gr.Markdown("### Closing Price Chart")

            if not price_df.empty:
                gr.LinePlot(
                    value=price_df,
                    x="Date",
                    y="Close",
                    color="Ticker",
                    title="Daily Closing Price",
                    tooltip=["Date", "Ticker", "Close"],
                    height=420,
                    x_label_angle=45,
                )
            else:
                gr.Markdown(
                    "> **Price chart unavailable.** "
                    "The Finnhub free tier does not include OHLCV candle data. "
                    "Stock summaries in the vector index were loaded from text descriptions."
                )

        # ── Tab 3: News Feed ──────────────────────────────────────────────────
        with gr.TabItem("📰 News Feed"):
            gr.Markdown(f"Showing **{len(news_df)} articles** fetched at startup.")
            with gr.Row():
                tf = gr.Dropdown(
                    choices=["All"] + TICKERS,
                    value="All",
                    label="Filter by Ticker",
                    scale=1,
                )
                imf = gr.Dropdown(
                    choices=["All", "high", "medium", "low"],
                    value="All",
                    label="Filter by Impact",
                    scale=1,
                )

            news_table = gr.Dataframe(
                value=news_df,
                wrap=True,
                max_height=560,
            )

            def _filter_news(ticker: str, impact: str) -> pd.DataFrame:
                df = news_df.copy()
                if ticker != "All":
                    df = df[df["Ticker"] == ticker]
                if impact != "All":
                    df = df[df["Impact"] == impact]
                return df

            tf.change(_filter_news, inputs=[tf, imf], outputs=news_table)
            imf.change(_filter_news, inputs=[tf, imf], outputs=news_table)


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        debug=True,
        theme=gr.themes.Soft(),
    )
