"""
File: app.py

Purpose:
Gradio Blocks UI for FinRAG – Chat, Stocks dashboard, News Feed, and Index tabs.

Role in Pipeline:
Application Layer – Initializes RAG components; ingestion is triggered
from the Index tab (or on startup if INGEST_ON_STARTUP=true).
"""

import pandas as pd
import gradio as gr
from datetime import datetime

from config.settings import (
    AWS_POSTGRES_URL,
    DAYS_BACK,
    DEFAULT_TICKERS,
    INGEST_ON_STARTUP,
    POSTGRES_URL,
    TOP_K,
    VECTOR_BACKEND,
)
from src.embeddings.factory import get_embedder
from src.generation.answer_generator import AnswerGenerator
from src.generation.factory import get_llm_provider
from src.ingestion.pipeline import run_ingestion, sync_to_aws
from src.retrieval.retriever import Retriever
from src.vector_store.factory import get_vector_store


# ── Initialization ────────────────────────────────────────────────────────────
print("Initializing FinRAG system...")

TICKERS = DEFAULT_TICKERS

embedder = get_embedder()
vector_store = get_vector_store()
retriever = Retriever(embedder, vector_store)
llm = get_llm_provider()
generator = AnswerGenerator(llm)

print(f"Vector backend: {VECTOR_BACKEND}")

company_docs: list = []
market_docs: list = []
stock_docs: list = []
raw_candles: dict = {}


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
        return "_No stock price data yet — build the index on the **Index** tab._"
    parts = []
    for ticker, candles in raw_candles.items():
        closes = candles["c"]
        highs = candles["h"]
        lows = candles["l"]
        latest = closes[-1]
        first = closes[0]
        pct = ((latest - first) / first) * 100
        arrow = "▲" if pct >= 0 else "▼"
        badge = "🟢" if pct >= 0 else "🔴"
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
            "Date": _ts_to_date(m.get("published_at")),
            "Ticker": m.get("ticker") or "Market",
            "Headline": m.get("headline", ""),
            "Source": m.get("source", ""),
            "Impact": m.get("impact_level", ""),
            "Tags": ", ".join(m.get("topic_tags") or []),
        })
    rows.sort(key=lambda x: x["Date"], reverse=True)
    cols = ["Date", "Ticker", "Headline", "Source", "Impact", "Tags"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _format_index_stats(total: int = 0, company: int = 0, market: int = 0, stock: int = 0) -> str:
    if total == 0:
        return (
            f"**Index empty** &nbsp;|&nbsp; backend: `{VECTOR_BACKEND}` &nbsp;|&nbsp; "
            f"Tickers: {', '.join(TICKERS)} &nbsp;|&nbsp; "
            "_Use the **Index** tab to fetch news and build the database._"
        )
    return (
        f"**{total} docs indexed** &nbsp;|&nbsp; "
        f"{company} company news &nbsp;|&nbsp; "
        f"{market} market news &nbsp;|&nbsp; "
        f"{stock} price summaries &nbsp;|&nbsp; "
        f"backend: `{VECTOR_BACKEND}` &nbsp;|&nbsp; "
        f"Tickers: {', '.join(TICKERS)}"
    )


def _apply_ingestion_result(result):
    global company_docs, market_docs, stock_docs, raw_candles
    company_docs = result.company_docs
    market_docs = result.market_docs
    stock_docs = result.stock_docs
    raw_candles = result.raw_candles


def build_local_index():
    try:
        result = run_ingestion(
            embedder, vector_store, TICKERS, DAYS_BACK, VECTOR_BACKEND
        )
        _apply_ingestion_result(result)
        price_df = _build_price_df()
        news_df = _build_news_df()
        return (
            result.summary(),
            _format_index_stats(
                result.total_fetched,
                result.company_count,
                result.market_count,
                result.stock_count,
            ),
            news_df,
            _build_stock_summary_md(),
            price_df,
            f"Showing **{len(news_df)} articles** — last built from Finnhub.",
        )
    except Exception as exc:
        msg = f"**Index build failed**\n\n`{exc}`"
        return (
            msg,
            _format_index_stats(),
            _build_news_df(),
            _build_stock_summary_md(),
            _build_price_df(),
            "_News feed empty — build failed._",
        )


def upload_to_aws():
    try:
        return sync_to_aws(vector_store, AWS_POSTGRES_URL)
    except Exception as exc:
        return f"**AWS sync failed**\n\n`{exc}`"


_EMPTY_SOURCES = pd.DataFrame(columns=["Headline", "Source", "Date", "Ticker", "Score"])

if INGEST_ON_STARTUP:
    print("INGEST_ON_STARTUP=true — building index at startup...")
    startup_result = run_ingestion(
        embedder, vector_store, TICKERS, DAYS_BACK, VECTOR_BACKEND
    )
    _apply_ingestion_result(startup_result)
    print(startup_result.summary().replace("**", "").replace("\n", " "))

print("FinRAG system ready.")


# ── Chat logic ────────────────────────────────────────────────────────────────

def respond(message: str, history: list, selected_ticker: str):
    ticker = None if selected_ticker == "Any" else selected_ticker
    if ticker is None:
        ticker = detect_ticker(message)
    if ticker is None:
        ticker = find_last_ticker(history)

    docs = retriever.retrieve(question=message, ticker=ticker, top_k=TOP_K)
    answer = generator.generate_answer(message, docs)

    ticker_note = (
        f"\n\n**Ticker context:** `{ticker}`"
        if ticker
        else "\n\n**Ticker context:** None — searching across all news."
    )

    updated_history = history + [
        {"role": "user", "content": message},
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

_initial_stats = _format_index_stats(
    len(company_docs) + len(market_docs) + len(stock_docs),
    len(company_docs),
    len(market_docs),
    len(stock_docs),
)
_initial_news = _build_news_df()
_initial_stock_summary = _build_stock_summary_md()
_initial_price_df = _build_price_df()

def _postgres_host_hint(url: str | None) -> str:
    if not url:
        return "not configured"
    if "@" in url:
        return url.split("@")[-1].split("/")[0]
    return url


_db_hint = (
    f"**Database target:** `{_postgres_host_hint(POSTGRES_URL)}` "
    f"(via `POSTGRES_URL`)"
)
_sync_hint = (
    f"Optional sync copy → `{_postgres_host_hint(AWS_POSTGRES_URL)}`"
    if AWS_POSTGRES_URL and AWS_POSTGRES_URL != POSTGRES_URL
    else "Optional **Sync to AWS** — set `AWS_POSTGRES_URL` if you use a separate local DB."
)

with gr.Blocks(title="FinRAG") as demo:

    gr.Markdown("# FinRAG: Financial Intelligence Assistant")
    index_stats_md = gr.Markdown(_initial_stats)

    with gr.Tabs():

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

        with gr.TabItem("📈 Stocks"):
            gr.Markdown("### Price Summary (30-day window)")
            stock_summary_md = gr.Markdown(_initial_stock_summary)
            gr.Markdown("---")
            gr.Markdown("### Closing Price Chart")
            price_plot = gr.LinePlot(
                value=_initial_price_df,
                x="Date",
                y="Close",
                color="Ticker",
                title="Daily Closing Price",
                tooltip=["Date", "Ticker", "Close"],
                height=420,
                x_label_angle=45,
            )

        with gr.TabItem("📰 News Feed"):
            news_count_md = gr.Markdown(
                f"Showing **{len(_initial_news)} articles** — build the index to refresh."
            )
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
                value=_initial_news,
                wrap=True,
                max_height=560,
            )

            def _filter_news(ticker: str, impact: str) -> pd.DataFrame:
                df = _build_news_df()
                if ticker != "All":
                    df = df[df["Ticker"] == ticker]
                if impact != "All":
                    df = df[df["Impact"] == impact]
                return df

            tf.change(_filter_news, inputs=[tf, imf], outputs=news_table)
            imf.change(_filter_news, inputs=[tf, imf], outputs=news_table)

        with gr.TabItem("🗄️ Index"):
            gr.Markdown(
                "### Build the vector index\n\n"
                f"**Build index** fetches Finnhub news, embeds, and saves to "
                f"`{VECTOR_BACKEND}` at:\n\n"
                f"{_db_hint}\n\n"
                f"{_sync_hint}\n\n"
                "For **AWS RDS**, set `POSTGRES_URL` to your RDS connection string "
                "(with `?sslmode=require`). No Docker needed."
            )
            ingest_status = gr.Markdown(
                "_Ready. Click **Build index** to fetch news and save to the database._"
                if POSTGRES_URL
                else "_Set `POSTGRES_URL` in `.env` (your RDS URL) before building the index._"
            )
            with gr.Row():
                build_btn = gr.Button("Build index", variant="primary")
                aws_btn = gr.Button("Sync to AWS", variant="secondary")

            build_outputs = [
                ingest_status,
                index_stats_md,
                news_table,
                stock_summary_md,
                price_plot,
                news_count_md,
            ]
            build_btn.click(build_local_index, outputs=build_outputs)
            aws_btn.click(upload_to_aws, outputs=ingest_status)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        debug=True,
        theme=gr.themes.Soft(),
    )
