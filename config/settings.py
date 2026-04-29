"""
File: settings.py

Purpose:
Centralizes project configuration values.

Role in Pipeline:
Config Layer – Reads environment variables and provides shared settings
for ingestion, embeddings, vector storage, retrieval, and generation.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Environment
# -------------------------
ENV = os.getenv("ENV", "dev")

# -------------------------
# API Keys
# -------------------------
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# Vector Store
# -------------------------
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "finrag_news")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
POSTGRES_URL = os.getenv("POSTGRES_URL")

# -------------------------
# Embeddings
# -------------------------
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "bge")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# -------------------------
# Retrieval
# -------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
DAYS_BACK = int(os.getenv("DAYS_BACK", "7"))
DEFAULT_TICKERS = os.getenv(
    "DEFAULT_TICKERS",
    "NVDA,AAPL,TSLA,MSFT"
).split(",")

# -------------------------
# LLM (IMPORTANT SECTION)
# -------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e4b")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")