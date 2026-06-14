"""
File: settings.py

Purpose:
Centralizes all project configuration. Loads .env and exposes settings
to the rest of the app. Import from here only — do not use os.getenv
elsewhere.

Role in Pipeline:
Config Layer – Application defaults with optional .env overrides;
secrets and deployment URLs from .env.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Environment
# -------------------------
ENV = os.getenv("ENV", "dev")

# -------------------------
# API keys (secrets — .env only, no defaults)
# -------------------------
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# Vector store
# -------------------------
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "pgvector")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "finrag_news")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
POSTGRES_URL = os.getenv("POSTGRES_URL")
AWS_POSTGRES_URL = os.getenv("AWS_POSTGRES_URL")

# -------------------------
# App behavior
# -------------------------
INGEST_ON_STARTUP = os.getenv("INGEST_ON_STARTUP", "false").lower() in ("1", "true", "yes")

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
DEFAULT_TICKERS = [
    t.strip()
    for t in os.getenv("DEFAULT_TICKERS", "NVDA,AAPL,TSLA,MSFT").split(",")
    if t.strip()
]

# -------------------------
# LLM
# -------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
