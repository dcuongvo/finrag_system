"""
File: factory.py

Purpose:
Creates the configured LLM provider.

Role in Pipeline:
Generation Layer – Centralizes LLM provider selection so the backend can
be changed through configuration.
"""

from config.settings import LLM_PROVIDER


def get_llm_provider():
    provider = LLM_PROVIDER.lower()

    if provider == "ollama":
        from src.generation.ollama_provider import OllamaProvider
        return OllamaProvider()

    if provider == "gemini":
        from src.generation.gemini_provider import GeminiProvider
        return GeminiProvider()

    raise ValueError(f"Unsupported LLM provider: {provider}")