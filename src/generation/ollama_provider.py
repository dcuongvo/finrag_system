"""
File: ollama_provider.py

Purpose:
Generates responses using a local Ollama model.

Role in Pipeline:
Generation Layer – Supports local LLM inference such as Gemma.
"""

import requests

from config.settings import LLM_MODEL, OLLAMA_BASE_URL
from src.generation.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = LLM_MODEL

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()
        return response.json().get("response", "")