"""
File: gemini_provider.py

Purpose:
Generates responses using Google's Gemini API.

Role in Pipeline:
Generation Layer – Supports cloud-based LLM generation for faster
or higher-quality responses.
"""

import google.generativeai as genai

from config.settings import GOOGLE_API_KEY, LLM_MODEL
from src.generation.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        api_key = GOOGLE_API_KEY

        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in .env")

        genai.configure(api_key=api_key)

        self.model_name = LLM_MODEL
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text