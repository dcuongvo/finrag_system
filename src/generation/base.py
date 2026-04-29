"""
File: base.py

Purpose:
Defines the common interface for LLM providers.

Role in Pipeline:
Generation Layer – Allows the system to swap between Gemini, Ollama,
OpenAI, or other LLM backends without changing answer generation logic.
"""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass