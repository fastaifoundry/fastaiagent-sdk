"""Embedding providers for knowledge base."""

from __future__ import annotations

import math
from typing import Protocol


class Embedder(Protocol):
    """Protocol for embedding providers."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SimpleEmbedder:
    """Simple character-frequency embedder (no external deps, for testing/fallback).

    NOT suitable for production — use FastEmbedEmbedder or OpenAIEmbedder instead.
    """

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for i, ch in enumerate(text.lower()):
            idx = ord(ch) % self.dimensions
            vec[idx] += 1.0
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FastEmbedEmbedder:
    """Embedder using FastEmbed (local, no API calls)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError("FastEmbed is required. Install with: pip install fastaiagent[kb]")
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = list(self._model.embed(texts))
        return [list(e) for e in embeddings]


class OpenAIEmbedder:
    """Embedder using OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            import openai
        except ImportError:
            raise ImportError("openai is required. Install with: pip install fastaiagent[openai]")
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]


def get_default_embedder() -> Embedder:
    """Get the best available embedder."""
    try:
        return FastEmbedEmbedder()
    except ImportError:
        pass
    try:
        return OpenAIEmbedder()
    except (ImportError, Exception):
        pass
    return SimpleEmbedder()
