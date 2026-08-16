"""Singleton embedding service wrapping sentence-transformers."""

from __future__ import annotations

import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Lazy-loaded singleton for generating embeddings."""

    _instance: EmbeddingService | None = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> EmbeddingService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self._model = SentenceTransformer(model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return the singleton EmbeddingService instance."""
    return EmbeddingService()
