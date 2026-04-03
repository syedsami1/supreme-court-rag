"""Embedding utilities using sentence-transformers."""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Wrapper around sentence-transformers for consistent embedding output."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'."
            ) from exc

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("No texts provided for embedding.")
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        vectors = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)
