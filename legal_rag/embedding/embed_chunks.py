"""Local embedding generation for judgment chunks."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ChunkEmbedder:
    """Small wrapper for batch embeddings using a lightweight local model."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)


def embed_chunks(chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
    """Attach a local embedding vector to each chunk."""
    embedder = ChunkEmbedder()
    vectors = embedder.embed_texts([chunk["chunk_text"] for chunk in chunks], batch_size)

    enriched = []
    for chunk, vector in zip(chunks, vectors):
        item = dict(chunk)
        item["embedding"] = vector
        enriched.append(item)

    return enriched
