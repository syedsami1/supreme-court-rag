"""FAISS vector store utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np


class FAISSVectorStore:
    """Stores chunk embeddings and chunk payloads in a local FAISS index."""

    def __init__(self) -> None:
        self.index = None
        self.chunks: List[Dict] = []

    def build(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Embeddings count must match chunks count.")
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings provided.")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity if vectors are normalized
        index.add(embeddings)

        self.index = index
        self.chunks = chunks

    def save(self, index_path: str, metadata_path: str) -> None:
        if self.index is None:
            raise RuntimeError("No FAISS index to save. Build or load first.")

        index_file = Path(index_path)
        meta_file = Path(metadata_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_file))
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    def load(self, index_path: str, metadata_path: str) -> None:
        index_file = Path(index_path)
        meta_file = Path(metadata_path)

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file.resolve()}")
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file.resolve()}")

        self.index = faiss.read_index(str(index_file))
        with meta_file.open("r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("Vector index not initialized.")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        distances, indices = self.index.search(query_embedding, top_k)
        results: List[Dict] = []

        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "score": float(score),
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                }
            )

        return results
