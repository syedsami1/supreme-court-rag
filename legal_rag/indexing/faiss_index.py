"""FAISS index build, search, save, and load helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np


class FaissIndex:
    def __init__(self) -> None:
        self.index = None
        self.metadata: List[Dict] = []

    def build_index(self, chunks: List[Dict]) -> None:
        if not chunks:
            raise ValueError("Cannot build FAISS index with no chunks.")

        vectors = np.vstack([chunk["embedding"] for chunk in chunks]).astype(np.float32)
        dim = vectors.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.metadata = [{k: v for k, v in chunk.items() if k != "embedding"} for chunk in chunks]

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[Dict, float]]:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")

        scores, indices = self.index.search(query_vector.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save_index(
        self,
        index_path: str = "storage/faiss.index",
        metadata_path: str = "storage/faiss_metadata.json",
    ) -> None:
        if self.index is None:
            raise RuntimeError("No FAISS index to save.")

        index_file = Path(index_path)
        metadata_file = Path(metadata_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_file))
        metadata_file.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def load_index(
        self,
        index_path: str = "storage/faiss.index",
        metadata_path: str = "storage/faiss_metadata.json",
    ) -> None:
        index_file = Path(index_path)
        metadata_file = Path(metadata_path)

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError("FAISS index or metadata file is missing.")

        self.index = faiss.read_index(str(index_file))
        self.metadata = json.loads(metadata_file.read_text(encoding="utf-8"))


def build_index(chunks: List[Dict]) -> FaissIndex:
    store = FaissIndex()
    store.build_index(chunks)
    return store
