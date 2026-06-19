"""Embed chunk JSONL data using Sentence Transformers and local BGE-small."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
INPUT_PATH = Path("data/processed/chunks_master.jsonl")
OUTPUT_PATH = Path("data/processed/chunks_with_vectors.jsonl")


class ChunkEmbedder:
    """Sentence Transformers wrapper for chunk and query embeddings."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
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
        query_text = query.strip()
        if not query_text:
            raise ValueError("Query cannot be empty.")

        vector = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)


def iter_chunks(input_path: Path = INPUT_PATH) -> Iterable[Dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {input_path}")

    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def embed_chunks(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
) -> int:
    chunks = list(iter_chunks(input_path))
    embedder = ChunkEmbedder(model_name=model_name)
    vectors = embedder.embed_texts([chunk["chunk_text"] for chunk in chunks], batch_size=batch_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk, vector in zip(chunks, vectors):
            record = dict(chunk)
            record["embedding_model"] = model_name
            record["embedding"] = vector.tolist()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(chunks)


def main() -> None:
    count = embed_chunks()
    print(f"Embedded {count} chunks using {MODEL_NAME}")
    print(f"Output written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
