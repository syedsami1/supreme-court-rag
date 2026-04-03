"""Text chunking utilities."""

from __future__ import annotations

from typing import Dict, List


def split_documents(
    documents: List[Dict], chunk_size: int = 500, overlap: int = 100
) -> List[Dict]:
    """
    Split each document into overlapping word chunks.

    Args:
        documents: Output from loader.load_pdfs(...)
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Dict] = []
    step = chunk_size - overlap

    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        words = text.split()
        if not words:
            continue

        start = 0
        chunk_id = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                chunk_meta = dict(metadata)
                chunk_meta["chunk_id"] = chunk_id
                chunks.append({"text": chunk_text, "metadata": chunk_meta})

            start += step
            chunk_id += 1

    if not chunks:
        raise ValueError("No chunks created from documents.")

    return chunks
