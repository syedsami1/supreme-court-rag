"""BM25 keyword search over judgment chunks."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Search:
    def __init__(self, chunks: List[Dict]):
        if not chunks:
            raise ValueError("Cannot build BM25 index with no chunks.")

        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk["chunk_text"]) for chunk in chunks]
        self.index = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self.chunks[idx], float(score)) for idx, score in ranked if score > 0]
