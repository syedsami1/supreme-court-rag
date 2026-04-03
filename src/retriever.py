"""Retriever module that combines embedder + FAISS search."""

from __future__ import annotations

from typing import Dict, List

from embedder import SentenceTransformerEmbedder
from vectorstore import FAISSVectorStore


class Retriever:
    def __init__(self, embedder: SentenceTransformerEmbedder, vectorstore: FAISSVectorStore):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_vector = self.embedder.embed_query(query)
        return self.vectorstore.search(query_vector, top_k=top_k)
