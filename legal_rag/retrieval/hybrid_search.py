"""Hybrid keyword + vector search with filtering and case-level deduplication."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

from sentence_transformers import CrossEncoder

from embedding.embed_chunks import ChunkEmbedder
from indexing.faiss_index import FaissIndex
from retrieval.bm25_search import BM25Search


BOOST_KEYWORDS = ("bail", "anticipatory", "section 438")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def _normalize(results: List[Tuple[Dict, float]]) -> Dict[str, float]:
    if not results:
        return {}

    scores = [score for _, score in results]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return {item["chunk_id"]: 1.0 for item, _ in results}

    return {
        item["chunk_id"]: (score - min_score) / (max_score - min_score)
        for item, score in results
    }


def _passes_filters(chunk: Dict, filters: Optional[Dict]) -> bool:
    if not filters:
        return True

    for key, expected in filters.items():
        if expected is None:
            continue
        actual = chunk.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class HybridSearch:
    def __init__(
        self,
        bm25: BM25Search,
        faiss_index: FaissIndex,
        embedder: ChunkEmbedder,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        min_score: float = 0.15,
        reranker_model: str = RERANKER_MODEL,
    ):
        self.bm25 = bm25
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.min_score = min_score
        self.reranker = CrossEncoder(reranker_model)

    @staticmethod
    def _apply_keyword_boost(score: float, chunk_text: str) -> float:
        lowered = chunk_text.lower()
        if any(keyword in lowered for keyword in BOOST_KEYWORDS):
            return score * 1.2
        return score

    def _rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []

        pairs = [(query, chunk["chunk_text"]) for chunk in chunks]
        rerank_scores = self.reranker.predict(pairs)

        for chunk, rerank_score in zip(chunks, rerank_scores):
            chunk["rerank_score"] = float(rerank_score)
            chunk["final_score"] = _sigmoid(float(rerank_score))

        return sorted(
            chunks,
            key=lambda item: item["final_score"],
            reverse=True
        )

    @staticmethod
    def _best_chunk_per_case(chunks: List[Dict]) -> List[Dict]:
        """Choose one authoritative chunk/category per case after reranking."""
        best_by_case: Dict[str, Dict] = {}

        for chunk in chunks:
            case_id = chunk["case_id"]
            current = best_by_case.get(case_id)
            if current is None:
                best_by_case[case_id] = chunk
                continue

            chunk_key = (chunk.get("final_score", 0.0), chunk.get("score", 0.0))
            current_key = (current.get("final_score", 0.0), current.get("score", 0.0))
            if chunk_key > current_key:
                best_by_case[case_id] = chunk

        return sorted(
            best_by_case.values(),
            key=lambda item: (item.get("final_score", 0.0), item.get("score", 0.0)),
            reverse=True,
        )

    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 10,
        candidate_k: int = 40,
        rerank_k: int = 20,
    ) -> List[Dict]:

        bm25_results = self.bm25.search(query, top_k=candidate_k)
        query_vector = self.embedder.embed_query(query)
        vector_results = self.faiss_index.search(query_vector, top_k=candidate_k)

        bm25_scores = _normalize(bm25_results)
        vector_scores = _normalize(vector_results)

        merged: Dict[str, Dict] = {}

        for chunk, _ in bm25_results + vector_results:
            if not _passes_filters(chunk, filters):
                continue

            chunk_id = chunk["chunk_id"]

            score = (
                self.bm25_weight * bm25_scores.get(chunk_id, 0.0)
                + self.vector_weight * vector_scores.get(chunk_id, 0.0)
            )

            score = self._apply_keyword_boost(score, chunk["chunk_text"])
            merged[chunk_id] = {"score": score, **chunk}

        ranked_chunks = [
            chunk
            for chunk in sorted(
                merged.values(),
                key=lambda item: item["score"],
                reverse=True
            )
            if chunk["score"] >= self.min_score
        ]

        reranked_chunks = self._rerank(query, ranked_chunks[:rerank_k])
        case_results = self._best_chunk_per_case(reranked_chunks)
        return case_results[:top_k]
