"""Group retrieval results by legal category."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Dict, List


def clean_title(title: str) -> str:
    cleaned = re.sub(r"\[[0-9]{4}\]\s*[^:]+:\s*", "", title or "")
    cleaned = re.sub(r"\b[0-9]{4}\s+INSC\s+[0-9]+\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b[0-9]{4}\s+Latest\s+Caselaw\s+[0-9]+\s+SC\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:|")
    return cleaned or "Untitled Supreme Court Case"


def make_snippet(text: str, max_chars: int = 280) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def group_results(query: str, results: List[Dict]) -> Dict:
    grouped = defaultdict(list)

    for result in results:
        grouped[result["category"]].append(
            {
                "case_id": result["case_id"],
                "title": clean_title(result["title"]),
                "category": result["category"],
                "snippet": make_snippet(result["chunk_text"]),
                "score": round(float(result.get("score", 0.0)), 4),
                "rerank_score": round(float(result.get("rerank_score", 0.0)), 4),
            }
        )

    return {"query": query, "grouped_results": dict(grouped)}
