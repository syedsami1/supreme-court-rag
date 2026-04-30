"""Paragraph-first chunking with recursive fallback for long paragraphs."""

from __future__ import annotations

import re
from typing import Dict, List


NUMBERED_PARA_RE = re.compile(
    r"(?ms)(?:^|\n)\s*(\d{1,4})[\.\)]\s+(.*?)(?=(?:\n\s*\d{1,4}[\.\)]\s+)|\Z)"
)


def recursive_chunk_text(text: str, size: int = 512, overlap: int = 50) -> List[str]:
    """Split oversized text by character windows with overlap."""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    step = size - overlap

    while start < len(text):
        end = min(start + size, len(text))
        candidate = text[start:end]

        # Prefer ending near a sentence boundary when possible.
        if end < len(text):
            boundary = max(candidate.rfind(". "), candidate.rfind("; "), candidate.rfind(", "))
            if boundary > size // 2:
                end = start + boundary + 1
                candidate = text[start:end]

        cleaned = " ".join(candidate.split())
        if cleaned:
            chunks.append(cleaned)

        if end >= len(text):
            break
        start = max(end - overlap, start + step)

    return chunks


def _extract_paragraphs(text: str) -> List[tuple[int, str]]:
    matches = NUMBERED_PARA_RE.findall(text)
    if matches:
        return [(int(index), " ".join(body.split())) for index, body in matches if body.strip()]

    fallback = [para.strip() for para in re.split(r"\n\s*\n+", text) if para.strip()]
    return [(i + 1, " ".join(para.split())) for i, para in enumerate(fallback)]


def chunk_judgment(text: str, metadata: Dict, max_chars: int = 1200) -> List[Dict]:
    """Create retrieval-ready chunks with legal metadata attached."""
    chunks: List[Dict] = []
    paragraphs = _extract_paragraphs(text)

    for para_index, paragraph in paragraphs:
        chunk_parts = (
            recursive_chunk_text(paragraph, size=512, overlap=50)
            if len(paragraph) > max_chars
            else [paragraph]
        )

        for part_index, chunk_text in enumerate(chunk_parts):
            chunk_id = f"{metadata['case_id']}_p{para_index}_{part_index}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "case_id": metadata["case_id"],
                    "title": metadata["title"],
                    "year": metadata["year"],
                    "court": metadata["court"],
                    "judge": metadata["judge"],
                    "category": metadata["category"],
                    "chunk_text": chunk_text,
                    "para_index": para_index,
                    "char_length": len(chunk_text),
                }
            )

    return chunks
