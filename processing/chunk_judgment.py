"""Paragraph-first chunking with fixed-size fallback."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from processing.classify_category import clean_metadata_record, normalize_text


NUMBERED_PARA_RE = re.compile(
    r"(?ms)(?:^|\n)\s*(\d{1,4})[\.\)]\s+(.*?)(?=(?:\n\s*\d{1,4}[\.\)]\s+)|\Z)"
)


def recursive_chunk_text(text: str, size: int = 512, overlap: int = 50) -> List[str]:
    if len(text) <= size:
        cleaned = normalize_text(text)
        return [cleaned] if cleaned else []

    chunks: List[str] = []
    step = size - overlap
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        candidate = text[start:end]

        if end < len(text):
            boundary = max(candidate.rfind(". "), candidate.rfind("; "), candidate.rfind(", "))
            if boundary > size // 2:
                end = start + boundary + 1
                candidate = text[start:end]

        cleaned = normalize_text(candidate)
        if cleaned:
            chunks.append(cleaned)

        if end >= len(text):
            break
        start = max(end - overlap, start + step)

    return chunks


def extract_paragraphs(text: str) -> List[Tuple[int, str]]:
    matches = NUMBERED_PARA_RE.findall(text)
    if matches:
        return [(int(index), normalize_text(body)) for index, body in matches if normalize_text(body)]

    fallback = [normalize_text(para) for para in re.split(r"\n\s*\n+", text) if normalize_text(para)]
    return [(i + 1, para) for i, para in enumerate(fallback)]


def chunk_case(record: Dict, min_chars: int = 100, max_chars: int = 1200) -> List[Dict]:
    record = clean_metadata_record(record)
    metadata = {
        "case_id": record["case_id"],
        "title": record["title"],
        "citation": record["citation"],
        "court": record["court"],
        "year": record["year"],
        "judge": record["judge"],
        "category": record["category"],
        "decision_date": record["decision_date"],
    }

    chunks: List[Dict] = []
    for para_index, paragraph in extract_paragraphs(record["full_text"]):
        parts = recursive_chunk_text(paragraph) if len(paragraph) > max_chars else [paragraph]
        for sub_index, chunk_text in enumerate(parts):
            if len(chunk_text) < min_chars:
                continue
            chunks.append(
                {
                    "chunk_id": f"{metadata['case_id']}_p{para_index}_s{sub_index}",
                    **metadata,
                    "chunk_text": chunk_text,
                    "para_index": para_index,
                    "char_length": len(chunk_text),
                }
            )
    return chunks
