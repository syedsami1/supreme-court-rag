"""Chunk parsed Supreme Court cases into paragraph-first JSONL records."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


PARSED_DIR = Path("data/processed/parsed")
OUTPUT_PATH = Path("data/processed/chunks_master.jsonl")
MIN_CHARS = 100
MAX_CHARS = 1200
FALLBACK_SIZE = 512
FALLBACK_OVERLAP = 50

NUMBERED_PARA_RE = re.compile(
    r"(?ms)(?:^|\n)\s*(\d{1,4})[\.\)]\s+(.*?)(?=(?:\n\s*\d{1,4}[\.\)]\s+)|\Z)"
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def recursive_chunk_text(text: str, size: int = FALLBACK_SIZE, overlap: int = FALLBACK_OVERLAP) -> List[str]:
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


def chunk_case(record: Dict) -> List[Dict]:
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
        parts = recursive_chunk_text(paragraph) if len(paragraph) > MAX_CHARS else [paragraph]
        for sub_index, chunk_text in enumerate(parts):
            if len(chunk_text) < MIN_CHARS:
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


def iter_parsed_cases() -> Iterable[Dict]:
    for path in sorted(PARSED_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            yield json.load(f)


def main() -> None:
    if not PARSED_DIR.exists():
        raise FileNotFoundError(f"Parsed directory not found: {PARSED_DIR}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_cases = 0
    all_chunks: List[Dict] = []
    chunks_per_case: List[int] = []

    for case in iter_parsed_cases():
        total_cases += 1
        case_chunks = chunk_case(case)
        all_chunks.extend(case_chunks)
        chunks_per_case.append(len(case_chunks))

        if total_cases % 50 == 0:
            print(f"Chunked {total_cases} cases")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    char_lengths = [chunk["char_length"] for chunk in all_chunks]
    category_counts = Counter(chunk["category"] for chunk in all_chunks)

    print("\nSummary")
    print("=======")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Avg chunks per case: {mean(chunks_per_case):.2f}")
    print(f"Avg chunk char length: {mean(char_lengths):.2f}")
    print(f"Min chunk char length: {min(char_lengths)}")
    print(f"Max chunk char length: {max(char_lengths)}")
    print("Chunk count by category:")
    for category, count in sorted(category_counts.items()):
        print(f"- {category}: {count}")
    print(f"Output written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
