"""Build the master JSONL chunk file from parsed case JSON files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from processing.chunk_judgment import chunk_case


PARSED_DIR = Path("data/processed/parsed")
OUTPUT_PATH = Path("data/processed/chunks_master.jsonl")


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
