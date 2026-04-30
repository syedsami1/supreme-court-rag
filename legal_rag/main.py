"""Local MVP pipeline for Indian Supreme Court judgment search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import fitz

from embedding.embed_chunks import ChunkEmbedder
from ingestion.open_supreme_court import ingest_cases
from indexing.faiss_index import build_index
from processing.chunk_judgment import chunk_judgment
from processing.classify_category import extract_metadata
from retrieval.bm25_search import BM25Search
from retrieval.group_results import group_results
from retrieval.hybrid_search import HybridSearch


RAW_DATA_DIR = Path("data/raw")
STORAGE_DIR = Path("storage")
METADATA_PATH = Path("data/metadata.json")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    pages = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text("text")
                if text:
                    pages.append(text)
    except Exception as exc:
        print(f"[warn] Could not read {pdf_path.name}: {exc}")

    return "\n\n".join(pages)


def find_pdf_files() -> List[Path]:
    """Read PDFs strictly from data/raw."""
    return sorted(RAW_DATA_DIR.glob("*.pdf"))


def process_pdfs() -> List[Dict]:
    """Load PDFs, extract metadata, chunk text, and return all chunks."""
    pdf_files = find_pdf_files()
    if not pdf_files:
        raise FileNotFoundError("No PDFs found in data/raw. Run ingestion first.")

    metadata_by_case = load_ingested_metadata()
    all_chunks: List[Dict] = []
    for pdf_path in pdf_files:
        print(f"[load] {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"[warn] No extractable text in {pdf_path.name}")
            continue

        metadata = extract_metadata(text, pdf_path.name, year=2020)
        metadata.update(metadata_by_case.get(metadata["case_id"], {}))
        chunks = chunk_judgment(text, metadata)
        print(f"[chunk] {pdf_path.name}: {len(chunks)} chunks, category={metadata['category']}")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No chunks were created from the PDFs.")

    return all_chunks


def load_ingested_metadata() -> Dict[str, Dict]:
    if not METADATA_PATH.exists():
        return {}

    try:
        records = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] Could not read metadata file: {exc}")
        return {}

    cleaned = {}
    for record in records:
        case_id = record.get("case_id")
        if not case_id:
            continue
        # Keep extracted PDF titles when metadata only has filename-derived labels.
        cleaned[case_id] = {k: v for k, v in record.items() if k != "title"}
    return cleaned


def print_grouped_results(output: Dict) -> None:
    print("\n" + json.dumps(output, indent=2, ensure_ascii=False))


def build_pipeline() -> HybridSearch:
    chunks = process_pdfs()

    print(f"[embed] Embedding {len(chunks)} chunks locally...")
    embedder = ChunkEmbedder()
    vectors = embedder.embed_texts([chunk["chunk_text"] for chunk in chunks], batch_size=32)
    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors):
        item = dict(chunk)
        item["embedding"] = vector
        embedded_chunks.append(item)

    print("[index] Building FAISS index...")
    faiss_index = build_index(embedded_chunks)
    faiss_index.save_index(
        index_path=str(STORAGE_DIR / "faiss.index"),
        metadata_path=str(STORAGE_DIR / "faiss_metadata.json"),
    )

    print("[index] Building BM25 index...")
    bm25 = BM25Search([{k: v for k, v in chunk.items() if k != "embedding"} for chunk in embedded_chunks])

    return HybridSearch(bm25=bm25, faiss_index=faiss_index, embedder=embedder)


def main() -> None:
    print("Local Legal RAG Search")
    print("======================")

    if not find_pdf_files():
        print("[ingest] data/raw is empty; fetching 2020 Supreme Court cases from the open dataset")
        ingest_cases(year=2020, limit=25)

    searcher = build_pipeline()

    print("\nEnter a legal search query. Type 'exit' to quit.")
    while True:
        query = input("\nQuery> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            print("Please enter a non-empty query.")
            continue

        results = searcher.search(query=query, filters=None, top_k=10)
        output = group_results(query, results)
        print_grouped_results(output)


if __name__ == "__main__":
    main()
