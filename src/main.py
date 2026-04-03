"""CLI entrypoint for Supreme Court RAG pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from collections import defaultdict
from typing import List

from embedder import SentenceTransformerEmbedder
from generator import GroqGenerator
from loader import load_pdfs
from retriever import Retriever
from splitter import split_documents
from vectorstore import FAISSVectorStore


def _is_case_specific_query(query: str) -> bool:
    q = query.lower()
    case_cues = (
        "this case",
        "this judgment",
        "this judgement",
        "in this case",
        "in this judgment",
        "finally hold",
        "final holding",
        "what did the supreme court finally hold",
    )
    return any(cue in q for cue in case_cues)


def _focus_on_dominant_case(chunks: List[dict]) -> List[dict]:
    """If retrieval spans multiple files, keep chunks from the strongest file."""
    if not chunks:
        return chunks

    by_file_score = defaultdict(float)
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "unknown")
        score = float(chunk.get("score", 0.0))
        by_file_score[filename] += max(score, 0.0)

    dominant_file = max(by_file_score, key=by_file_score.get)
    focused = [
        chunk
        for chunk in chunks
        if chunk.get("metadata", {}).get("filename", "unknown") == dominant_file
    ]
    return focused or chunks


def build_index(
    data_dir: str,
    index_path: str,
    metadata_path: str,
    embedder: SentenceTransformerEmbedder,
) -> None:
    """Load PDFs, split into chunks, embed, and persist FAISS index + metadata."""
    print("[build] Loading PDFs...")
    documents = load_pdfs(data_dir)
    print(f"[build] Loaded {len(documents)} pages.")

    print("[build] Splitting text into chunks...")
    chunks = split_documents(documents, chunk_size=500, overlap=100)
    print(f"[build] Created {len(chunks)} chunks.")

    print("[build] Computing embeddings...")
    embeddings = embedder.embed_texts([chunk["text"] for chunk in chunks])

    print("[build] Building FAISS index...")
    store = FAISSVectorStore()
    store.build(embeddings, chunks)
    store.save(index_path=index_path, metadata_path=metadata_path)
    print(f"[build] Saved index to: {Path(index_path).resolve()}")
    print(f"[build] Saved metadata to: {Path(metadata_path).resolve()}")


def print_sources(chunks: List[dict]) -> None:
    seen = set()
    print("\nSources:")
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = (meta.get("filename", "unknown"), meta.get("page", "?"))
        if source in seen:
            continue
        seen.add(source)
        print(f"- {source[0]} (page {source[1]})")


def run_cli(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(index_dir / "faiss.index")
    metadata_path = str(index_dir / "chunks.json")

    print("[init] Loading embedder model...")
    embedder = SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    needs_build = args.rebuild or not Path(index_path).exists() or not Path(metadata_path).exists()
    if needs_build:
        build_index(
            data_dir=args.data_dir,
            index_path=index_path,
            metadata_path=metadata_path,
            embedder=embedder,
        )

    store = FAISSVectorStore()
    store.load(index_path=index_path, metadata_path=metadata_path)
    retriever = Retriever(embedder=embedder, vectorstore=store)

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Please set it before running the CLI."
        )
    generator = GroqGenerator(api_key=groq_api_key, model=args.model)

    print("\nRAG CLI ready. Type your question.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not query:
            print("Please enter a non-empty question.")
            continue

        try:
            retrieved_chunks = retriever.retrieve(query=query, top_k=args.top_k)
            if _is_case_specific_query(query):
                retrieved_chunks = _focus_on_dominant_case(retrieved_chunks)
            answer = generator.generate(query=query, retrieved_chunks=retrieved_chunks)
            print("\nAnswer:")
            print(answer)
            print_sources(retrieved_chunks)
            print("")
        except Exception as exc:
            print(f"[error] {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supreme Court PDF RAG CLI")
    parser.add_argument("--data-dir", default="data", help="Path to PDF directory")
    parser.add_argument(
        "--index-dir",
        default="vector_store",
        help="Directory to save/load FAISS index and metadata",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index from PDFs even if existing index files are found",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        help="Groq model name (or set GROQ_MODEL env var)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_cli(parse_args())
