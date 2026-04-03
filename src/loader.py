"""Document loading utilities for PDF files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF


def _normalize_text(text: str) -> str:
    """Normalize whitespace so downstream chunking is cleaner."""
    return " ".join(text.split())


def load_pdfs(data_dir: str) -> List[Dict]:
    """
    Load all PDFs from data_dir and extract text page-by-page.

    Returns:
        A list of dicts, each shaped like:
        {
            "text": "...",
            "metadata": {
                "filename": "case.pdf",
                "page": 1
            }
        }
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root.resolve()}")

    pdf_paths = sorted(root.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {root.resolve()}")

    documents: List[Dict] = []

    for pdf_path in pdf_paths:
        try:
            with fitz.open(pdf_path) as doc:
                for page_idx, page in enumerate(doc):
                    raw_text = page.get_text("text") or ""
                    text = _normalize_text(raw_text)
                    if not text:
                        continue

                    documents.append(
                        {
                            "text": text,
                            "metadata": {
                                "filename": pdf_path.name,
                                "page": page_idx + 1,
                            },
                        }
                    )
        except Exception as exc:
            # Continue loading other files even if one file fails.
            print(f"[loader] Failed to parse {pdf_path.name}: {exc}")

    if not documents:
        raise ValueError("No extractable text found in provided PDFs.")

    return documents
