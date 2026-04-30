"""Indian Kanoon API ingestion for Supreme Court judgments."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import fitz
import requests


BASE_URL = "https://api.indiankanoon.org"
RAW_DIR = Path("data/raw")
METADATA_PATH = Path("data/metadata.json")


def _headers() -> Dict[str, str]:
    token = os.getenv("INDIAN_KANOON_API_TOKEN") or os.getenv("INDIAN_KANOON_API_KEY")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    return headers


def ensure_api_token() -> None:
    token = os.getenv("INDIAN_KANOON_API_TOKEN") or os.getenv("INDIAN_KANOON_API_KEY")
    if not token:
        raise EnvironmentError(
            "Indian Kanoon API token is required. Set INDIAN_KANOON_API_TOKEN "
            "or INDIAN_KANOON_API_KEY before running ingestion."
        )


def _request_json(url: str, params: Dict, retries: int = 3, sleep_seconds: int = 2) -> Dict:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=_headers(), timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            print(f"[warn] API request failed attempt {attempt}/{retries}: {exc}")
            if attempt < retries:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"Indian Kanoon API request failed after retries: {last_error}")


def _request_bytes(url: str, retries: int = 3, sleep_seconds: int = 2) -> Optional[bytes]:
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=_headers(), timeout=45)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            print(f"[warn] Download failed attempt {attempt}/{retries}: {url} ({exc})")
            if attempt < retries:
                time.sleep(sleep_seconds)
    return None


def _clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title or "").strip()
    title = re.sub(r"\b\d{4}\s+(?:Latest\s+)?Caselaw\s+\d+\s+SC\b", "", title, flags=re.I)
    title = re.sub(r"\b\d{4}\s+INSC\s+\d+\b", "", title, flags=re.I)
    title = re.sub(r"\s*[-|:]\s*$", "", title).strip()
    return title or "Untitled Supreme Court Case"


def _extract_search_docs(payload: Dict) -> List[Dict]:
    docs = payload.get("docs") or payload.get("results") or []
    return docs if isinstance(docs, list) else []


def search_supreme_court_cases(year: int = 2020, limit: int = 25) -> List[Dict]:
    """Search Indian Kanoon for Supreme Court judgments in one year."""
    ensure_api_token()
    collected: List[Dict] = []
    page = 0
    query = f"doctypes:supremecourt fromdate:01-01-{year} todate:31-12-{year}"

    while len(collected) < limit:
        payload = _request_json(
            f"{BASE_URL}/search/",
            params={"formInput": query, "pagenum": page},
        )
        docs = _extract_search_docs(payload)
        if not docs:
            break

        for doc in docs:
            doc_id = str(doc.get("tid") or doc.get("docid") or doc.get("id") or "").strip()
            if not doc_id:
                continue
            title = _clean_title(doc.get("title") or doc.get("headline") or doc_id)
            collected.append(
                {
                    "case_id": doc_id,
                    "title": title,
                    "year": year,
                    "court": "Supreme Court",
                }
            )
            if len(collected) >= limit:
                break

        page += 1
        time.sleep(1)

    return collected


def _write_text_pdf(pdf_path: Path, title: str, text: str) -> None:
    """Create a local PDF from API text when no court-copy PDF is returned."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(
        fitz.Rect(50, 50, 545, 792),
        f"{title}\n\n{text}",
        fontsize=10,
        fontname="helv",
    )
    doc.save(pdf_path)
    doc.close()


def _download_case_pdf(case: Dict) -> bool:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    case_id = case["case_id"]
    pdf_path = RAW_DIR / f"{case_id}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return True

    content = _request_bytes(f"{BASE_URL}/origdoc/{case_id}/")
    if content and content.startswith(b"%PDF"):
        pdf_path.write_bytes(content)
        return True

    # Indian Kanoon often exposes normalized document text instead of a raw PDF.
    # We keep the downstream contract PDF-based by writing that text into a PDF.
    try:
        payload = _request_json(f"{BASE_URL}/doc/{case_id}/", params={})
        text = payload.get("doc") or payload.get("text") or payload.get("judgment") or ""
        if not text.strip():
            print(f"[warn] No document text found for {case_id}; skipping")
            return False
        _write_text_pdf(pdf_path, case["title"], re.sub(r"<[^>]+>", " ", text))
        return True
    except Exception as exc:
        print(f"[warn] Could not create PDF for {case_id}: {exc}")
        return False


def ingest_cases(year: int = 2020, limit: int = 25) -> List[Dict]:
    """Fetch metadata and PDFs for a small Supreme Court test corpus."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cases = search_supreme_court_cases(year=year, limit=limit)
    saved: List[Dict] = []

    for case in cases:
        print(f"[ingest] {case['case_id']} - {case['title']}")
        if _download_case_pdf(case):
            saved.append(case)
        else:
            print(f"[warn] Skipped failed download: {case['case_id']}")
        time.sleep(1)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print(f"[ingest] Saved {len(saved)} PDFs and metadata records")
    return saved
