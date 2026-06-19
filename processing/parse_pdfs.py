"""Download, extract, parse, and reclassify 2024 Supreme Court PDFs."""

from __future__ import annotations

import json
import tarfile
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import fitz
import pandas as pd
import requests

from processing.classify_category import classify_case, normalize_text


TAR_URL = "https://indian-supreme-court-judgments.s3.amazonaws.com/data/tar/year=2024/english/english.tar"
METADATA_PATH = Path("data/processed/2024_metadata_cleaned.parquet")
TAR_PATH = Path("data/raw/english_2024.tar")
PDF_DIR = Path("data/raw/pdfs")
PARSED_DIR = Path("data/processed/parsed")
ERROR_LOG = Path("data/processed/parse_errors.txt")


def download_tar() -> None:
    if TAR_PATH.exists() and TAR_PATH.stat().st_size > 0:
        print(f"Tar already exists: {TAR_PATH}")
        return

    TAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading tar: {TAR_URL}")
    with requests.get(TAR_URL, stream=True, timeout=300) as response:
        response.raise_for_status()
        downloaded = 0
        with TAR_PATH.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded and downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                    print(f"Downloaded: {downloaded // (1024 * 1024)} MB")

    print(f"Download complete: {TAR_PATH.stat().st_size / (1024 * 1024):.1f} MB")


def load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned metadata not found: {METADATA_PATH}")
    return pd.read_parquet(METADATA_PATH)


def needed_filenames(df: pd.DataFrame) -> set[str]:
    return {f"{str(path).strip()}_EN.pdf" for path in df["path"].dropna()}


def extract_needed_pdfs(target_names: set[str]) -> int:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with tarfile.open(TAR_PATH, "r") as tar:
        for member in tar:
            filename = Path(member.name).name
            if filename not in target_names:
                continue

            output_path = PDF_DIR / filename
            if output_path.exists() and output_path.stat().st_size > 0:
                extracted += 1
            else:
                source = tar.extractfile(member)
                if source is None:
                    continue
                output_path.write_bytes(source.read())
                extracted += 1

            if extracted % 50 == 0:
                print(f"Extracted {extracted} PDFs")

    print(f"PDF extraction complete: {extracted}")
    return extracted


def extract_pdf_text(pdf_path: Path) -> Tuple[str, int]:
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        for page in doc:
            pages.append(page.get_text("text") or "")
    return "\n\n".join(pages).strip(), total_pages


def row_to_record(row: pd.Series, full_text: str, total_pages: int) -> Dict:
    case_id = str(row["path"]).strip()
    description = normalize_text(str(row.get("description", "")))
    title = normalize_text(str(row.get("title", "")))
    category = classify_case(title, description, full_text)
    decision_date = pd.to_datetime(row["decision_date"], errors="coerce")

    return {
        "case_id": case_id,
        "title": title,
        "citation": normalize_text(str(row.get("citation", ""))),
        "court": "Supreme Court",
        "year": int(row.get("year", 2024)),
        "judge": normalize_text(str(row.get("judge", ""))),
        "decision_date": decision_date.date().isoformat() if not pd.isna(decision_date) else "",
        "category": category,
        "full_text": full_text,
        "total_pages": total_pages,
        "char_length": len(full_text),
    }


def parse_pdfs(df: pd.DataFrame) -> Tuple[List[Dict], List[str]]:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    metadata_by_path = {str(row["path"]).strip(): row for _, row in df.iterrows()}

    parsed_records: List[Dict] = []
    errors: List[str] = []

    for idx, (case_id, row) in enumerate(metadata_by_path.items(), start=1):
        pdf_path = PDF_DIR / f"{case_id}_EN.pdf"
        try:
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF missing: {pdf_path}")
            full_text, total_pages = extract_pdf_text(pdf_path)
            record = row_to_record(row, full_text, total_pages)
            output_path = PARSED_DIR / f"{case_id}.json"
            output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            parsed_records.append(record)
        except Exception as exc:
            errors.append(f"{case_id}: {exc}")

        if idx % 50 == 0:
            print(f"Parsed {idx} / {len(metadata_by_path)} metadata rows")

    ERROR_LOG.write_text("\n".join(errors), encoding="utf-8")
    return parsed_records, errors


def category_distribution(records: Iterable[Dict]) -> Counter:
    return Counter(record["category"] for record in records)


def main() -> None:
    df = load_metadata()
    targets = needed_filenames(df)

    download_tar()
    extracted_count = extract_needed_pdfs(targets)
    records, errors = parse_pdfs(df)
    char_lengths = [record["char_length"] for record in records]
    avg_chars = mean(char_lengths) if char_lengths else 0
    categories = category_distribution(records)

    print("\nSummary")
    print("=======")
    print(f"PDFs extracted successfully: {extracted_count}")
    print(f"Parsed successfully: {len(records)}")
    print(f"Failed parses: {len(errors)}")
    print(f"Avg char length per case: {avg_chars:.0f}")
    print("Category distribution:")
    for category, count in sorted(categories.items()):
        print(f"- {category}: {count}")
    print(f"Parse error log: {ERROR_LOG}")


if __name__ == "__main__":
    main()
