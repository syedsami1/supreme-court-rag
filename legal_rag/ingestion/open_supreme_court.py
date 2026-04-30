"""Free/open ingestion from the public Indian Supreme Court judgments dataset."""

from __future__ import annotations

import json
import re
import tarfile
import time
from pathlib import Path
from typing import Dict, List

import requests


BASE_URL = "https://indian-supreme-court-judgments.s3.amazonaws.com"
RAW_DIR = Path("data/raw")
METADATA_PATH = Path("data/metadata.json")


def _get_json(url: str, retries: int = 3, sleep_seconds: int = 2) -> Dict:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            print(f"[warn] Open dataset request failed {attempt}/{retries}: {exc}")
            if attempt < retries:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"Open dataset request failed after retries: {last_error}")


def _clean_title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    title = re.sub(r"_EN$", "", stem)
    title = title.replace("_", " ")
    return f"Supreme Court Judgment {title}".strip()


def _case_metadata(filename: str, year: int) -> Dict:
    case_id = Path(filename).stem
    return {
        "case_id": case_id,
        "title": _clean_title_from_filename(filename),
        "year": year,
        "court": "Supreme Court",
        "filename": filename,
        "source": "vanga/indian-supreme-court-judgments public S3 dataset",
    }


def ingest_cases(year: int = 2020, limit: int = 25) -> List[Dict]:
    """
    Stream a public year-wise Supreme Court PDF archive and extract a small sample.

    The dataset is public and CC-BY-4.0. We read the archive as a stream and stop
    after `limit` PDFs, which keeps the MVP test run lightweight.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index_url = f"{BASE_URL}/data/tar/year={year}/english/english.index.json"
    index = _get_json(index_url)
    part_name = index["parts"][0]["name"]
    tar_url = f"{BASE_URL}/data/tar/year={year}/english/{part_name}"

    print(f"[ingest] Source: {tar_url}")
    print(f"[ingest] Archive size: {index.get('total_size_human', 'unknown')}")

    saved: List[Dict] = []
    with requests.get(tar_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with tarfile.open(fileobj=response.raw, mode="r|") as tar:
            for member in tar:
                if len(saved) >= limit:
                    break
                if not member.isfile() or not member.name.lower().endswith(".pdf"):
                    continue

                filename = Path(member.name).name
                target = RAW_DIR / filename
                extracted = tar.extractfile(member)
                if extracted is None:
                    print(f"[warn] Could not extract {filename}; skipping")
                    continue

                try:
                    target.write_bytes(extracted.read())
                    saved.append(_case_metadata(filename, year))
                    print(f"[ingest] Saved {filename}")
                except Exception as exc:
                    print(f"[warn] Failed saving {filename}: {exc}")

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print(f"[ingest] Saved {len(saved)} PDFs and metadata records")
    return saved
