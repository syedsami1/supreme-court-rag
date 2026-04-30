"""Metadata extraction and rule-based judgment classification."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


CATEGORY_KEYWORDS = {
    "Criminal": [
        "bail",
        "anticipatory bail",
        "fir",
        "ipc",
        "crpc",
        "criminal",
        "accused",
        "offence",
        "conviction",
        "sentence",
    ],
    "Corporate": [
        "company",
        "companies act",
        "director",
        "shareholder",
        "insolvency",
        "ibc",
        "nclt",
        "nclat",
        "corporate",
    ],
    "Family": [
        "marriage",
        "divorce",
        "custody",
        "maintenance",
        "matrimonial",
        "adoption",
        "family",
    ],
    "Cyber": [
        "cyber",
        "information technology act",
        "it act",
        "digital",
        "online",
        "electronic record",
        "data protection",
    ],
    "Tax": [
        "income tax",
        "gst",
        "customs",
        "excise",
        "tax",
        "assessment",
        "revenue",
    ],
    "Property": [
        "property",
        "land",
        "possession",
        "title",
        "lease",
        "tenancy",
        "specific performance",
        "sale deed",
    ],
    "Labour": [
        "labour",
        "industrial dispute",
        "workman",
        "employee",
        "employer",
        "service law",
        "termination",
        "wages",
    ],
    "Constitutional": [
        "constitution",
        "article",
        "fundamental rights",
        "writ",
        "constitutional",
        "legislative competence",
    ],
}


def extract_title(text: str, filename: str) -> str:
    """Use the first useful heading-like line, with filename as fallback."""
    for line in text.splitlines()[:30]:
        cleaned = " ".join(line.split())
        if len(cleaned) < 8:
            continue
        if cleaned.lower().startswith(("page ", "item no", "section")):
            continue
        return clean_title(cleaned)
    return Path(filename).stem.replace("_", " ")


def clean_title(title: str) -> str:
    """Remove common citation clutter from case titles."""
    title = re.sub(r"\[[0-9]{4}\]\s*[^:]+:\s*", "", title)
    title = re.sub(r"\b[0-9]{4}\s+INSC\s+[0-9]+\b", "", title, flags=re.I)
    title = re.sub(r"\b[0-9]{4}\s+Latest\s+Caselaw\s+[0-9]+\s+SC\b", "", title, flags=re.I)
    title = re.sub(r"\s+", " ", title).strip(" -:|")
    return (title or "Untitled Supreme Court Case")[:180]


def extract_judge(text: str) -> str:
    """Best-effort judge extraction from common judgment headers."""
    patterns = [
        r"before\s+hon'?ble\s+.*?justice\s+([a-z .]+)",
        r"hon'?ble\s+mr\.?\s+justice\s+([a-z .]+)",
        r"hon'?ble\s+ms\.?\s+justice\s+([a-z .]+)",
        r"justice\s+([a-z .]+)",
    ]
    sample = text[:5000].lower()
    for pattern in patterns:
        match = re.search(pattern, sample, flags=re.IGNORECASE)
        if match:
            return " ".join(match.group(1).split()).title()
    return ""


def classify_category(text: str) -> str:
    """Classify a judgment by counting category keyword hits."""
    lowered = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        scores[category] = sum(lowered.count(keyword) for keyword in keywords)

    best_category = max(scores, key=scores.get)
    return best_category if scores[best_category] > 0 else "General"


def extract_metadata(text: str, filename: str, year: int = 2024) -> Dict:
    """Return normalized case metadata used by chunking and retrieval."""
    return {
        "case_id": Path(filename).stem,
        "title": clean_title(extract_title(text, filename)),
        "year": year,
        "court": "Supreme Court",
        "judge": extract_judge(text),
        "category": classify_category(text),
        "filename": filename,
    }
