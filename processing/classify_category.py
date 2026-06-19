"""Rule-based metadata cleanup and legal category classification."""

from __future__ import annotations

import re
from typing import Dict, Iterable


CATEGORY_KEYWORDS = {
    "Criminal": [
        "bail",
        "anticipatory bail",
        "section 438",
        "fir",
        "ipc",
        "crpc",
        "criminal",
        "accused",
        "offence",
        "conviction",
        "sentence",
        "murder",
        "kidnapping",
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
        "arbitration",
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
        "regularisation",
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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def classify_text_parts(parts: Iterable[str]) -> str:
    signal = " ".join(normalize_text(part).lower() for part in parts if part)
    scores = {
        category: sum(signal.count(keyword) for keyword in keywords)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    best_category = max(scores, key=scores.get)
    return best_category if scores[best_category] > 0 else "General"


def classify_case(title: str, description: str, full_text: str, preview_chars: int = 1000) -> str:
    return classify_text_parts([title, description, full_text[:preview_chars]])


def clean_metadata_record(record: Dict) -> Dict:
    """Normalize a parsed case record in one place for downstream steps."""
    cleaned = dict(record)
    for key in ("title", "citation", "judge", "court", "decision_date", "category"):
        if key in cleaned:
            cleaned[key] = normalize_text(str(cleaned[key]))
    return cleaned
