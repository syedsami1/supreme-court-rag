# Legal RAG MVP

Local-only RAG pipeline for Indian Supreme Court judgment search.

Default ingestion uses the public `vanga/indian-supreme-court-judgments` S3 dataset.
Indian Kanoon ingestion is available as optional code, but it requires an API token.

## Install

```bash
pip install -r requirements.txt
```

## Add PDFs

Place PDFs in:

```bash
data/raw/
```

If `data/raw/` is empty, the pipeline downloads 25 English Supreme Court PDFs for 2020
from the public open dataset and saves metadata to `data/metadata.json`.

## Run

```bash
python main.py
```

## Example Query

```text
anticipatory bail in criminal cases
```

## Expected Output Shape

```json
{
  "query": "anticipatory bail in criminal cases",
  "grouped_results": {
    "Criminal": [
      {
        "case_id": "2024_case_file",
        "title": "Example Case Title",
        "category": "Criminal",
        "snippet": "The Court considered the principles governing anticipatory bail...",
        "score": 0.8732
      }
    ]
  }
}
```

## Pipeline

1. Extract text from PDFs using PyMuPDF.
2. Extract simple metadata and classify category with keyword rules.
3. Chunk by numbered paragraphs with recursive fallback for long chunks.
4. Embed chunks locally with `sentence-transformers/all-MiniLM-L6-v2`.
5. Build FAISS vector index.
6. Build BM25 keyword index.
7. Merge BM25 and vector scores.
8. Deduplicate results by `case_id`.
9. Group final results by legal category.
