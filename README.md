# Supreme Court RAG

This repository now follows a PRD-aligned structure for the Indian Supreme Court case search MVP.

## Current Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── processing/
│   ├── classify_category.py
│   ├── parse_pdfs.py
│   ├── chunk_judgment.py
│   └── build_chunks.py
├── embedding/
│   └── embed_chunks.py
├── indexing/
│   └── faiss_index.py
├── retrieval/
│   └── bm25_search.py
├── requirements.txt
└── main.py                # to be expanded as backend/search entrypoint
```

## What Is Ready

- `data/processed/2024_metadata_cleaned.parquet`
- `data/processed/parsed/`
- `data/processed/chunks_master.jsonl`
- paragraph-first chunking with fallback splitting
- local embedding module using Sentence Transformers and `BAAI/bge-small-en-v1.5`
- FAISS index helper
- BM25 helper

## Key Scripts

- Parse PDFs:

```powershell
python -m processing.parse_pdfs
```

- Build chunk JSONL:

```powershell
python -m processing.build_chunks
```

- Embed chunks:

```powershell
python -m embedding.embed_chunks
```

## Embedding Model

The active local embedding model is:

- `BAAI/bge-small-en-v1.5`

It is lightweight, free, local, and works through `sentence-transformers`.
