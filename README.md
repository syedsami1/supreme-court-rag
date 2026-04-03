# Supreme Court RAG 

A clean, modular Retrieval-Augmented Generation (RAG) pipeline for answering questions from Supreme Court case PDFs.

## Tech Stack

- PDF parsing: PyMuPDF (`fitz`)
- Embeddings: `sentence-transformers` with `all-MiniLM-L6-v2`
- Vector store: FAISS
- LLM: Groq API (`llama3-8b-8192`)
- Frameworks: No LangChain (custom modules only)

## Project Structure

```text
.
├── data/                   # Place Supreme Court PDF files here
├── src/
│   ├── loader.py           # PDF loader + per-page metadata extraction
│   ├── splitter.py         # Word-based chunking (500/100)
│   ├── embedder.py         # SentenceTransformer embeddings
│   ├── vectorstore.py      # FAISS index build/load/search
│   ├── retriever.py        # Retrieve top-k chunks
│   ├── generator.py        # Groq API call (llama3-8b-8192)
│   └── main.py             # CLI app
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set your Groq API key:

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

4. Ensure your PDFs are inside the `data/` folder.

## How To Run

Run the CLI:

```powershell
python src/main.py
```

Optional flags:

- Rebuild index:

```powershell
python src/main.py --rebuild
```

- Change retrieval size:

```powershell
python src/main.py --top-k 7
```

- Use custom data or index directory:

```powershell
python src/main.py --data-dir data --index-dir vector_store
```

## What The Pipeline Does

1. Loads all PDFs from `data/` and extracts text page-wise.
2. Splits each page into overlapping chunks:
- Chunk size: ~500 words
- Overlap: 100 words
3. Creates embeddings using `all-MiniLM-L6-v2`.
4. Builds a FAISS index and saves it locally (`vector_store/faiss.index` + `vector_store/chunks.json`).
5. For each query:
- Embeds query
- Retrieves top-k relevant chunks
- Sends context + question to Groq (`llama3-8b-8192`)
- Enforces instruction:
  - `Answer ONLY from the context. If not found, say 'I don't know'.`
6. Prints answer and source pages (`PDF name + page`).

## Design Decisions

- **Custom modular architecture**: Keeps each concern isolated (`loader`, `splitter`, `embedder`, `retriever`, `generator`), making it easy to test and swap components.
- **Word-based chunking**: Simple and deterministic for legal text; overlap preserves continuity across chunk boundaries.
- **Normalized embeddings + FAISS IndexFlatIP**: Enables cosine-similarity-style retrieval while keeping implementation straightforward.
- **Persistent local index**: Avoids recomputing embeddings every run; `--rebuild` updates index when PDFs change.
- **Explicit grounding prompt**: Reduces hallucinations by forcing context-only answers with fallback `"I don't know"`.

## Error Handling Included

- Missing `data/` directory or no PDFs found
- PDF parsing failures (continues processing remaining files)
- Invalid chunking parameters
- Missing Groq API key
- Missing/corrupt FAISS or metadata files
- API/network/response-format failures from Groq
