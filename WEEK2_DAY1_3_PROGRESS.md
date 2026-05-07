# Week 2 Progress Report (Days 1-3)

## Project Context

This week focused on building the data foundation for the legal case search MVP based on Indian Supreme Court judgments. The goal was to move from raw metadata and bundled judgment PDFs into a structured, parsed, and chunked dataset that is ready for embedding, indexing, and hybrid retrieval.

The work completed so far covers:

1. Metadata download and cleaning
2. PDF extraction and structured parsing
3. Full-text reclassification
4. Paragraph-first chunking with fallback splitting

This gives us a clean pipeline from raw source data to chunk-level retrieval records.

---

## Day 1: Metadata Download and Cleaning

### Objective

Download the 2024 Supreme Court metadata dataset, inspect its schema, clean the records, and save a reliable metadata file for downstream processing.

### Source Dataset

- Downloaded from the public Indian Supreme Court judgments S3 dataset
- Saved raw file to:
  - `data/raw/2024_metadata.parquet`

### Initial Inspection

The raw metadata dataset contained:

- Total raw cases: `782`
- Columns:
  - `title`
  - `petitioner`
  - `respondent`
  - `description`
  - `judge`
  - `author_judge`
  - `citation`
  - `case_id`
  - `cnr`
  - `decision_date`
  - `disposal_nature`
  - `court`
  - `available_languages`
  - `raw_html`
  - `path`
  - `nc_display`
  - `scraped_at`
  - `year`

### Cleaning Steps Performed

The metadata was cleaned with the following logic:

1. Removed duplicate records using `case_id`
2. Removed records missing critical fields:
   - `case_id`
   - `title`
   - `citation`
3. Standardized `decision_date` into proper datetime format
4. Normalized string columns by trimming whitespace
5. Replaced the `court` field with a consistent normalized value:
   - `Supreme Court of India`
6. Dropped the very large `raw_html` field from the cleaned output to keep the metadata file lean and easier to use
7. Added a rule-based `category` field using `title + description`

### Day 1 Output

- Cleaned metadata saved to:
  - `data/processed/2024_metadata_cleaned.parquet`

### Day 1 Results

- Total cases before cleaning: `782`
- Total cases after cleaning: `745`
- Duplicates removed: `37`
- Cases with null dates after cleaning: `0`

### Initial Category Distribution

The first-pass rule-based classification produced:

- `Constitutional`: `8`
- `Corporate`: `49`
- `Criminal`: `40`
- `Family`: `6`
- `General`: `587`
- `Labour`: `9`
- `Property`: `31`
- `Tax`: `15`

### Key Insight from Day 1

The `General` category dominated the cleaned dataset (`587` out of `745`), which showed that title and description alone were not strong enough signals for classification. This was expected for Supreme Court judgments because many case titles are neutral and do not directly reveal the legal subject area.

This led to an important design decision:

- Reclassification should use judgment text, not just metadata

---

## Day 2: PDF Download, Extraction, Parsing, and Reclassification

### Objective

Download the actual 2024 judgment PDFs, extract only the relevant files, parse them into structured JSON, and reclassify cases using judgment text.

### Path Investigation

Before downloading PDFs, the metadata `path` field was inspected.

Sample `path` value:

- `2024_10_108_125`

This was identified as a judgment slug, not a direct file URL.

The open dataset documentation confirmed that judgments are stored inside a tar archive:

- Tar URL:
  - `https://indian-supreme-court-judgments.s3.amazonaws.com/data/tar/year=2024/english/english.tar`
- Tar size:
  - `193.7 MB`

The 2024 English index file showed:

- Total files in archive: `782`
- Every cleaned metadata record matched an archive filename after appending `_EN.pdf`
- Match rate: `745 / 745`

This meant the dataset was complete and consistent enough to proceed with selective extraction.

### Script Written

A standalone script was written:

- `parse_pdfs.py`

### What `parse_pdfs.py` Does

#### Part A: Download and Extract

1. Downloads the 2024 English tar archive if not already present
2. Extracts only the `745` matched PDFs into:
   - `data/raw/pdfs/`
3. Prints extraction progress every `50` files

#### Part B: PDF Parsing

1. Uses PyMuPDF (`fitz`) to extract text from each PDF
2. Saves each parsed case as a structured JSON file to:
   - `data/processed/parsed/{case_id}.json`
3. Uses the following schema:

```json
{
  "case_id": "2024_10_108_125",
  "title": "...",
  "citation": "...",
  "court": "Supreme Court",
  "year": 2024,
  "judge": "...",
  "decision_date": "2024-01-03",
  "category": "...",
  "full_text": "...",
  "total_pages": 18,
  "char_length": 42000
}
```

#### Part C: Reclassification

The category classifier was upgraded to use:

- `title`
- `description`
- first `1000` characters of `full_text`

This significantly improved classification quality compared to the Day 1 metadata-only approach.

### Error Handling

If a PDF failed to parse, the script logged the failure to:

- `data/processed/parse_errors.txt`

and continued processing the rest of the dataset.

### Day 2 Outputs

- Tar archive:
  - `data/raw/english_2024.tar`
- Extracted PDFs:
  - `data/raw/pdfs/`
- Parsed JSON files:
  - `data/processed/parsed/`
- Error log:
  - `data/processed/parse_errors.txt`

### Day 2 Results

- PDFs extracted successfully: `745`
- Parsed successfully: `745`
- Failed parses: `0`
- Average character length per case: `53,069`

### Reclassified Category Distribution

After using judgment text in the classifier, the category distribution improved dramatically:

- `Constitutional`: `93`
- `Corporate`: `73`
- `Criminal`: `285`
- `Cyber`: `3`
- `Family`: `25`
- `General`: `63`
- `Labour`: `52`
- `Property`: `121`
- `Tax`: `30`

### Key Insight from Day 2

The biggest signal improvement was the drop in `General` from `587` to `63`. That confirmed the classification pipeline works much better when it has access to the actual judgment text rather than only metadata.

This was a strong validation that text-aware classification should remain part of the pipeline going forward.

---

## Day 3: Chunking Parsed Cases into Retrieval Records

### Objective

Convert all parsed judgments into chunk-level records suitable for embedding and search.

### Chunking Strategy

The chunking approach reused the paragraph-first strategy built earlier:

1. Try to split on numbered paragraphs using regex
2. If numbered paragraphs are not available, fall back to splitting on blank-line paragraph boundaries
3. If a paragraph is too long, recursively split it into smaller segments with overlap

### Chunking Rules

- Minimum chunk size: `100` characters
- Maximum direct paragraph size before fallback splitting: `1200` characters
- Fallback chunk size: `512` characters
- Fallback overlap: `50` characters

Very short fragments under `100` characters were filtered out as likely headers, footers, page numbers, or formatting artifacts.

### Script Written

A standalone script was written:

- `chunk_cases.py`

### What `chunk_cases.py` Produces

Output file:

- `data/processed/chunks_master.jsonl`

Each line is a chunk in this structure:

```json
{
  "chunk_id": "2024_10_108_125_p3_s0",
  "case_id": "2024_10_108_125",
  "title": "...",
  "citation": "...",
  "court": "Supreme Court",
  "year": 2024,
  "judge": "...",
  "category": "Criminal",
  "decision_date": "2024-01-03",
  "chunk_text": "...",
  "para_index": 3,
  "char_length": 487
}
```

### Day 3 Results

- Total chunks created: `68,316`
- Average chunks per case: `91.70`
- Average chunk character length: `479.49`
- Minimum chunk character length: `100`
- Maximum chunk character length: `1200`

### Chunk Distribution by Category

- `Constitutional`: `15,913`
- `Corporate`: `8,261`
- `Criminal`: `20,623`
- `Cyber`: `64`
- `Family`: `2,017`
- `General`: `4,494`
- `Labour`: `2,960`
- `Property`: `9,265`
- `Tax`: `4,719`

### Validation Performed

The chunk output was verified to ensure:

1. JSONL file was created successfully
2. The first record had the expected schema
3. The chunk ID format followed:
   - `{case_id}_p{para_index}_s{sub_index}`
4. Minimum chunk size filtering worked
5. Chunk size ceiling matched the fallback logic

Output size:

- `chunks_master.jsonl`: approximately `57.9 MB`

### Key Insight from Day 3

The chunking distribution is healthy and gives us a sufficiently large retrieval surface for embedding and hybrid search. With around `68K` chunks from `745` cases, the pipeline is now ready for sample embedding, vector indexing, and backend search development.

---

## Summary of Work Completed So Far

By the end of Day 3, the following pieces are complete:

1. Downloaded and cleaned the 2024 Supreme Court metadata
2. Validated archive structure and matched metadata paths to actual PDFs
3. Downloaded and extracted all required judgment PDFs
4. Parsed every matched PDF into structured JSON
5. Reclassified cases using actual judgment text
6. Chunked the full parsed dataset into retrieval-ready JSONL records

### Current Deliverables

- Cleaned metadata:
  - `data/processed/2024_metadata_cleaned.parquet`
- Extracted PDFs:
  - `data/raw/pdfs/`
- Parsed judgments:
  - `data/processed/parsed/`
- Chunk dataset:
  - `data/processed/chunks_master.jsonl`
- Scripts:
  - `parse_pdfs.py`
  - `chunk_cases.py`

---

## Current Status

The project is in a strong position going into the next phase.

The data pipeline from metadata to parsed cases to chunked retrieval records is now functioning end to end with full coverage over the 2024 dataset subset.

This means the next phase can focus on:

1. Embedding model selection
2. Sample embedding generation
3. Elasticsearch index setup
4. Query backend skeleton
5. Hybrid retrieval

The foundation is stable enough to build on.
