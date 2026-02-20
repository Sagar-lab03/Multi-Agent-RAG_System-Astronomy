# Production-Grade Multi-Agent RAG System (Step-by-step)

This repository is built **step-by-step** to develop a production-thinking, debuggable RAG system.

## Step 1 (current): Document ingestion & chunking

### What you get
- **Loaders** for `.txt/.md`, `.html/.htm`, `.pdf`
- **Chunking** with overlap and boundary-aware splitting (prefers paragraph/sentence/whitespace)
- **Chunk store**: SQLite DB for documents and chunks (easy to inspect and query); optional JSONL export

### Folder layout
```
data/
  raw/            # put your source docs here (pdf/html/txt/md)
  processed/      # chunks.db (SQLite) and optionally chunks.jsonl
src/
  rag_system/
    ingest/
```

### Setup (Windows / PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run ingestion
Put documents under `data/raw/`, then run **at least one** of `--db` or `--output`:

```powershell
# SQLite chunk store (recommended; open with DB Browser or any SQL client):
python .\ingest.py --input data\raw --db data\processed\chunks.db

# JSONL only:
python .\ingest.py --input data\raw --output data\processed\chunks.jsonl

# Both:
python .\ingest.py --input data\raw --db data\processed\chunks.db --output data\processed\chunks.jsonl
```

### SQLite schema
- **documents**: `doc_id`, `source_path`, `source_ext`, `metadata_json`, `ingested_at`
- **chunks**: `chunk_id`, `doc_id`, `chunk_index`, `text`, `metadata_json`, `created_at`  
Use [DB Browser for SQLite](https://sqlitebrowser.org/) or any SQL client to inspect `data/processed/chunks.db`.

### Output format (chunk records)
Each chunk has: `doc_id` (sha256 of file bytes), `chunk_id`, `text`, and `metadata` (offsets, file type, title, page for PDFs).

---

Next steps after Step 1:
- embeddings + vector index
- hybrid retrieval (BM25 + vectors)
- multi-agent orchestration (retriever/answerer/verifier)

## Query routing (intents)

The router classifies a user query into **exactly one** label:
`DOCUMENT_SEARCH`, `APOD`, `NEO`, `DONKI`, `EONET`, or `UNKNOWN`.

### Run routing (label-only stdout)
```powershell
python .\route.py "show today's astronomy picture"
```

### Debug routing (trace to stderr; stdout remains label-only)
```powershell
python .\route.py "asteroids approaching earth this week" --debug
```

