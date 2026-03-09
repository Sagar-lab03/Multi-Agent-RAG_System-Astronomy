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
    ingest/       # loaders, chunking, SQLite store
    retrieval/    # embeddings (HF API), vector search
    routing/      # query intent router
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
- **chunks**: `chunk_id`, `doc_id`, `chunk_index`, `text`, `metadata_json`, `created_at`, `embedding` (BLOB, added by retrieval index)  
Use [DB Browser for SQLite](https://sqlitebrowser.org/) or any SQL client to inspect `data/processed/chunks.db`.

### Output format (chunk records)
Each chunk has: `doc_id` (sha256 of file bytes), `chunk_id`, `text`, and `metadata` (offsets, file type, title, page for PDFs).

---

## Embeddings & vector search

Embeddings are stored in the same SQLite DB (`chunks.embedding` BLOB).

**Default: local model** (`sentence-transformers/all-MiniLM-L6-v2`) via `sentence-transformers` — no API key, runs on CPU.

### Setup
1. Install deps: `pip install -r requirements.txt` (includes `sentence-transformers`, `python-dotenv`)
2. (Optional) Use another model: set in `.env`: `EMBEDDING_MODEL=thenlper/gte-small`
3. (Optional) Use Hugging Face Inference API instead of local: set `EMBEDDING_BACKEND=huggingface` and `HF_TOKEN=hf_xxxx` in `.env`

### Index (embed all chunks)
Run once after ingestion, or after adding new documents:

```powershell
python .\retrieval.py index --db data\processed\chunks.db
```

To set batch size (default - 32)

```powershell
python .\retrieval.py index --db data\processed\chunks.db --batch-size 8
```

### Search
By default search uses **hybrid** mode (semantic + BM25):

```powershell
python .\retrieval.py search --db data\processed\chunks.db --query "What is gravitational redshift?" --top-k 5
```

Other modes:

```powershell
# Semantic-only:
python .\retrieval.py search --db data\processed\chunks.db -q "black holes" --top-k 3 --mode semantic

# Lexical-only (BM25):
python .\retrieval.py search --db data\processed\chunks.db -q "black holes" --top-k 3 --mode lexical

# Hybrid with custom semantic weight (alpha):
python .\retrieval.py search --db data\processed\chunks.db -q "black holes" --top-k 3 --mode hybrid --alpha 0.7

# JSON output:
python .\retrieval.py search --db data\processed\chunks.db -q "black holes" --top-k 3 --json
```

---

Note: BM25 is cached **within the running Python process**. The CLI runs one query per process, so caching matters most once we wire retrieval into a long-running agent/service.

---

### Retrieval REPL (interactive, uses cache)

For quick experiments in one process (sharing BM25 cache and the local embedder):

```powershell
python .\retrieval.py repl --db data\processed\chunks.db --mode hybrid --top-k 3
```

Then type queries at the `query>` prompt. Use `--mode semantic` or `--mode lexical` to compare behaviors.

---

Next steps:
- multi-agent orchestration (retriever/answerer/verifier)

## Question Answering (DOCUMENT_SEARCH, Gemini-based)

High-level flow:
- Retriever agent selects top-k context chunks from the SQLite store (semantic / lexical / hybrid).
- Answer agent (Gemini) generates an answer grounded in those chunks, with inline citations like `[1]`, `[2]`.

### Setup
1. Create a Gemini API key and add it to `.env`:
   - `GEMINI_API_KEY=your_key_here`
2. Install deps (as before): `pip install -r requirements.txt`

### Run QA pipeline
```powershell
python .\qa.py --db data\processed\chunks.db --query "What is gravitational redshift?" --top-k 6 --mode hybrid
```

- `--top-k`: number of context chunks retrieved.
- `--mode`: retriever mode (`semantic`, `lexical`, or `hybrid`).
- `--json`: print full JSON (answer, citations, context).

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

