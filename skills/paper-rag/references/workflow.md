# PaperRAG Workflow Guide

## First-Time Setup

1. Copy `assets/PaperRAG/` to the user's working directory
2. Create a venv and install dependencies:
   ```bash
   cd PaperRAG
   python -m venv .venv
   # Windows:    .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Edit `PaperRAG/config.json` — set `api_key`, `base_url`, and `collection_name`
4. Place PDF files in `PaperRAG/papers/`
5. Run Stage 1: `python pdf_to_text_converter.py`
6. Run Stage 2: `python build_rag_index.py`
7. The index is ready — import `query_rag` in the agent

## Adding New Papers (Incremental Update)

Both scripts are **idempotent** — safe to re-run at any time.

1. Copy new PDFs into `PaperRAG/papers/`
2. Run Stage 1 again — only unconverted PDFs are processed (existing `.txt` files are skipped)
3. Run Stage 2 again — only new chunks are embedded (existing chunk IDs in ChromaDB are skipped)

No need to rebuild from scratch. The ChromaDB collection accumulates entries across runs.

> **Note:** `chunk_mapping.json` is always fully rewritten on each Stage 2 run to reflect
> the complete state of all text files. This is required for neighbor expansion to work correctly.

## Changing `collection_name`

If you change `collection_name` in `config.json` after the index has been built:
- A **new empty collection** will be created on the next `build_rag_index.py` run
- The old collection still exists in `chroma_db/` under its original name
- Re-run Stage 2 to rebuild the index under the new name

To delete the old collection, either delete the `chroma_db/` directory entirely and rebuild,
or use the ChromaDB Python API:
```python
import chromadb
client = chromadb.PersistentClient(path="chroma_db")
client.delete_collection("old_collection_name")
```

## Rebuilding the Index from Scratch

Delete the generated directories and re-run (venv must be activated):

```bash
# From inside PaperRAG/
rm -rf texts/ chroma_db/
python pdf_to_text_converter.py
python build_rag_index.py
```

## Troubleshooting

### `[ERROR] texts/ directory is empty`
Run `pdf_to_text_converter.py` first to populate `texts/`.

### `[ERROR] Please set api_key in config.json`
Open `config.json` and replace `"your-api-key-here"` with a valid API key.

### Rate limit errors during embedding
Reduce `max_workers` and/or `batch_size` in `config.json` and re-run `build_rag_index.py`.
The script will skip already-embedded chunks and only process the remaining ones.

### `failed_chunks.jsonl` appears in `chroma_db/`
Some chunks failed to embed. Inspect the file to see reasons. Re-running `build_rag_index.py`
will retry only the chunks that are not yet in the collection.

### PDF produces empty or very short text
Some PDFs are image-based (scanned) and `pdfplumber` cannot extract text from them.
These require OCR preprocessing (e.g., `pytesseract`, `pypdf2` with OCR) before ingestion.

### ChromaDB collection not found on query
Make sure `collection_name` in `config.json` matches the name used when building the index.
Run `build_rag_index.py` if the collection does not exist yet.

## Config Reference

| Field | Type | Default | Notes |
|---|---|---|---|
| `api_key` | string | — | Required |
| `base_url` | string | `https://api.openai.com/v1` | Change for proxies |
| `embedding_model` | string | `text-embedding-3-large` | Or `text-embedding-3-small` |
| `collection_name` | string | `my_papers` | Short, descriptive, no spaces |
| `chunk_size` | int | 1000 | Characters per chunk |
| `chunk_overlap` | int | 100 | Overlap between adjacent chunks |
| `max_workers` | int | 20 | Parallel embedding threads |
| `batch_size` | int | 50 | Chunks per embedding API call |
| `top_k` | int | 5 | Default retrieval count |
| `expand_neighbors` | bool | true | Fetch adjacent chunks for context |
