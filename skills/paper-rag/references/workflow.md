# PaperRAG Workflow Guide

## First-Time Setup

1. Copy `assets/PaperRAG/` to the user's working directory
2. Ask whether the user has an OpenAI-compatible HTTP embedding endpoint.
   If not, use the default local `intfloat/multilingual-e5-small` model; no API key is needed.
3. Create a venv with Python 3.10+ and install dependencies:
   ```bash
   cd PaperRAG
   python -m venv .venv
   # Windows:    .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Edit `PaperRAG/config.json` — choose `collection_name`. Leave `embedding_provider: "local"`
   and `embedding_model: "intfloat/multilingual-e5-small"` unless the user chose HTTP.
5. Place PDF files in `PaperRAG/papers/`
6. Run Stage 1: `python pdf_to_text_converter.py`
7. Run Stage 2: `python build_rag_index.py`
8. The index is ready — import `query_rag` in the agent

### Local Model Lifecycle

Both indexing and queries automatically load E5 on the CPU when needed; no daemon or GPU
setup is required. Weights download from Hugging Face on the first run, then remain in its
local cache. Each Python process reuses one model instance. Local batches run with one
model worker regardless of `max_workers`; reduce `batch_size` if RAM is limited.
Only model files are downloaded: local mode never sends paper text to an embedding API.

The model outputs 384-dimensional, normalized embeddings. Chunks get `passage: ` and
queries get `query: ` prefixes, including Chinese and other non-English input. E5 truncates
each input to 512 tokens; a 1000-character multilingual chunk can exceed that limit.

The small model prioritizes local resource usage over maximum retrieval accuracy.
Short or ambiguous cross-language queries can mis-rank papers. Use specific search terms
and inspect retrieved passages before treating them as evidence; multilingual support
does not guarantee correct top-1 results.

### User-Provided HTTP Endpoint

Ask for the API base URL, model ID, and whether authentication is required. The endpoint
must accept OpenAI's embeddings request/response format. For example, change these fields
in `config.json` (keep the remaining chunking/retrieval settings):

```json
{
  "embedding_provider": "http",
  "base_url": "https://embeddings.example/v1",
  "embedding_model": "your-embedding-model",
  "api_key_env": "PAPER_RAG_API_KEY"
}
```

The SDK appends `/embeddings` to `base_url`; do not include that route in the base URL.
If authentication is required, have the user securely set the selected environment
variable in the environment running the scripts. Never request a secret in chat or put
one in `config.json`. Leave the variable unset for an unauthenticated endpoint.
The HTTP path never loads local E5 weights; it sends chunks/queries to the chosen endpoint.
Official `multilingual-e5-*` model IDs get E5 prefixes; other IDs get plain text. If the
service uses an E5 alias, it must supply the appropriate preprocessing itself.

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

## Changing Embeddings / Migrating an Older Index

Never append or query embeddings from another model, provider, endpoint, or input format,
even when dimensions match. Both build and query reject a nonempty incompatible index.
Older indexes without provider metadata also require a rebuild.

1. Update the template scripts and `config.json` together. Remove the obsolete `api_key`
   field; HTTP credentials now come from the environment variable named by `api_key_env`.
2. Choose local E5 or configure the user's HTTP endpoint.
3. Set a **new `collection_name`**, preserving the old collection.
4. Run `python build_rag_index.py` against the existing `texts/`.

This re-embeds all text into the selected embedding space. If an endpoint changes its
served model while retaining the same URL and model ID, rebuild as well; that server-side
change cannot be detected from the saved configuration.


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

### Local model download fails
The first local run needs internet access to Hugging Face and disk space for model files.
Restore access and rerun; subsequent runs reuse cached weights. Do not silently switch to
a cloud API. If the user prefers their own HTTP endpoint, configure it explicitly.

### HTTP authentication errors
Check that `api_key_env` names the environment variable set in the script's process.
Do not print the key. Local mode never needs it; unauthenticated HTTP endpoints can omit it.

### Embedding configuration differs from this index
Use a new `collection_name` and rebuild from `texts/`, or restore the configuration that
originally built the collection. Do not relabel old vectors with new model metadata.

### Rate limit errors during embedding
For HTTP mode, reduce `max_workers` and/or `batch_size` and re-run `build_rag_index.py`.
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
| `embedding_provider` | string | `local` | `local` or `http` |
| `embedding_model` | string | `intfloat/multilingual-e5-small` | Fixed locally; user's model ID for HTTP |
| `base_url` | string | empty | Required only for HTTP; API base, not `/embeddings` |
| `api_key_env` | string | `PAPER_RAG_API_KEY` | Optional HTTP key's environment variable name |
| `collection_name` | string | `my_papers` | Short, descriptive, no spaces |
| `chunk_size` | int | 1000 | Characters per chunk |
| `chunk_overlap` | int | 100 | Overlap between adjacent chunks |
| `max_workers` | int | 20 | HTTP threads; local always uses one model worker |
| `batch_size` | int | 32 | Texts per local/HTTP embedding batch |
| `top_k` | int | 5 | Default retrieval count |
| `expand_neighbors` | bool | true | Fetch adjacent chunks for context |
