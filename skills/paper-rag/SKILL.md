---
name: paper-rag
description: >
  Set up and use a local paper RAG (Retrieval-Augmented Generation) knowledge base
  for academic PDF papers. Use this skill when the user wants to: build a paper
  knowledge base, create a PaperRAG directory, index research papers, search or
  retrieve content from a paper library, add new papers to an existing RAG index,
  query a paper database, set up a literature retrieval system, or use phrases like
  "论文RAG库", "文献知识库", "PaperRAG", "搭建论文检索", "论文向量库", "build paper rag",
  "setup research RAG", "paper knowledge base", "从论文库中查找", "检索文献".
  The agent calls query_for_agent() to retrieve relevant paper chunks, then reads and summarizes
  the returned context directly — no secondary LLM call needed, as the agent itself is the LLM.
---

# PaperRAG Skill

A 3-stage pipeline that turns a folder of PDF papers into a vector search index for agent use.
All scripts live inside `PaperRAG/` alongside the data directories.

## Directory Structure

Copy the entire `assets/PaperRAG/` template into the user's working directory.

```
PaperRAG/
├── .venv/                       ← Virtual environment (created during setup)
├── config.json                  ← Edit before first run
├── requirements.txt
├── papers/                      ← Drop PDF files here
├── texts/                       ← Auto-created by Stage 1
├── chroma_db/                   ← Auto-created by Stage 2
├── pdf_to_text_converter.py     ← Stage 1
├── build_rag_index.py           ← Stage 2
└── query_rag.py                 ← Stage 3 (agent import)
```

## Setup

### 1. Create virtual environment and install dependencies

Always use a `.venv` inside `PaperRAG/` to keep dependencies isolated.

**Windows (PowerShell):**
```powershell
cd PaperRAG
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
cd PaperRAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The venv only needs to be created once. To reactivate later: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux).

### 2. Edit `config.json`

```json
{
  "api_key": "sk-...",
  "base_url": "https://api.openai.com/v1",
  "embedding_model": "text-embedding-3-large",
  "collection_name": "my_papers",
  "chunk_size": 1000,
  "chunk_overlap": 100,
  "max_workers": 20,
  "batch_size": 50,
  "top_k": 5,
  "expand_neighbors": true
}
```

| Field | Description |
|---|---|
| `api_key` | OpenAI-compatible API key (required) |
| `base_url` | API endpoint — change for custom proxies |
| `embedding_model` | `text-embedding-3-large` (best) or `text-embedding-3-small` (cheaper) |
| `collection_name` | ChromaDB collection name — use a short descriptive name |
| `chunk_size` | Characters per chunk (default 1000) |
| `chunk_overlap` | Overlap between chunks for context continuity (default 100) |
| `max_workers` | Parallel embedding threads (reduce if hitting rate limits) |
| `batch_size` | Chunks per API batch (reduce if hitting rate limits) |
| `top_k` | Default number of chunks to retrieve per query |
| `expand_neighbors` | Include adjacent chunks for richer context (recommended: `true`) |

## Running the Pipeline

All commands run from inside `PaperRAG/` with the venv activated.

```bash
cd PaperRAG
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
```

### Stage 1 — PDF → Text

```bash
python pdf_to_text_converter.py
```

Reads `papers/*.pdf`, writes `texts/*.txt`. Safe to re-run — skips already-converted files.

### Stage 2 — Build Index

**Before running this stage, ask the user about `chunk_size` and `chunk_overlap`:**

> "Before building the index: chunk_size controls how many characters each text segment contains (default 1000), and chunk_overlap controls how much adjacent chunks overlap (default 100). Larger chunk_size gives more context per chunk but less precision; larger overlap helps preserve continuity across chunk boundaries. Would you like to adjust these, or use the defaults?"

If the user is unsure, keep the defaults. Update `config.json` accordingly before running.

```bash
python build_rag_index.py
```

Reads `texts/*.txt`, generates embeddings, stores in `chroma_db/`. Safe to re-run — skips already-embedded chunks (incremental update).

## Agent Integration

`query_rag.py` is the agent-facing module. It handles only vector retrieval — no LLM is called
inside the module. The agent receives the retrieved paper chunks as a context string and
**summarizes or answers directly from that context**, since the agent itself is the LLM.
There is no need to forward the context to any secondary model.

### Simple interface

```python
import sys
sys.path.insert(0, "/path/to/PaperRAG")
from query_rag import query_for_agent

# Returns a merged context string — the agent reads this and answers directly
context = query_for_agent("your question or keywords", top_k=5, expand_neighbors=False)

# expand_neighbors (default: config value, True): when True, also includes the chunk
# immediately before and after each top match for richer context. Set to False for
# exact top_k results only.
```

> **Note:** When importing `query_rag` from outside `PaperRAG/`, make sure the venv's
> site-packages are on the path, or run the agent process with the venv's Python interpreter:
> `PaperRAG/.venv/bin/python` (macOS/Linux) or `PaperRAG\.venv\Scripts\python.exe` (Windows).

### Advanced interface

```python
from query_rag import RAGQueryEngine

engine = RAGQueryEngine()
doc_chunks = engine.retrieve("your query", top_k=5, expand_neighbors=True)

# doc_chunks: {doc_id: [{"chunk_id", "chunk_index", "text", "filename"}, ...]}
for doc_id, chunks in doc_chunks.items():
    for chunk in chunks:
        print(chunk["filename"], chunk["text"])
```

### Typical agent pattern

```python
context = query_for_agent("the user's question")

# The agent IS the LLM — read the context and answer directly.
# No secondary LLM call needed.
# e.g.: "Based on the retrieved literature: {context}\n\nAnswer: ..."
```

### Inspecting build-time metadata

The `RAGQueryEngine` exposes `collection_metadata` with the parameters used when the
index was built. This is useful for the agent to understand the chunking strategy:

```python
engine = RAGQueryEngine()
print(engine.collection_metadata)
# {'chunk_size': 1000, 'chunk_overlap': 100, 'embedding_model': 'text-embedding-3-large', ...}
```

## Workflow Details

See `references/workflow.md` for:
- Step-by-step first-time setup walkthrough
- Adding new papers incrementally
- Troubleshooting common errors
