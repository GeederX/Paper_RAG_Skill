# paper-rag skill

A reusable `paper-rag` skill template for agent environments, designed to build a local RAG knowledge base from academic PDF papers.

## Repository Contents

- `skills/paper-rag/SKILL.md`: skill definition and usage instructions
- `skills/paper-rag/references/workflow.md`: detailed workflow and troubleshooting guide
- `skills/paper-rag/assets/PaperRAG/`: runnable template (scripts + config)

## What It Does

- `pdf_to_text_converter.py`: extracts text from PDFs in `papers/`
- `build_rag_index.py`: chunks text and writes embeddings into ChromaDB
- `embeddings.py`: runs `intfloat/multilingual-e5-small` locally by default; also supports a user's OpenAI-compatible HTTP endpoint
- `query_rag.py`: exposes `query_for_agent()` for agent-side retrieval
- Supports incremental updates: re-runs skip already processed content
- The agent asks whether the user has an embedding endpoint; without one, local multilingual E5 needs no API key or GPU

## Local Setup

```bash
cd skills/paper-rag/assets/PaperRAG
python -m venv .venv  # Python 3.10+
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

No API key is needed for the default local model. Place PDFs in `papers/` and run:

```bash
python pdf_to_text_converter.py
python build_rag_index.py
```

The first embedding call downloads model weights from Hugging Face; subsequent runs reuse
the cache. Indexing and retrieval automatically load the CPU model inside the Python
process, with E5 passage/query prefixes and normalized 384-dimensional vectors.
See [the skill instructions](skills/paper-rag/SKILL.md) for endpoint selection and
[the workflow](skills/paper-rag/references/workflow.md) for HTTP authentication and setup.

When migrating an older index or changing embedding models/providers/endpoints, choose a
new `collection_name` and rebuild from `texts/`. Incompatible nonempty indexes are rejected
instead of silently mixing vectors.

## Regression Checks

With the template dependencies installed, run from the repository root:

```bash
python -m unittest discover -s tests -v
```

These checks use temporary ChromaDB indexes; no model download or API key is required.

## Install as a Skill

Copy `skills/paper-rag` into your skills directory (for example `~/.agents/skills/paper-rag`).

## License

This project is licensed under the MIT License. See `LICENSE` for details.
