# paper-rag skill

A reusable `paper-rag` skill template for agent environments, designed to build a local RAG knowledge base from academic PDF papers.

## Repository Contents

- `skills/paper-rag/SKILL.md`: skill definition and usage instructions
- `skills/paper-rag/references/workflow.md`: detailed workflow and troubleshooting guide
- `skills/paper-rag/assets/PaperRAG/`: runnable template (scripts + config)

## What It Does

- `pdf_to_text_converter.py`: extracts text from PDFs in `papers/`
- `build_rag_index.py`: chunks text and writes embeddings into ChromaDB
- `query_rag.py`: exposes `query_for_agent()` for agent-side retrieval
- Supports incremental updates: re-runs skip already processed content

## Local Validation (Optional)

```bash
cd skills/paper-rag/assets/PaperRAG
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then set `api_key` in `config.json`, place PDFs in `papers/`, and run:

```bash
python pdf_to_text_converter.py
python build_rag_index.py
```

## Install as a Skill

Copy `skills/paper-rag` into your skills directory (for example `~/.agents/skills/paper-rag`).

## License

This project is licensed under the MIT License. See `LICENSE` for details.
