"""
RAG Index Builder
- Reads text files from texts/ directory
- Chunks text with overlap and sentence-boundary awareness
- Generates OpenAI embeddings in parallel batches
- Stores vectors and metadata in ChromaDB
- Supports incremental re-runs (skips already-embedded chunks)
"""

import json
from pathlib import Path
from typing import List, Dict
import chromadb
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== Load config ====================
_CONFIG_PATH = Path(__file__).parent / "config.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)

OPENAI_API_KEY = _CFG["api_key"]
OPENAI_BASE_URL = _CFG.get("base_url", "https://api.openai.com/v1")
EMBEDDING_MODEL = _CFG.get("embedding_model", "text-embedding-3-large")
COLLECTION_NAME = _CFG.get("collection_name", "my_papers")
CHUNK_SIZE = _CFG.get("chunk_size", 1000)
CHUNK_OVERLAP = _CFG.get("chunk_overlap", 100)
MAX_WORKERS = _CFG.get("max_workers", 20)
BATCH_SIZE = _CFG.get("batch_size", 50)

# Paths are always relative to this script's directory (PaperRAG/)
_BASE_DIR = Path(__file__).parent
TEXTS_DIR = _BASE_DIR / "texts"
CHROMA_DB_DIR = _BASE_DIR / "chroma_db"

SKIP_EXISTING = True
EXISTING_ID_PAGE_SIZE = 5000

# ==================== OpenAI client ====================
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def chunk_text(
    text: str, filename: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[Dict]:
    """
    Split text into overlapping chunks with sentence-boundary awareness.

    Returns:
        List of dicts: {"text", "chunk_id", "doc_id", "chunk_index", "filename"}
    """
    chunks = []
    start = 0
    chunk_index = 0
    doc_id = Path(filename).stem

    while start < len(text):
        end = start + chunk_size
        chunk_str = text[start:end]

        # Try to end on a sentence boundary
        if end < len(text):
            last_period = max(
                chunk_str.rfind("."),
                chunk_str.rfind("!"),
                chunk_str.rfind("?"),
                chunk_str.rfind("\n\n"),
            )
            if last_period > chunk_size * 0.7:
                end = start + last_period + 1
                chunk_str = text[start:end]

        cleaned = chunk_str.strip()
        if cleaned:
            chunks.append(
                {
                    "text": cleaned,
                    "chunk_id": f"{doc_id}_{chunk_index}",
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "filename": filename,
                }
            )

        chunk_index += 1
        start = end - overlap

    return chunks


def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text string."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def build_index():
    """Build (or incrementally update) the RAG index."""
    print("Starting RAG index build...", flush=True)
    print(f"  Collection : {COLLECTION_NAME}", flush=True)
    print(f"  Embed model: {EMBEDDING_MODEL}", flush=True)
    print(f"  API base   : {OPENAI_BASE_URL}\n", flush=True)

    # 1. Init ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print("[OK] Loaded existing ChromaDB collection", flush=True)
    except Exception:
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "embedding_model": EMBEDDING_MODEL,
                "description": "PaperRAG index",
            },
        )
        print("[OK] Created new ChromaDB collection", flush=True)

    # 2. Read text files
    text_files = list(TEXTS_DIR.glob("*.txt"))
    print(f"\nFound {len(text_files)} text files in {TEXTS_DIR}")

    if not text_files:
        print("[ERROR] texts/ directory is empty. Run pdf_to_text_converter.py first.")
        return

    # 3. Chunk all files
    all_chunks: List[Dict] = []
    chunk_mapping: Dict = {}

    print("\nChunking text files...")
    for file_path in tqdm(text_files, desc="Chunking"):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if len(content.strip()) < 100:
                continue
            chunks = chunk_text(content, file_path.name)
            all_chunks.extend(chunks)
            for chunk in chunks:
                chunk_mapping[chunk["chunk_id"]] = {
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": len(chunks),
                }
        except Exception as e:
            print(f"\n[WARNING] Failed to process {file_path.name}: {e}")

    print(f"\nTotal chunks generated: {len(all_chunks)}")

    # 4. Incremental resume: skip already-embedded chunks
    if SKIP_EXISTING:
        print("\nChecking for already-embedded chunks (resume support)...")
        existing_ids = set()
        offset = 0
        while True:
            page = collection.get(
                limit=EXISTING_ID_PAGE_SIZE, offset=offset, include=[]
            )
            ids = page.get("ids", [])
            if not ids:
                break
            existing_ids.update(ids)
            offset += len(ids)
        print(f"  Already embedded: {len(existing_ids)} chunks")
        before = len(all_chunks)
        all_chunks = [c for c in all_chunks if c["chunk_id"] not in existing_ids]
        print(f"  Chunks to embed : {len(all_chunks)} / {before}")

    # Save chunk mapping (needed by query_rag.py for neighbor expansion)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    mapping_file = CHROMA_DB_DIR / "chunk_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(chunk_mapping, f, ensure_ascii=False, indent=2)
    print(f"\nSaved chunk mapping -> {mapping_file}")

    if not all_chunks:
        print("\n[OK] Index is already up to date. Nothing to embed.")
        return

    # 5. Embed in parallel batches
    print(
        f"\nGenerating embeddings ({MAX_WORKERS} threads, batch size {BATCH_SIZE})..."
    )
    failed_log = CHROMA_DB_DIR / "failed_chunks.jsonl"

    def process_batch(batch_data):
        batch_idx, batch = batch_data
        try:
            texts = [
                c["text"]
                for c in batch
                if isinstance(c.get("text"), str) and c["text"].strip()
            ]
            if not texts:
                return batch_idx, [], [], None
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            embeddings = [item.embedding for item in response.data]
            return batch_idx, batch, embeddings, None
        except Exception as e:
            return batch_idx, batch, None, str(e)

    batches = [
        (i // BATCH_SIZE, all_chunks[i : i + BATCH_SIZE])
        for i in range(0, len(all_chunks), BATCH_SIZE)
    ]
    success_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_batch, b): b[0] for b in batches}
        with tqdm(total=len(batches), desc="Embedding batches") as pbar:
            for future in as_completed(futures):
                batch_idx, batch, embeddings, error = future.result()

                if error:
                    print(f"\n[WARNING] Batch {batch_idx + 1} failed: {error}")
                    failed_count += 1
                    # Fallback: embed one-by-one to salvage what we can
                    for chunk in batch:
                        text = chunk.get("text", "")
                        if not isinstance(text, str) or not text.strip():
                            with open(failed_log, "a", encoding="utf-8") as fl:
                                fl.write(
                                    json.dumps(
                                        {
                                            "chunk_id": chunk.get("chunk_id"),
                                            "reason": "empty_text",
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            continue
                        try:
                            emb = (
                                client.embeddings.create(
                                    model=EMBEDDING_MODEL, input=text
                                )
                                .data[0]
                                .embedding
                            )
                            collection.add(
                                ids=[chunk["chunk_id"]],
                                embeddings=[emb],
                                documents=[text],
                                metadatas=[
                                    {
                                        "doc_id": chunk["doc_id"],
                                        "chunk_index": chunk["chunk_index"],
                                        "filename": chunk["filename"],
                                    }
                                ],
                            )
                        except Exception as e2:
                            with open(failed_log, "a", encoding="utf-8") as fl:
                                fl.write(
                                    json.dumps(
                                        {
                                            "chunk_id": chunk.get("chunk_id"),
                                            "reason": str(e2),
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                else:
                    try:
                        collection.add(
                            ids=[c["chunk_id"] for c in batch],
                            embeddings=embeddings,
                            documents=[c["text"] for c in batch],
                            metadatas=[
                                {
                                    "doc_id": c["doc_id"],
                                    "chunk_index": c["chunk_index"],
                                    "filename": c["filename"],
                                }
                                for c in batch
                            ],
                        )
                        success_count += 1
                    except Exception as e:
                        print(f"\n[WARNING] Batch {batch_idx + 1} store failed: {e}")
                        failed_count += 1

                pbar.update(1)

    print(
        f"\n[DONE] Embedding complete: {success_count} batches succeeded, {failed_count} failed"
    )
    print(f"[DONE] Index build complete!")
    print(f"  Papers  : {len(text_files)}")
    print(f"  New chunks embedded: {len(all_chunks)}")
    print(f"  DB path : {CHROMA_DB_DIR}")
    print(f"  Collection: {COLLECTION_NAME}")

    # Persist build parameters as collection metadata for future queries
    try:
        collection.modify(
            metadata={
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "embedding_model": EMBEDDING_MODEL,
                "description": "PaperRAG index",
            }
        )
        print(f"  Metadata : chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    except Exception as e:
        print(f"  [WARNING] Could not update collection metadata: {e}")


if __name__ == "__main__":
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
        print("[ERROR] Please set api_key in config.json")
        exit(1)
    build_index()
