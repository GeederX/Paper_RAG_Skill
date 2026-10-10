"""
RAG Query Engine — Agent Interface
Handles vector retrieval only. No LLM is called inside this module.

The agent calls query_for_agent() to get relevant paper chunks as a context string,
then reads and answers from that context directly — the agent itself is the LLM,
so no secondary model call is needed.

Usage:
    from query_rag import query_for_agent, RAGQueryEngine

    # Simple: get merged context string, then answer from it directly
    context = query_for_agent("your query", top_k=5, expand_neighbors=False)

    # Advanced: get per-document structured results
    engine = RAGQueryEngine()
    doc_chunks = engine.retrieve("your query", top_k=5, expand_neighbors=True)
    # doc_chunks: {doc_id: [{"chunk_id", "chunk_index", "text", "filename"}, ...]}
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from openai import OpenAI

# ==================== Load config ====================
_CONFIG_PATH = Path(__file__).parent / "config.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)

OPENAI_API_KEY = _CFG["api_key"]
OPENAI_BASE_URL = _CFG.get("base_url", "https://api.openai.com/v1")
EMBEDDING_MODEL = _CFG.get("embedding_model", "text-embedding-3-large")
COLLECTION_NAME = _CFG.get("collection_name", "my_papers")
TOP_K = _CFG.get("top_k", 5)
EXPAND_NEIGHBORS = _CFG.get("expand_neighbors", True)

_BASE_DIR = Path(__file__).parent
CHROMA_DB_DIR = _BASE_DIR / "chroma_db"

# ==================== OpenAI client ====================
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


class RAGQueryEngine:
    def __init__(
        self,
        chroma_db_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the RAG query engine.

        Args:
            chroma_db_dir: Path to ChromaDB directory (defaults to config/script-relative path)
            collection_name: ChromaDB collection name (defaults to config value)
        """
        db_path = str(chroma_db_dir) if chroma_db_dir else str(CHROMA_DB_DIR)
        col_name = collection_name or COLLECTION_NAME

        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name=col_name)

        # Read collection metadata (chunk_size, chunk_overlap, etc. from build time)
        self.collection_metadata = self.collection.metadata or {}

        mapping_file = Path(db_path) / "chunk_mapping.json"
        with open(mapping_file, "r", encoding="utf-8") as f:
            self.chunk_mapping = json.load(f)

    def get_embedding(self, text: str) -> List[float]:
        """Embed a query string."""
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding

    def expand_with_neighbors(self, chunk_ids: List[str]) -> List[str]:
        """
        Given retrieved chunk IDs, also include the immediately preceding
        and following chunk from the same document for richer context.
        """
        expanded: set = set()
        for chunk_id in chunk_ids:
            if chunk_id not in self.chunk_mapping:
                continue
            info = self.chunk_mapping[chunk_id]
            doc_id = info["doc_id"]
            idx = info["chunk_index"]
            total = info["total_chunks"]

            expanded.add(chunk_id)
            if idx > 0:
                expanded.add(f"{doc_id}_{idx - 1}")
            if idx < total - 1:
                expanded.add(f"{doc_id}_{idx + 1}")

        return list(expanded)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expand_neighbors: Optional[bool] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: Natural language query string
            top_k: Number of top results to retrieve (defaults to config value)
            expand_neighbors: Whether to include adjacent chunks (defaults to config value)

        Returns:
            Dict keyed by doc_id, each value is a list of chunk dicts sorted by chunk_index:
            {
                "paper_stem": [
                    {"chunk_id": str, "chunk_index": int, "text": str, "filename": str},
                    ...
                ],
                ...
            }
        """
        k = top_k if top_k is not None else TOP_K
        expand = expand_neighbors if expand_neighbors is not None else EXPAND_NEIGHBORS

        query_embedding = self.get_embedding(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        chunk_ids = results["ids"][0]

        if expand:
            chunk_ids = self.expand_with_neighbors(chunk_ids)

        all_chunks = self.collection.get(
            ids=chunk_ids, include=["documents", "metadatas"]
        )

        doc_chunks: Dict[str, List[Dict]] = {}
        for i, chunk_id in enumerate(all_chunks["ids"]):
            meta = all_chunks["metadatas"][i]
            doc_id = meta["doc_id"]
            doc_chunks.setdefault(doc_id, []).append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": meta["chunk_index"],
                    "text": all_chunks["documents"][i],
                    "filename": meta["filename"],
                }
            )

        for doc_id in doc_chunks:
            doc_chunks[doc_id].sort(key=lambda x: x["chunk_index"])

        return doc_chunks


def query_for_agent(
    query: str,
    top_k: Optional[int] = None,
    expand_neighbors: Optional[bool] = None,
) -> str:
    """
    Agent interface — retrieves relevant paper chunks and returns them as a plain string.

    The agent reads this context and answers the user directly.
    No secondary LLM call is needed — the agent itself is the LLM.

    Args:
        query: Natural language query
        top_k: Number of top chunks to retrieve (defaults to config value)
        expand_neighbors: Whether to include adjacent chunks for richer context
            (defaults to config value). Set to False for exact top_k results.

    Returns:
        Merged context string, e.g.:
        【文献: paper.txt】
        ... chunk text ...

        ---

        【文献: paper2.txt】
        ... chunk text ...
    """
    engine = RAGQueryEngine()
    doc_chunks = engine.retrieve(query, top_k=top_k, expand_neighbors=expand_neighbors)

    parts = []
    for doc_id, chunks in doc_chunks.items():
        merged = "\n\n".join(c["text"] for c in chunks)
        parts.append(f"【文献: {chunks[0]['filename']}】\n{merged}")

    return "\n\n---\n\n".join(parts)
