"""Shared local E5 / OpenAI-compatible HTTP embeddings for indexing and queries."""

import os
from functools import lru_cache

LOCAL_MODEL = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer

    # CPU works without a GPU on macOS, Linux, and Windows. Reuse within a process.
    model = SentenceTransformer(LOCAL_MODEL, device="cpu")
    model.max_seq_length = 512
    return model


class EmbeddingBackend:
    def __init__(self, config):
        self.provider = config.get("embedding_provider", "local")
        if self.provider not in {"local", "http"}:
            raise ValueError("embedding_provider must be 'local' or 'http'")
        self.model_name = config.get("embedding_model", LOCAL_MODEL)
        self.base_url = config.get("base_url", "").rstrip("/")
        self.batch_size = config.get("batch_size", 32)
        self.client = None
        if self.provider == "local":
            if self.model_name != LOCAL_MODEL:
                raise ValueError(f"Local embeddings use {LOCAL_MODEL}; update embedding_model")
        else:
            if not self.base_url or not self.model_name:
                raise ValueError("HTTP embeddings require base_url and embedding_model")
            from openai import OpenAI

            # An unauthenticated endpoint still needs a nonempty SDK placeholder.
            key = os.environ.get(config.get("api_key_env", "PAPER_RAG_API_KEY"))
            self.client = OpenAI(api_key=key or "not-required", base_url=self.base_url)
        self.e5_prefixes = self.model_name.rsplit("/", 1)[-1].startswith("multilingual-e5-")

    @property
    def metadata(self):
        return {
            "embedding_provider": self.provider,
            "embedding_model": self.model_name,
            "embedding_base_url": self.base_url if self.provider == "http" else "",
            "embedding_format": "e5-query-passage-v1" if self.e5_prefixes else "plain-v1",
        }

    def check_collection(self, collection):
        """Never query or append vectors from a different embedding space."""
        if collection.count() == 0:
            return
        stored = collection.metadata or {}
        if any(stored.get(key) != value for key, value in self.metadata.items()):
            raise ValueError(
                "Embedding configuration differs from this index (or it predates provider metadata). "
                "Choose a new collection_name and run build_rag_index.py to rebuild; "
                "do not mix old and new embeddings."
            )

    def _embed(self, texts, prefix):
        inputs = [prefix + text for text in texts] if self.e5_prefixes else texts
        if self.provider == "local":
            return _local_model().encode(
                inputs,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
        response = self.client.embeddings.create(model=self.model_name, input=inputs)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def embed_documents(self, texts):
        return self._embed(texts, "passage: ")

    def embed_query(self, text):
        return self._embed([text], "query: ")[0]
