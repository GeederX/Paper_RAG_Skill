"""Prevent same-dimension vectors from different embedding spaces being mixed."""

import sys
import tempfile
import unittest
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/paper-rag/assets/PaperRAG"))
from embeddings import EmbeddingBackend, LOCAL_MODEL


class EmbeddingCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.client = chromadb.PersistentClient(path=self.directory.name)
        self.backend = EmbeddingBackend({})
        self.collection = self.client.create_collection(
            "papers", metadata=self.backend.metadata, embedding_function=None
        )
        self.collection.add(ids=["paper_0"], embeddings=[[1.0, 0.0]], documents=["Paper"])

    def test_matching_embedding_space_allows_incremental_use(self):
        self.backend.check_collection(self.collection)
        self.collection.add(ids=["paper_1"], embeddings=[[0.0, 1.0]], documents=["Other"])
        self.assertEqual(self.collection.count(), 2)

    def test_different_embedding_spaces_rejected_even_with_same_dimensions(self):
        for key, value in (
            ("embedding_provider", "http"),
            ("embedding_model", "another-384-dimensional-model"),
            ("embedding_base_url", "https://other.example/v1"),
            ("embedding_format", "plain-v1"),
        ):
            with self.subTest(field=key):
                self.collection.modify(metadata={**self.backend.metadata, key: value})
                with self.assertRaisesRegex(ValueError, "new collection_name"):
                    self.backend.check_collection(self.collection)
                self.assertEqual(self.collection.count(), 1)

    def test_legacy_index_requires_rebuild(self):
        self.collection.modify(metadata={"embedding_model": LOCAL_MODEL})
        with self.assertRaisesRegex(ValueError, "rebuild"):
            self.backend.check_collection(self.collection)

    def test_empty_collection_can_be_rebuilt_with_new_embeddings(self):
        self.collection.delete(ids=["paper_0"])
        self.collection.modify(metadata={"embedding_model": "old-model"})
        self.backend.check_collection(self.collection)
        self.collection.modify(metadata=self.backend.metadata)
        self.collection.add(ids=["new_0"], embeddings=[[0.0, 1.0]], documents=["New"])
        self.backend.check_collection(self.collection)
        self.assertEqual(self.collection.get()["ids"], ["new_0"])


if __name__ == "__main__":
    unittest.main()
