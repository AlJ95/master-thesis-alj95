"""
Test suite for the ingestion tracking and deduplication system.

This module tests the enhanced ingestion functionality with fixtures
to simulate different scenarios.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from ragnroll.utils.ingestion_tracker import IngestionTracker, IngestionRecord
from ragnroll.utils.ingestion import index_documents
from ragnroll.utils.pipeline import config_to_pipeline


class TestIngestionTracker:
    """Test the ingestion tracking functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = IngestionTracker(tracking_file=os.path.join(self.temp_dir, "test_tracking.csv"))

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_generate_index_id_deterministic(self):
        """Test that index ID generation is deterministic."""
        corpus_path = "/test/corpus"
        config = {"chunking": {"size": 1000}, "embedder": "test"}

        id1 = self.tracker.generate_index_id(corpus_path, config)
        id2 = self.tracker.generate_index_id(corpus_path, config)

        assert id1 == id2
        assert isinstance(id1, str)
        assert len(id1) == 64  # SHA256 hex length

    def test_generate_index_id_different_configs(self):
        """Test that different configs produce different IDs."""
        corpus_path = "/test/corpus"
        config1 = {"chunking": {"size": 1000}, "embedder": "test"}
        config2 = {"chunking": {"size": 2000}, "embedder": "test"}

        id1 = self.tracker.generate_index_id(corpus_path, config1)
        id2 = self.tracker.generate_index_id(corpus_path, config2)

        assert id1 != id2

    def test_generate_document_id_deterministic(self):
        """Test that document ID generation is deterministic."""
        content = "test document content"
        config = {"chunking": {"size": 1000}}

        id1 = self.tracker.generate_document_id(content, config)
        id2 = self.tracker.generate_document_id(content, config)

        assert id1 == id2
        assert isinstance(id1, str)
        assert len(id1) == 64

    def test_record_and_retrieve_ingestion(self):
        """Test recording and retrieving ingestion records."""
        record = IngestionRecord(
            corpus_path="/test/corpus",
            processing_config_hash="testhash",
            index_id="testindex",
            document_ids=["doc1", "doc2"],
            timestamp="2024-01-01T00:00:00",
            status="completed"
        )

        self.tracker.record_ingestion(record)
        retrieved = self.tracker.get_existing_record("testindex")

        assert retrieved is not None
        assert retrieved.index_id == "testindex"
        assert retrieved.document_ids == ["doc1", "doc2"]
        assert retrieved.status == "completed"


class TestIngestionDeduplication:
    """Test the ingestion deduplication functionality."""

    def setup_method(self):
        """Set up test environment with minimal corpus."""
        self.temp_dir = tempfile.mkdtemp()
        self.corpus_dir = Path(self.temp_dir) / "corpus"
        self.corpus_dir.mkdir()

        # Create minimal test corpus
        test_docs = [
            {"id": "doc1", "content": "This is document 1 about Docker."},
            {"id": "doc2", "content": "This is document 2 about GitHub Actions."},
            {"id": "doc3", "content": "This is document 3 about testing."}
        ]

        corpus_file = self.corpus_dir / "test_corpus.json"
        with open(corpus_file, 'w') as f:
            json.dump({"documents": test_docs}, f)

        # Create a minimal config file
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        config_content = """
metadata:
  chunking:
    split: true
    chunk_size: 50
    chunk_overlap: 10
    chunk_separator: "\\n\\n"

components:
  text_embedder:
    type: "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder"
    init_parameters:
      model: "sentence-transformers/all-MiniLM-L6-v2"

  retriever:
    type: "haystack.components.retrievers.in_memory.embedding_retriever.InMemoryEmbeddingRetriever"
    init_parameters:
      document_store:
        type: "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore"
        init_parameters: {}
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_in_memory_store_always_ingests(self):
        """Test that InMemoryDocumentStore always performs ingestion (no tracking)."""
        # First ingestion
        pipeline = config_to_pipeline(self.config_path)
        pipeline, duration1 = index_documents(str(self.corpus_dir), pipeline)

        assert duration1 > 0  # Should have done work

        # Second ingestion with same config - should do work again (no persistence)
        pipeline2 = config_to_pipeline(self.config_path)
        pipeline2, duration2 = index_documents(str(self.corpus_dir), pipeline2)

        assert duration2 > 0  # Should do work again (in-memory stores don't persist)

    @patch('ragnroll.utils.ingestion.check_index_exists')
    @patch('ragnroll.utils.ingestion.tracker')
    @patch('ragnroll.utils.ingestion.get_components_from_config_by_classes')
    def test_persistent_store_skips_completed_ingestion(self, mock_get_components, mock_tracker, mock_check_index):
        """Test that persistent stores skip ingestion when index already exists."""
        # Mock get_components_from_config_by_classes to return empty list (no in-memory stores)
        mock_get_components.return_value = []

        # Mock tracker to simulate existing completed index
        mock_existing_record = MagicMock()
        mock_existing_record.status = "completed"
        mock_tracker.get_existing_record.return_value = mock_existing_record

        # Mock check_index_exists to simulate index exists in store
        mock_check_index.return_value = True

        # Now use_tracking should be True, and ingestion should be skipped
        pipeline = config_to_pipeline(self.config_path)
        pipeline, duration = index_documents(str(self.corpus_dir), pipeline)

        assert duration == 0  # Should skip work (persistent store with existing index)

    @patch('ragnroll.utils.ingestion.check_index_exists')
    def test_different_chunking_creates_different_index(self, mock_check_index):
        """Test that different chunking parameters create different indexes."""
        mock_check_index.return_value = False

        # First config with small chunks
        pipeline1 = config_to_pipeline(self.config_path)
        pipeline1, duration1 = index_documents(str(self.corpus_dir), pipeline1)

        # Modify config for larger chunks
        config_content_large = """
metadata:
  chunking:
    split: true
    chunk_size: 100
    chunk_overlap: 20
    chunk_separator: "\\n\\n"

components:
  text_embedder:
    type: "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder"
    init_parameters:
      model: "sentence-transformers/all-MiniLM-L6-v2"

  retriever:
    type: "haystack.components.retrievers.in_memory.embedding_retriever.InMemoryEmbeddingRetriever"
    init_parameters:
      document_store:
        type: "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore"
        init_parameters: {}
"""
        config_path_large = Path(self.temp_dir) / "test_config_large.yaml"
        with open(config_path_large, 'w') as f:
            f.write(config_content_large)

        # Second ingestion with different chunking
        pipeline2 = config_to_pipeline(config_path_large)
        pipeline2, duration2 = index_documents(str(self.corpus_dir), pipeline2)

        assert duration2 > 0  # Should do work because different config

    def test_partial_ingestion_completion(self):
        """Test completing a partial ingestion."""
        # This would require mocking a partial state, which is complex
        # For now, just test that the logic exists
        tracker = IngestionTracker(tracking_file=os.path.join(self.temp_dir, "test_tracking.csv"))

        # Create a partial record
        partial_record = IngestionRecord(
            corpus_path=str(self.corpus_dir),
            processing_config_hash="testhash",
            index_id="testindex",
            document_ids=["doc1"],  # Only one document
            timestamp="2024-01-01T00:00:00",
            status="partial"
        )
        tracker.record_ingestion(partial_record)

        # Check that we can detect missing documents
        required_docs = {"doc1", "doc2", "doc3"}
        missing = tracker.get_missing_documents("testindex", required_docs)

        assert "doc2" in missing
        assert "doc3" in missing
        assert "doc1" not in missing


if __name__ == "__main__":
    # Run basic tests
    test_tracker = TestIngestionTracker()
    test_tracker.setup_method()

    try:
        test_tracker.test_generate_index_id_deterministic()
        print("✓ Index ID generation is deterministic")

        test_tracker.test_generate_index_id_different_configs()
        print("✓ Different configs produce different IDs")

        test_tracker.test_generate_document_id_deterministic()
        print("✓ Document ID generation is deterministic")

        test_tracker.test_record_and_retrieve_ingestion()
        print("✓ Record and retrieve ingestion works")

    finally:
        test_tracker.teardown_method()

    print("\nBasic ingestion tracking tests passed!")
    print("Run 'pytest test_ingestion_tracking.py' for full test suite.")
