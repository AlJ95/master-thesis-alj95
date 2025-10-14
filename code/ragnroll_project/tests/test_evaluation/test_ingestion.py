import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from haystack import AsyncPipeline
from ragnroll.utils.ingestion import index_documents
from ragnroll.utils.pipeline import config_to_pipeline


class TestIngestion:
    """Tests for document ingestion functionality."""

    def test_index_documents_with_bm25_retriever(self):
        """Test that index_documents works with BM25 retriever (like predefined.yaml)."""
        # Load the predefined config that has BM25 retriever
        config_path = Path(__file__).parent.parent.parent / "configs" / "examples" / "predefined.yaml"
        pipeline = config_to_pipeline(config_path)

        # Create temporary corpus directory with test documents
        with tempfile.TemporaryDirectory() as corpus_dir:
            corpus_file = Path(corpus_dir) / "test_corpus.json"
            # Create individual text files for documents
            doc1 = Path(corpus_dir) / "doc1.txt"
            doc1.write_text("This is a test document for BM25 indexing.")

            doc2 = Path(corpus_dir) / "doc2.txt"
            doc2.write_text("Another document to ensure multiple docs are handled.")

            # Mock the ingestion tracker to avoid file system issues
            with patch('ragnroll.utils.ingestion.tracker') as mock_tracker:
                mock_tracker.generate_index_id.return_value = "test_index_123"
                mock_tracker.get_existing_record.return_value = None
                mock_tracker.generate_document_id.side_effect = lambda content, config: f"doc_{hash(content)}"

                # Index documents
                indexed_pipeline, duration = index_documents(corpus_dir, pipeline)

                # Verify that indexing was attempted (not skipped)
                assert duration > 0, "Indexing should have taken some time"

                # Verify document store is connected to retriever
                retriever = indexed_pipeline.get_component("retriever")
                assert retriever.document_store is not None, "Retriever should have document store"

                # Verify documents were written
                doc_count = retriever.document_store.count_documents()
                assert doc_count > 0, f"Expected documents in store, got {doc_count}"

    def test_index_documents_without_retriever(self):
        """Test that index_documents skips when no retriever is present."""
        # Create minimal pipeline without retriever - use a simple mock pipeline
        pipeline = MagicMock(spec=AsyncPipeline)
        pipeline.to_dict.return_value = {
            "components": {
                "llm": {"type": "haystack.components.generators.openai.OpenAIGenerator"}
            }
        }

        with tempfile.TemporaryDirectory() as corpus_dir:
            corpus_file = Path(corpus_dir) / "test_corpus.json"
            corpus_file.write_text(json.dumps([{"content": "Test document"}]))

            # Index documents - should skip and return original pipeline
            indexed_pipeline, duration = index_documents(corpus_dir, pipeline)

            # Verify no indexing occurred
            assert duration == 0, "Should skip indexing when no retriever present"
            assert indexed_pipeline == pipeline, "Pipeline should be unchanged"

    def test_index_documents_with_existing_index(self):
        """Test that index_documents skips when index already exists."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "examples" / "predefined.yaml"
        pipeline = config_to_pipeline(config_path)

        with tempfile.TemporaryDirectory() as corpus_dir:
            corpus_file = Path(corpus_dir) / "test_corpus.json"
            corpus_file.write_text(json.dumps([{"content": "Test document"}]))

            # Mock tracker to simulate existing completed index
            with patch('ragnroll.utils.ingestion.tracker') as mock_tracker:
                mock_tracker.generate_index_id.return_value = "existing_index_123"
                mock_tracker.get_existing_record.return_value = MagicMock(status="completed")

                # Index documents - should skip
                indexed_pipeline, duration = index_documents(corpus_dir, pipeline)

                # Verify skipping occurred
                assert duration == 0, "Should skip when index already exists"

                # Verify document store is still connected
                retriever = indexed_pipeline.get_component("retriever")
                assert retriever.document_store is not None, "Document store should be connected even when skipping"

    def test_retriever_detection_logic(self):
        """Test the retriever detection logic in index_documents."""
        from ragnroll.utils.ingestion import BM25Retriever, EmbeddingRetriever, SentenceWindowRetriever, HybridRetriever

        # Test BM25 retriever detection - use individual retriever names, not joined
        config_with_bm25 = {
            "components": {
                "retriever": {
                    "type": "haystack.components.retrievers.in_memory.bm25_retriever.InMemoryBM25Retriever",
                    "init_parameters": {"document_store": {"type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore"}}
                }
            }
        }

        # Should find BM25 retriever using individual retriever name
        from ragnroll.utils.config import get_components_from_config_by_classes
        bm25_found = get_components_from_config_by_classes(config_with_bm25, "InMemoryBM25Retriever")
        assert bm25_found, "Should detect BM25 retriever"

        # Test config without retriever
        config_without_retriever = {
            "components": {
                "llm": {"type": "haystack.components.generators.openai.OpenAIGenerator"}
            }
        }

        bm25_not_found = get_components_from_config_by_classes(config_without_retriever, "InMemoryBM25Retriever")
        embedding_not_found = get_components_from_config_by_classes(config_without_retriever, "InMemoryEmbeddingRetriever")
        assert not bm25_not_found, "Should not find BM25 retriever"
        assert not embedding_not_found, "Should not find embedding retriever"
