import pytest
from unittest.mock import MagicMock, patch
from haystack import Pipeline
import pandas as pd

@pytest.fixture
def mock_pipeline():
    """Create a mocked pipeline for testing."""
    pipeline = MagicMock(spec=Pipeline)
    
    # Mock pipeline.to_dict() to return a structure similar to a real pipeline
    pipeline.to_dict.return_value = {
        "components": {
            "retriever": {
                "type": "haystack.components.retrievers.InMemoryBM25Retriever"
            },
            "llm": {
                "type": "haystack.components.generators.huggingface.HuggingFaceLocalGenerator"
            },
            "answer_builder": {
                "type": "custom.component"
            }
        },
        "connections": [
            {"sender": "retriever.documents", "receiver": "llm.documents"},
            {"sender": "llm.replies", "receiver": "answer_builder.answer"}
        ]
    }
    
    # Mock pipeline.run() to return a predefined output
    pipeline.run.return_value = {
        "retriever": {
            "documents": [
                {"content": "Document 1 content", "score": 0.95},
                {"content": "Document 2 content", "score": 0.85}
            ]
        },
        "llm": {
            "replies": ["This is a generated answer."]
        },
        "answer_builder": {
            "answer": "This is a final answer."
        }
    }
    
    return pipeline

@pytest.fixture
def sample_evaluation_data():
    """Sample evaluation data with test cases."""
    return {
        "test_cases": [
            {
                "input": "What is RAG?",
                "expected_output": "valid"
            },
            {
                "input": "How does retrieval work?",
                "expected_output": "valid"
            },
            {
                "input": "Unrelated question?",
                "expected_output": "invalid"
            },
            {
                "input": "Another unrelated question?",
                "expected_output": "invalid"
            }
        ]
    }

@pytest.fixture
def mock_metrics_registry():
    """Mock the MetricRegistry with predefined metrics."""
    with patch("ragnroll.metrics.MetricRegistry") as mock_registry:
        # Create mock E2E metrics with proper mock instances
        mock_accuracy = MagicMock()
        mock_accuracy_instance = MagicMock()
        mock_accuracy_instance.run.return_value = {"score": 0.75, "details": {}}
        mock_accuracy.return_value = mock_accuracy_instance
        
        mock_precision = MagicMock()
        mock_precision_instance = MagicMock()
        mock_precision_instance.run.return_value = {"score": 0.8, "details": {}}
        mock_precision.return_value = mock_precision_instance
        
        # Do the same for all other metrics
        mock_recall = MagicMock()
        mock_recall_instance = MagicMock()
        mock_recall_instance.run.return_value = {"score": 0.7, "details": {}}
        mock_recall.return_value = mock_recall_instance
        
        # Component metrics with proper mock instances
        mock_map = MagicMock()
        mock_map_instance = MagicMock()
        mock_map_instance.run.return_value = {"score": 0.9, "details": {}}
        mock_map.return_value = mock_map_instance
        
        mock_relevance = MagicMock()
        mock_relevance_instance = MagicMock()
        mock_relevance_instance.run.return_value = {"score": 0.85, "details": {}}
        mock_relevance.return_value = mock_relevance_instance
        
        # Configure the mock registry
        mock_registry.get_end_to_end_metrics.return_value = {
            "AccuracyMetric": mock_accuracy,
            "PrecisionMetric": mock_precision,
            "RecallMetric": mock_recall
        }
        
        mock_registry.get_component_metrics.return_value = {
            "retriever": {
                "MAPMetric": mock_map
            },
            "generator": {
                "RelevanceMetric": mock_relevance
            }
        }
        
        yield mock_registry

@pytest.fixture
def processed_test_cases():
    """Sample processed test cases after running through the pipeline."""
    return [
        {
            "input": "What is RAG?",
            "expected_output": "valid",
            "actual_output": "valid",
            "component_outputs": {
                "retriever": {"documents": [{"content": "Doc1", "score": 0.9}]},
                "llm": {"replies": ["RAG is retrieval augmented generation."]},
                "answer_builder": {"answer": "valid"}
            }
        },
        {
            "input": "How does retrieval work?",
            "expected_output": "valid",
            "actual_output": "valid",
            "component_outputs": {
                "retriever": {"documents": [{"content": "Doc2", "score": 0.85}]},
                "llm": {"replies": ["Retrieval works by finding relevant documents."]},
                "answer_builder": {"answer": "valid"}
            }
        },
        {
            "input": "Unrelated question?",
            "expected_output": "invalid",
            "actual_output": "valid",  # Incorrect prediction
            "component_outputs": {
                "retriever": {"documents": [{"content": "Doc3", "score": 0.5}]},
                "llm": {"replies": ["This might be related."]},
                "answer_builder": {"answer": "valid"}
            }
        },
        {
            "input": "Another unrelated question?",
            "expected_output": "invalid",
            "actual_output": "invalid",
            "component_outputs": {
                "retriever": {"documents": []},
                "llm": {"replies": ["I don't have information about this."]},
                "answer_builder": {"answer": "invalid"}
            }
        }
    ] 