import pytest
from ragnroll.evaluation.eval import EvaluationDataset
from unittest.mock import MagicMock, Mock

@pytest.fixture
def sample_data():
    """Sample test data for Docker script validation."""
    return {
        "test_cases": [
            {
                "input": "FROM python:3.9\nRUN pip install flask\nCMD ['python', 'app.py']",
                "expected_output": "valid"
            },
            {
                "input": "FRUM python:3.9\nRUN pip instal flask\nCMD python app.py",  # Contains typos and incorrect syntax
                "expected_output": "invalid"
            }
        ]
    }

@pytest.fixture
def mock_pipeline():
    """Mock pipeline for tests."""
    pipeline = MagicMock()
    
    # Simulate pipeline response
    mock_answer = Mock()
    mock_answer.data = "invalid"
    pipeline.run.return_value = {
        "retriever": {
            "documents": [{"content": "Dockerfile validation rules and syntax", "score": 0.95}]
        },
        "llm": {
            "replies": ["The Dockerfile contains syntax errors: 'FRUM' is not a valid instruction, should be 'FROM'. Also 'instal' is misspelled."]
        },
        "answer_builder": {
            "answers": [mock_answer]
        }
    }
    
    return pipeline

def test_initialization(sample_data):
    """Test correct initialization of EvaluationDataset class."""
    dataset = EvaluationDataset(sample_data)
    
    # Check if data was stored correctly
    assert dataset.evaluation_data == sample_data
    assert dataset.processed_data == []

def test_generate_predictions(sample_data, mock_pipeline):
    """Test prediction generation."""
    dataset = EvaluationDataset(sample_data)
    
    # Generate predictions
    dataset.generate_predictions(mock_pipeline)
    
    # Get processed data
    processed_data = dataset.get_processed_data()
    
    # Check if predictions were generated for all test cases
    assert len(processed_data) == len(sample_data["test_cases"])
    
    # Check if pipeline was called for each test case
    assert mock_pipeline.run.call_count == len(sample_data["test_cases"])
    
    # Check structure of generated data
    for test_case in processed_data:
        assert "input" in test_case
        assert "expected_output" in test_case
        assert "actual_output" in test_case
        assert "component_outputs" in test_case
        
        # actual_output should be extracted from pipeline response
        assert test_case["actual_output"] == "invalid"

def test_extract_answer_from_pipeline():
    """Test answer extraction from pipeline response."""
    dataset = EvaluationDataset({"test_cases": []})
    
    # With answer_builder
    mock_answer1 = Mock()
    mock_answer1.data = "Test Answer 1"
    response1 = {
        "answer_builder": {
            "answers": [mock_answer1]
        }
    }
    assert dataset._extract_answer_from_pipeline(response1) == "Test Answer 1"
    
    # Without valid component should raise exception
    response3 = {
        "retriever": {
            "documents": []
        }
    }
    with pytest.raises(ValueError):
        dataset._extract_answer_from_pipeline(response3)

def test_error_handling(sample_data):
    """Test error handling during prediction generation."""
    dataset = EvaluationDataset(sample_data)
    
    # Mock pipeline that throws an exception
    failing_pipeline = MagicMock()
    failing_pipeline.run.side_effect = Exception("Pipeline error")
    
    # Should handle errors gracefully without throwing
    dataset.generate_predictions(failing_pipeline)
    
    # Processed data should be empty
    processed_data = dataset.get_processed_data()
    assert processed_data == []