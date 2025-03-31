import pytest
from ragnroll.evaluation.eval import Evaluator, EvaluationDataset
from unittest.mock import patch, MagicMock

def test_evaluator_initialization(mock_pipeline):
    """Test that Evaluator initializes correctly."""
    evaluator = Evaluator(mock_pipeline)
    
    # Check default labels
    assert evaluator.positive_label == "valid"
    assert evaluator.negative_label == "invalid"
    
    # Check metrics initialization
    assert len(evaluator.end_to_end_metrics) > 0
    assert len(evaluator.component_metrics) > 0

def test_evaluation_dataset_initialization(sample_evaluation_data):
    """Test that EvaluationDataset initializes correctly."""
    dataset = EvaluationDataset(sample_evaluation_data)
    
    # Check that data is stored correctly
    assert dataset.evaluation_data == sample_evaluation_data
    assert dataset.processed_data == []


@patch("ragnroll.evaluation.eval.EvaluationDataset")
@patch("ragnroll.evaluation.eval.Evaluator")
@patch("ragnroll.evaluation.eval.print_scores")
def test_evaluate_function(mock_print, mock_evaluator_class, mock_dataset_class, 
                           mock_pipeline, sample_evaluation_data):
    """Test the main evaluate function."""
    # Setup
    mock_dataset = MagicMock()
    mock_dataset_class.return_value = mock_dataset
    mock_dataset.get_processed_data.return_value = []
    
    mock_evaluator = MagicMock()
    mock_evaluator_class.return_value = mock_evaluator
    mock_evaluator._results_to_df.return_value = MagicMock()
    
    # Execute
    evaluator = Evaluator(mock_pipeline)
    result = evaluator.evaluate(sample_evaluation_data)
    
    # Verify
    mock_dataset_class.assert_called_once_with(sample_evaluation_data)
    mock_dataset.generate_predictions.assert_called_once_with(mock_pipeline)
    assert "dataframe" in result
    assert "trace_ids" in result
    assert "metrics" in result
    assert "component_metrics" in result
    mock_print.assert_called_once() 