import pytest
from ragnroll.evaluation.eval import Evaluator, EvaluationDataset
from unittest.mock import patch, MagicMock
from haystack import Pipeline

@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock(spec=Pipeline)
    pipeline.to_dict.return_value = {
        "components": {
            "retriever": {"type": "haystack_integrations.components.retrievers.chroma.ChromaEmbeddingRetriever"},
            "llm": {"type": "haystack.components.generators.openai.GPTGenerator"}
        },
        "connections": [
            {
                "sender": "retriever.documents",
                "receiver": "llm.documents"
            }
        ]
    }
    return pipeline

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
    mock_dataset.get_trace_ids.return_value = []

    mock_evaluator = MagicMock()
    mock_evaluator_class.return_value = mock_evaluator

    with patch.object(Evaluator, '_evaluate_end_to_end', return_value={"accuracy": 1.0}) as mock_e2e, \
         patch.object(Evaluator, '_evaluate_components', return_value={"generator": {}, "retriever": {}}) as mock_comp, \
         patch.object(Evaluator, '_results_to_df') as mock_results_to_df:

        mock_pipeline.to_dict.return_value = {
            "components": {
                "retriever": {"type": "haystack_integrations.components.retrievers.chroma.ChromaEmbeddingRetriever"},
                "llm": {"type": "haystack.components.generators.openai.GPTGenerator"}
            },
            "connections": [
                {
                    "sender": "retriever.documents",
                    "receiver": "llm.documents"
                }
            ]
        }
        
        evaluator = Evaluator(mock_pipeline)
        result = evaluator.evaluate(sample_evaluation_data, run_name="test_run")
        
        mock_dataset_class.assert_called_once_with(sample_evaluation_data)
        mock_dataset.generate_predictions.assert_called_once_with(evaluator.pipeline)
        mock_e2e.assert_called_once()
        mock_comp.assert_called_once()
        mock_print.assert_called_once()
        mock_results_to_df.assert_called_once() 