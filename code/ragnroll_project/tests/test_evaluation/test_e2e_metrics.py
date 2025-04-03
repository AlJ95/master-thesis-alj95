import pytest
from ragnroll.evaluation.eval import Evaluator
from unittest.mock import patch, MagicMock

def test_all_e2e_metrics_registered(mock_metrics_registry, mock_pipeline):
    """Test that all registered end-to-end metrics are instantiated."""
    with patch("ragnroll.metrics.MetricRegistry", mock_metrics_registry):
        evaluator = Evaluator(mock_pipeline)
        
        # Verify all metrics from registry are in the evaluator
        expected_metrics = mock_metrics_registry.get_end_to_end_metrics.return_value.keys()
        for metric_name in expected_metrics:
            assert metric_name in evaluator.end_to_end_metrics

def test_e2e_metrics_executed(mock_pipeline, processed_test_cases):
    """Test that all end-to-end metrics are executed during evaluation."""
    # Evaluator erstellen
    evaluator = Evaluator(mock_pipeline)
    
    # Alle originalen Metriken durch Mocks ersetzen
    mock_metrics = {}
    for name in evaluator.end_to_end_metrics.keys():
        mock = MagicMock()
        mock.run.return_value = {"score": 0.75}
        mock_metrics[name] = mock
    
    # Ersetzen Sie die echten Metriken durch Mock-Objekte
    evaluator.end_to_end_metrics = mock_metrics
    
    # Execute
    results = evaluator._evaluate_end_to_end(processed_test_cases, trace_ids=[])
    
    # Verify
    for metric_name, mock_metric in mock_metrics.items():
        assert metric_name in results
        # Jetzt wird die .called Eigenschaft statt assert_called_once() verwendet
        assert mock_metric.run.called, f"Metric {metric_name} was not called"

def test_e2e_metrics_receive_correct_data(mock_pipeline, processed_test_cases):
    """Test that end-to-end metrics receive the correct data."""
    # Create a real evaluator with mocked metrics
    evaluator = Evaluator(mock_pipeline)
    
    # Replace metrics with mocks
    mock_metric = MagicMock()
    mock_metric.run.return_value = {"score": 0.5}
    evaluator.end_to_end_metrics = {"MockMetric": mock_metric}
    
    # Execute
    evaluator._evaluate_end_to_end(processed_test_cases, trace_ids=[])
    
    # Verify that the metric received the correct data
    expected_outputs = [tc["expected_output"] for tc in processed_test_cases]
    actual_outputs = [tc["actual_output"] for tc in processed_test_cases]
    
    assert mock_metric.run.called, "Metric was not called"
    args, kwargs = mock_metric.run.call_args
    assert "expected_outputs" in kwargs
    assert "actual_outputs" in kwargs
    assert kwargs["expected_outputs"] == expected_outputs
    assert kwargs["actual_outputs"] == actual_outputs

def test_e2e_metrics_error_handling(mock_pipeline, processed_test_cases):
    """Test that errors in individual metrics don't crash the evaluation."""
    # Create evaluator with a mix of working and failing metrics
    evaluator = Evaluator(mock_pipeline)
    
    working_metric = MagicMock()
    working_metric.run.return_value = {"score": 0.5}
    
    failing_metric = MagicMock()
    failing_metric.run.side_effect = Exception("Metric failure")
    
    evaluator.end_to_end_metrics = {
        "WorkingMetric": working_metric,
        "FailingMetric": failing_metric
    }
    
    # Execute
    results = evaluator._evaluate_end_to_end(processed_test_cases, trace_ids=[])
    
    # Verify
    assert "WorkingMetric" in results
    assert results["WorkingMetric"] == 0.5
    assert "FailingMetric" in results
    assert results["FailingMetric"] == 0.0  # Default for failed metrics
