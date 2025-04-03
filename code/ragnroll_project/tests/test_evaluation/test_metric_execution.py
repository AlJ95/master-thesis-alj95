import pytest
from unittest.mock import patch, MagicMock, call, Mock
from ragnroll.evaluation.eval import Evaluator
from haystack import Pipeline

@pytest.fixture
def mock_pipeline():
    """Mock-Pipeline für Tests."""
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
    # Define mock_answer before using it
    mock_answer = Mock()
    mock_answer.data = "valid"
    pipeline.run.return_value = {
        "retriever": {"documents": [{"content": "Test Document", "score": 0.9}]},
        "llm": {"replies": ["Test Reply"]},
        "answer_builder": {"answers": [mock_answer]}
    }
    return pipeline

@pytest.fixture
def test_cases():
    """Einfache Testfälle."""
    return [
        {
            "input": "Test Question?",
            "expected_output": "valid",
            "actual_output": "valid",
            "component_outputs": {
                "retriever": {"documents": [{"content": "Test Document", "score": 0.9}]},
                "llm": {"replies": ["Test Reply"]},
                "answer_builder": {"answer": "valid"}
            }
        }
    ]

def test_e2e_metrics_execution(mock_pipeline, test_cases):
    """Test ob alle E2E-Metriken während der Evaluierung aufgerufen werden."""
    # Evaluator mit gemockten Metriken erstellen
    evaluator = Evaluator(mock_pipeline)
    
    # Originale Metriken sichern
    original_metrics = evaluator.end_to_end_metrics.copy()
    
    try:
        # Alle Metriken durch Mocks ersetzen
        mock_metrics = {}
        for name in original_metrics.keys():
            mock = MagicMock()
            mock.run.return_value = {"score": 0.5}
            mock_metrics[name] = mock
        
        evaluator.end_to_end_metrics = mock_metrics
        
        # Evaluierung durchführen
        evaluator._evaluate_end_to_end(test_cases, trace_ids=[])
        
        # Prüfen ob jede Metrik aufgerufen wurde
        for name, mock_metric in mock_metrics.items():
            assert mock_metric.run.called, f"E2E-Metrik '{name}' wurde nicht aufgerufen"
            
            # Prüfen ob die richtigen Parameter übergeben wurden
            args, kwargs = mock_metric.run.call_args
            assert "expected_outputs" in kwargs
            assert "actual_outputs" in kwargs
            assert len(kwargs["expected_outputs"]) == len(test_cases)
            assert len(kwargs["actual_outputs"]) == len(test_cases)
    
    finally:
        # Originale Metriken wiederherstellen
        evaluator.end_to_end_metrics = original_metrics

def test_component_metrics_execution(mock_pipeline, test_cases):
    """Test ob alle Komponenten-Metriken während der Evaluierung aufgerufen werden."""
    # Evaluator erstellen
    evaluator = Evaluator(mock_pipeline)
    
    # Originale Metriken sichern
    original_metrics = evaluator.component_metrics.copy()
    
    try:
        # Komponenten-Metriken durch Mocks ersetzen
        mock_component_metrics = {}
        for component_type, metrics in original_metrics.items():
            mock_component_metrics[component_type] = {}
            for name in metrics.keys():
                mock = MagicMock()
                mock.run.return_value = {"score": 0.5}
                mock_component_metrics[component_type][name] = mock
        
        evaluator.component_metrics = mock_component_metrics
        
        # Patch the utility function that causes the IndexError
        with patch("ragnroll.evaluation.eval.get_last_component_with_documents") as mock_get_last_comp:
            mock_get_last_comp.return_value = "retriever" # Assume retriever provides docs
            
            # Evaluierung durchführen
            evaluator._evaluate_components(test_cases)
            
            # Prüfen ob relevante Metriken aufgerufen wurden
            # (nicht alle Komponententypen müssen in den Testdaten vorkommen)
            mock_get_last_comp.assert_called_with(mock_pipeline, "llm") # Verify it was called for the generator
            
            for component_type, metrics in mock_component_metrics.items():
                if component_type in ["retriever", "generator"]:
                    for name, mock_metric in metrics.items():
                        assert mock_metric.run.called, f"Komponenten-Metrik '{component_type}.{name}' wurde nicht aufgerufen"
    
    finally:
        # Originale Metriken wiederherstellen
        evaluator.component_metrics = original_metrics

@patch("ragnroll.evaluation.eval.EvaluationDataset")
def test_evaluate_function_calls_all_metrics(mock_dataset_class, mock_pipeline):
    """Test ob die evaluate-Funktion alle Metriken aufruft."""
    # Testdaten vorbereiten
    test_data = {"test_cases": [{"input": "Test?", "expected_output": "valid"}]}
    
    # Mock-Dataset erstellen
    mock_dataset = MagicMock()
    mock_dataset_class.return_value = mock_dataset
    mock_dataset.get_processed_data.return_value = [
        {
            "input": "Test?",
            "expected_output": "valid",
            "actual_output": "valid",
            "component_outputs": {
                "retriever": {"documents": [{"content": "Test", "score": 0.9}]},
                "llm": {"replies": ["Test"]}
            }
        }
    ]
    
    mock_dataset.get_trace_ids.return_value = []
    
    # In der evaluate-Funktion werden die Metriken direkt in einer Schleife aufgerufen
    # Wir müssen daher die run-Methode jeder Metrik mocken
    with patch("ragnroll.metrics.AccuracyMetric.run") as mock_accuracy_run, \
         patch("ragnroll.evaluation.eval.Evaluator._evaluate_components") as mock_comp, \
         patch("ragnroll.evaluation.eval.print_scores"), \
         patch("ragnroll.evaluation.eval.Evaluator._results_to_df"):
        
        # Mock-Rückgabewerte setzen
        mock_accuracy_run.return_value = {"score": 0.75}
        mock_comp.return_value = {"generator": {"llm": {"some_metric": 0.5}}, "retriever": {"retriever": {"some_metric": 0.6}}}
        
        # evaluate Funktion aufrufen
        evaluator = Evaluator(mock_pipeline)
        evaluator.evaluate(test_data, run_name="test_run")
        
        # Prüfen ob die Metrik-run-Methode aufgerufen wurde
        assert mock_accuracy_run.called, "AccuracyMetric.run wurde nicht aufgerufen"
        assert mock_comp.called, "_evaluate_components wurde nicht aufgerufen" 