import pytest
from unittest.mock import patch, MagicMock
from ragnroll.evaluation.eval import Evaluator
from ragnroll.metrics import MetricRegistry, AVAILABLE_METRICS
from haystack import Pipeline

def test_all_e2e_metrics_registered():
    """Test ob alle E2E-Metriken korrekt registriert werden."""
    # Prüfen Sie, ob alle erwarteten E2E-Metriken aus __init__.py registriert sind
    expected_e2e_metrics = [
        "AccuracyMetric", 
        "PrecisionMetric", 
        "RecallMetric", 
        "F1ScoreMetric",
        "MatthewsCorrCoefMetric",
        "FalsePositiveRateMetric",
        "FalseNegativeRateMetric",
    ]
    
    registered_metrics = MetricRegistry.get_end_to_end_metrics()
    
    for metric_name in expected_e2e_metrics:
        assert metric_name in registered_metrics, f"{metric_name} fehlt in den registrierten E2E-Metriken"

def test_all_component_metrics_registered():
    """Test ob alle Komponenten-Metriken korrekt registriert werden."""
    expected_component_metrics = {
        "retriever": ["HaystackContextRelevanceMetric", "MAPAtKMetric"],
        "generator": ["FormatValidatorMetric", "ContextUtilizationMetric", "AnswerRelevancyMetric"]
    }
    
    registered_metrics = MetricRegistry.get_component_metrics()
    
    # Prüfen Sie, ob alle erwarteten Komponententypen vorhanden sind
    for component_type in expected_component_metrics:
        assert component_type in registered_metrics, f"Komponententyp {component_type} fehlt in den registrierten Metriken"
        
        # Prüfen Sie, ob alle erwarteten Metriken für diesen Komponententyp vorhanden sind
        for metric_name in expected_component_metrics[component_type]:
            assert metric_name in registered_metrics[component_type], f"{metric_name} fehlt in den {component_type}-Metriken"

def test_evaluator_instantiates_all_metrics():
    """Test ob der Evaluator alle registrierten Metriken initialisiert."""
    # Mock-Pipeline erstellen
    pipeline = MagicMock(spec=Pipeline)
    pipeline.to_dict.return_value = {
        "components": {
            "retriever": {"type": ".retrievers."},
            "generator": {"type": ".generators."}
        }
    }
    
    # Evaluator erstellen
    evaluator = Evaluator(pipeline)
    
    # Prüfen Sie, ob alle in der Registry registrierten E2E-Metriken vom Evaluator initialisiert wurden
    registered_e2e_metrics = MetricRegistry.get_end_to_end_metrics()
    for metric_name in registered_e2e_metrics:
        assert metric_name in evaluator.end_to_end_metrics
    
    # Prüfen Sie, ob alle relevanten Komponenten-Metriken initialisiert wurden
    registered_component_metrics = MetricRegistry.get_component_metrics()
    for component_type, metrics in registered_component_metrics.items():
        if component_type in ["retriever", "generator"]:
            assert component_type in evaluator.component_metrics
            for metric_name in metrics:
                assert metric_name in evaluator.component_metrics[component_type]

def test_metrics_are_called_with_correct_label_values():
    """Test ob Metriken mit den korrekten Label-Werten initialisiert werden."""
    # Mock-Pipeline erstellen
    pipeline = MagicMock(spec=Pipeline)
    pipeline.to_dict.return_value = {"components": {}}
    
    # Evaluator mit benutzerdefinierten Labels erstellen
    custom_pos_label = "correct"
    custom_neg_label = "incorrect"
    evaluator = Evaluator(pipeline, positive_label=custom_pos_label, negative_label=custom_neg_label)
    
    # Prüfen Sie, ob die Labels korrekt gesetzt wurden
    assert evaluator.positive_label == custom_pos_label
    assert evaluator.negative_label == custom_neg_label
    
    # Prüfen Sie ob die ClassificationBaseMetric-basierten Metriken die richtigen Labels haben
    for metric_name, metric in evaluator.end_to_end_metrics.items():
        # Ignorieren Sie Metriken, die nicht von ClassificationBaseMetric erben
        if hasattr(metric, 'positive_label') and hasattr(metric, 'negative_label'):
            assert metric.positive_label == custom_pos_label
            assert metric.negative_label == custom_neg_label 