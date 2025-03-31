import pytest
from ragnroll.metrics import (
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1ScoreMetric
)

# Einfache Testfälle für die Klassifikation
# 0=invalid, 1=valid (in unseren Tests)
@pytest.fixture
def simple_classification_data():
    """Einfache Klassifikationsdaten mit bekannten erwarteten Ergebnissen."""
    expected = ["valid", "valid", "valid", "invalid", "invalid"]
    actual = ["valid", "valid", "invalid", "invalid", "valid"]
    return expected, actual

def test_accuracy_calculation(simple_classification_data):
    """Test ob AccuracyMetric korrekt berechnet wird."""
    expected, actual = simple_classification_data
    
    # Metrik mit den richtigen Labels für valid/invalid initialisieren
    metric = AccuracyMetric(positive_label="valid", negative_label="invalid")
    
    # Berechnung ausführen
    result = metric.run(expected_outputs=expected, actual_outputs=actual)
    
    # 3 von 5 Vorhersagen sind richtig: (TP=2, TN=1, FP=1, FN=1)
    # Accuracy = (TP + TN) / (TP + TN + FP + FN) = (2 + 1) / 5 = 0.6
    assert abs(result["score"] - 0.6) < 0.001, f"Accuracy sollte 0.6 sein, ist aber {result['score']}"

def test_precision_calculation(simple_classification_data):
    """Test ob PrecisionMetric korrekt berechnet wird."""
    expected, actual = simple_classification_data
    
    # Metrik mit den richtigen Labels für valid/invalid initialisieren
    metric = PrecisionMetric(positive_label="valid", negative_label="invalid")
    
    # Berechnung ausführen
    result = metric.run(expected_outputs=expected, actual_outputs=actual)
    
    # Precision = TP / (TP + FP) = 2 / (2 + 1) = 2/3 ≈ 0.67
    assert abs(result["score"] - 0.67) < 0.01, f"Precision sollte ca. 0.67 sein, ist aber {result['score']}"

def test_recall_calculation(simple_classification_data):
    """Test ob RecallMetric korrekt berechnet wird."""
    expected, actual = simple_classification_data
    
    # Metrik mit den richtigen Labels für valid/invalid initialisieren
    metric = RecallMetric(positive_label="valid", negative_label="invalid")
    
    # Berechnung ausführen
    result = metric.run(expected_outputs=expected, actual_outputs=actual)
    
    # Recall = TP / (TP + FN) = 2 / (2 + 1) = 2/3 ≈ 0.67
    assert abs(result["score"] - 0.67) < 0.01, f"Recall sollte ca. 0.67 sein, ist aber {result['score']}"

def test_f1_calculation(simple_classification_data):
    """Test ob F1ScoreMetric korrekt berechnet wird."""
    expected, actual = simple_classification_data
    
    # Metrik mit den richtigen Labels für valid/invalid initialisieren
    metric = F1ScoreMetric(positive_label="valid", negative_label="invalid")
    
    # Berechnung ausführen
    result = metric.run(expected_outputs=expected, actual_outputs=actual)
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    # F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2 * (4/9) / (4/3) = 2 * 4/9 * 3/4 = 2/3 ≈ 0.67
    assert abs(result["score"] - 0.67) < 0.01, f"F1-Score sollte ca. 0.67 sein, ist aber {result['score']}"

def test_confusion_matrix_values(simple_classification_data):
    """Test ob die Elemente der Konfusionsmatrix korrekt berechnet werden."""
    expected, actual = simple_classification_data
    
    # Metrik mit den richtigen Labels für valid/invalid initialisieren
    metric = AccuracyMetric(positive_label="valid", negative_label="invalid")
    
    # Berechnung ausführen
    result = metric.run(expected_outputs=expected, actual_outputs=actual)
    
    # Confusion Matrix prüfen:
    # TP (true positive): 2
    # TN (true negative): 1
    # FP (false positive): 1
    # FN (false negative): 1
    cm = result["details"]["confusion_matrix"]
    assert cm["TP"] == 2, f"TP sollte 2, ist aber {cm['TP']}"
    assert cm["TN"] == 1, f"TN sollte 1, ist aber {cm['TN']}"
    assert cm["FP"] == 1, f"FP sollte 1, ist aber {cm['FP']}"
    assert cm["FN"] == 1, f"FN sollte 1, ist aber {cm['FN']}" 