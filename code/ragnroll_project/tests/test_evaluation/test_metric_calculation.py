import pytest
from haystack.utils import Secret
from ragnroll.metrics import (
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1ScoreMetric,
    MatthewsCorrCoefMetric,
    FalsePositiveRateMetric,
    FalseNegativeRateMetric,
)
from ragnroll.metrics.component.retriever import MAPAtKMetric
from ragnroll.metrics.component.generator import FormatValidatorMetric

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

def test_mcc_calculation(simple_classification_data):
    """Test ob MatthewsCorrCoefMetric korrekt berechnet wird."""
    expected, actual = simple_classification_data
    metric = MatthewsCorrCoefMetric(positive_label="valid", negative_label="invalid")
    result = metric.run(expected_outputs=expected, actual_outputs=actual)
    
    # TP=2, TN=1, FP=1, FN=1
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    # MCC = (2*1 - 1*1) / sqrt((2+1)*(2+1)*(1+1)*(1+1))
    # MCC = (2 - 1) / sqrt(3 * 3 * 2 * 2) = 1 / sqrt(36) = 1 / 6 ≈ 0.167
    assert abs(result["score"] - 1/6) < 0.001, f"MCC sollte ca. 0.167 sein, ist aber {result['score']}"

def test_fpr_calculation(simple_classification_data):
    """Test ob FalsePositiveRateMetric korrekt berechnet wird."""
    expected, actual = simple_classification_data
    metric = FalsePositiveRateMetric(positive_label="valid", negative_label="invalid")
    result = metric.run(expected_outputs=expected, actual_outputs=actual)
    
    # TP=2, TN=1, FP=1, FN=1
    # FPR = FP / (FP + TN) = 1 / (1 + 1) = 0.5
    assert abs(result["score"] - 0.5) < 0.001, f"FPR sollte 0.5 sein, ist aber {result['score']}"

def test_fnr_calculation(simple_classification_data):
    """Test ob FalseNegativeRateMetric korrekt berechnet wird."""
    expected, actual = simple_classification_data
    metric = FalseNegativeRateMetric(positive_label="valid", negative_label="invalid")
    result = metric.run(expected_outputs=expected, actual_outputs=actual)
    
    # TP=2, TN=1, FP=1, FN=1
    # FNR = FN / (FN + TP) = 1 / (1 + 2) = 1/3 ≈ 0.333
    assert abs(result["score"] - 1/3) < 0.001, f"FNR sollte ca. 0.333 sein, ist aber {result['score']}"

def test_map_at_k_average_precision_calculation():
    """Testet die _calculate_average_precision Methode von MAPAtKMetric."""
    map_metric = MAPAtKMetric(k=5, api_key=Secret.from_token("dummy"))
    
    # Beispiel 1: Perfektes Ranking
    relevance1 = [1, 1, 1, 0, 0]
    # Precision@1 = 1/1 = 1
    # Precision@2 = 2/2 = 1
    # Precision@3 = 3/3 = 1
    # AP = (1 + 1 + 1) / 3 = 1.0
    assert abs(map_metric._calculate_average_precision(relevance1) - 1.0) < 0.001

    # Beispiel 2: Gemischtes Ranking
    relevance2 = [1, 0, 1, 0, 1]
    # Precision@1 = 1/1 = 1
    # Precision@3 = 2/3 ≈ 0.667
    # Precision@5 = 3/5 = 0.6
    # AP = (1 + 0.667 + 0.6) / 3 ≈ 0.756
    assert abs(map_metric._calculate_average_precision(relevance2) - ((1 + 2/3 + 3/5) / 3)) < 0.001

    # Beispiel 3: Keine relevanten Dokumente
    relevance3 = [0, 0, 0, 0, 0]
    assert abs(map_metric._calculate_average_precision(relevance3) - 0.0) < 0.001

    # Beispiel 4: Relevante Dokumente außerhalb von K
    map_metric_k3 = MAPAtKMetric(k=3)
    relevance4 = [0, 0, 1, 1, 1]
    # Precision@3 = 1/3
    # AP = (1/3) / 1 = 1/3
    assert abs(map_metric_k3._calculate_average_precision(relevance4) - 1/3) < 0.001

def test_format_validator_check_format():
    """Testet die _check_format Methode von FormatValidatorMetric."""
    validator = FormatValidatorMetric(
        positive_pattern=r'The answer is \"(Valid|Yes)\".',
        negative_pattern=r'The answer is \"(Invalid|No)\".',
        case_sensitive=False
    )
    
    # Positive Fälle
    assert validator._check_format('Some text... The answer is "Valid". More text.') == True
    assert validator._check_format('the answer is \"yes\".') == True
    
    # Negative Fälle
    assert validator._check_format('The answer is \"Invalid\".') == True
    assert validator._check_format('answer is \"no\".') == False
    
    # The default pattern requires "The answer is \"...". Let's use a default validator for this case.
    default_validator = FormatValidatorMetric(case_sensitive=False)
    assert default_validator._check_format('The answer is \"Invalid\".') == True
    assert default_validator._check_format('the answer is \"no\".') == True
    assert default_validator._check_format('answer is \"no\".') == False
    
    # Falsche Formate
    assert default_validator._check_format('The answer is Valid.') == False
    assert default_validator._check_format('Answer: Valid') == False
    assert default_validator._check_format('Yes') == False
    assert default_validator._check_format('invalid') == False
    
    # Beide Patterns gefunden (inkonsistent)
    assert default_validator._check_format('The answer is \"Valid\". The answer is \"Invalid\".') == False
    
    # Case sensitivity check
    validator_cs = FormatValidatorMetric(
        positive_pattern=r'The answer is \"Valid\".',
        negative_pattern=r'The answer is \"Invalid\".',
        case_sensitive=True
    )
    assert validator_cs._check_format('The answer is \"Valid\".') == True
    assert validator_cs._check_format('The answer is \"valid\".') == False
    assert validator_cs._check_format('The answer is \"Invalid\".') == True
    assert validator_cs._check_format('The answer is \"invalid\".') == False 