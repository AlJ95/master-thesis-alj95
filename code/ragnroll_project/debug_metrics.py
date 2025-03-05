import json
import logging
from ragnroll.metrics import (
    AccuracyMetric, PrecisionMetric, RecallMetric, F1ScoreMetric, 
    MatthewsCorrCoefMetric
)
from ragnroll.evaluation.eval import Evaluator, DEFAULT_POSITIVE_LABEL, DEFAULT_NEGATIVE_LABEL

# Konfiguriere Logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('metric_debug')

# Lade die Testdaten
with open('data/dev_data/synthetic_rag_evaluation.json', 'r') as f:
    test_data = json.load(f)

# Extrahiere expected_outputs und simuliere actual_outputs (als wären sie korrekt vorhergesagt)
expected_outputs = [case["expected_output"] for case in test_data["test_cases"]]
actual_outputs = expected_outputs.copy()  # Perfekte Vorhersagen für den Test

# Überprüfe die Standard-Labels
logger.info(f"Standard-Labels: positiv='{DEFAULT_POSITIVE_LABEL}', negativ='{DEFAULT_NEGATIVE_LABEL}'")

# Überprüfe die Verteilung der Labels
valid_count = expected_outputs.count("valid")
invalid_count = expected_outputs.count("invalid")
logger.info(f"Label-Verteilung: valid={valid_count}, invalid={invalid_count}")

# Teste mit den Standard-Labels
logger.info(f"----- Mit Standard-Labels (positive='{DEFAULT_POSITIVE_LABEL}', negative='{DEFAULT_NEGATIVE_LABEL}') -----")
metrics = [
    AccuracyMetric(positive_label=DEFAULT_POSITIVE_LABEL, negative_label=DEFAULT_NEGATIVE_LABEL),
    PrecisionMetric(positive_label=DEFAULT_POSITIVE_LABEL, negative_label=DEFAULT_NEGATIVE_LABEL),
    RecallMetric(positive_label=DEFAULT_POSITIVE_LABEL, negative_label=DEFAULT_NEGATIVE_LABEL),
    F1ScoreMetric(positive_label=DEFAULT_POSITIVE_LABEL, negative_label=DEFAULT_NEGATIVE_LABEL),
    MatthewsCorrCoefMetric(positive_label=DEFAULT_POSITIVE_LABEL, negative_label=DEFAULT_NEGATIVE_LABEL),
]

for metric in metrics:
    result = metric.run(expected_outputs, actual_outputs)
    
    # Debug-Informationen ausgeben
    y_true, y_pred = metric._process_predictions(expected_outputs, actual_outputs)
    cm = metric.get_confusion_matrix(y_true, y_pred)
    
    logger.info(f"{metric.__class__.__name__}: score={result['score']:.4f}")
    logger.info(f"Binäre Werte - y_true: {y_true}")
    logger.info(f"Binäre Werte - y_pred: {y_pred}")
    logger.info(f"Konfusionsmatrix: TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")
    logger.info("-" * 80)

print("Debug-Informationen wurden im Log ausgegeben. Überprüfen Sie die Konsolenausgabe.") 