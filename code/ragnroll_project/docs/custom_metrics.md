# Eigene Metriken definieren

In RagnRoll können Sie eigene Metriken in der Datei `custom_metrics.py` definieren. Dabei stehen Ihnen drei verschiedene Registrierungsmöglichkeiten zur Verfügung:

## 1. End-to-End Metriken

End-to-End Metriken evaluieren die Gesamtleistung der Pipeline. Verwenden Sie den `@MetricRegistry.register_end_to_end` Decorator:

```python
from ragnroll.metrics.base import BaseMetric, MetricRegistry

@MetricRegistry.register_end_to_end
class MyCustomEndToEndMetric(BaseMetric):
    def run(self, expected_outputs: List[str], actual_outputs: List[str], **kwargs) -> Dict[str, Any]:
        # Ihre Metrik-Logik hier
        score = compute_score(expected_outputs, actual_outputs)
        return {
            "score": score,
            "success": score >= self.threshold,
            "details": {
                # Zusätzliche Details
            }
        }
```

## 2. Komponenten-spezifische Metriken

### Für Retriever
Metriken für die Retriever-Komponente. Verwenden Sie `@MetricRegistry.register_component_metric("retriever")`:

```python
@MetricRegistry.register_component_metric("retriever")
class MyCustomRetrieverMetric(BaseMetric):
    def run(self, component_outputs: List[Dict[str, Any]], queries: List[str], **kwargs) -> Dict[str, Any]:
        # Ihre Retriever-Metrik-Logik hier
        return {
            "score": computed_score,
            "success": computed_score >= self.threshold,
            "details": {}
        }
```

### Für Generator
Metriken für die Generator-Komponente. Verwenden Sie `@MetricRegistry.register_component_metric("generator")`:

```python
@MetricRegistry.register_component_metric("generator")
class MyCustomGeneratorMetric(BaseMetric):
    def run(self, component_outputs: List[Dict[str, Any]], expected_outputs: List[str], **kwargs) -> Dict[str, Any]:
        # Ihre Generator-Metrik-Logik hier
        return {
            "score": computed_score,
            "success": computed_score >= self.threshold,
            "details": {}
        }
```

## Wichtige Hinweise

1. **Basisklasse**: Alle Metriken müssen von `BaseMetric` erben
2. **run() Methode**: Implementieren Sie die abstrakte `run()` Methode
3. **Rückgabeformat**: Die `run()` Methode muss ein Dictionary mit mindestens:
   - `score`: float zwischen 0 und 1
   - `success`: bool basierend auf dem threshold
   - `details`: Dict mit zusätzlichen Informationen (optional)

4. **Registrierung**: Die Metriken werden automatisch beim Import registriert
5. **Threshold**: Der Schwellenwert kann im Konstruktor überschrieben werden:
   ```python
   metric = MyCustomMetric(threshold=0.7)  # Standard ist 0.5
   ```

## Beispiel einer vollständigen Metrik

```python
from typing import Dict, Any, List
from ragnroll.metrics.base import BaseMetric, MetricRegistry

@MetricRegistry.register_end_to_end
class CustomAccuracyMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold=threshold)
        
    def run(self, expected_outputs: List[str], actual_outputs: List[str], **kwargs) -> Dict[str, Any]:
        if len(expected_outputs) != len(actual_outputs):
            raise ValueError("Length of expected and actual outputs must match")
            
        # Berechne exakte Übereinstimmungen
        correct = sum(1 for e, a in zip(expected_outputs, actual_outputs) if e == a)
        total = len(expected_outputs)
        score = correct / total if total > 0 else 0.0
        
        return {
            "score": score,
            "success": score >= self.threshold,
            "details": {
                "correct_matches": correct,
                "total_samples": total
            }
        }
```

## Verwendung

Nach der Definition in `custom_metrics.py` können Sie Ihre Metriken wie die eingebauten Metriken verwenden:

```python
from ragnroll.metrics.custom_metrics import CustomAccuracyMetric

metric = CustomAccuracyMetric(threshold=0.8)
result = metric.run(expected_outputs=["a", "b"], actual_outputs=["a", "c"])
print(f"Score: {result['score']}, Success: {result['success']}")
```
