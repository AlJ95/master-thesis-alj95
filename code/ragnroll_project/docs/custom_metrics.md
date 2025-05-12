# Define Custom Metrics

In RagnRoll, you can define your own metrics in the `ragnroll_project/metrics` folder. Three different registration options are available:

## 1. End-to-End Metrics

End-to-End metrics evaluate the overall performance of the pipeline. Use the `@MetricRegistry.register_end_to_end` decorator:

```python
from ragnroll.metrics.base import BaseMetric, MetricRegistry

@MetricRegistry.register_end_to_end
class MyCustomEndToEndMetric(BaseMetric):
    def run(self, expected_outputs: List[str], actual_outputs: List[str], **kwargs) -> Dict[str, Any]:
        # Your metric logic here
        score = compute_score(expected_outputs, actual_outputs)
        return {
            "score": score,
            "success": score >= self.threshold,
            "details": {
                # Additional details
            }
        }
```

## 2. Component-Specific Metrics

### For Retriever
Metrics for the Retriever component. Use `@MetricRegistry.register_component_metric("retriever")`:

```python
@MetricRegistry.register_component_metric("retriever")
class MyCustomRetrieverMetric(BaseMetric):
    def run(self, component_outputs: List[Dict[str, Any]], queries: List[str], **kwargs) -> Dict[str, Any]:
        # Your retriever metric logic here
        return {
            "score": computed_score,
            "success": computed_score >= self.threshold,
            "details": {}
        }
```

### For Generator
Metrics for the Generator component. Use `@MetricRegistry.register_component_metric("generator")`:

```python
@MetricRegistry.register_component_metric("generator")
class MyCustomGeneratorMetric(BaseMetric):
    def run(self, component_outputs: List[Dict[str, Any]], expected_outputs: List[str], **kwargs) -> Dict[str, Any]:
        # Your generator metric logic here
        return {
            "score": computed_score,
            "success": computed_score >= self.threshold,
            "details": {}
        }
```

## Important Notes

1.  **Base Class**: All metrics must inherit from `BaseMetric`
2.  **run() Method**: Implement the abstract `run()` method
3.  **Return Format**: The `run()` method must return a dictionary with at least:
    *   `score`: float between 0 and 1
    *   `success`: bool based on the threshold
    *   `details`: Dict with additional information (optional)

4.  **Registration**: Metrics are automatically registered upon import
5.  **Threshold**: The threshold can be overridden in the constructor:
    ```python
    metric = MyCustomMetric(threshold=0.7)  # Default is 0.5
    ```

## Example of a Complete Metric

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

        # Calculate exact matches
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