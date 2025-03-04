from typing import Dict, Any
from ragnroll.metrics.base import BaseMetric, MetricRegistry

@MetricRegistry.register_end_to_end
class ExactMatchMetric(BaseMetric):
    """Exact match metric that checks if the actual output exactly matches the expected output."""
    
    def __init__(self, threshold: float = 1.0):
        super().__init__(threshold=threshold)
    
    def run(self, expected_output: str, actual_output: str) -> Dict[str, Any]:
        """Run the exact match metric."""
        try:
            self.score = float(expected_output == actual_output)
            self.success = self.score >= self.threshold
            self.details = {
                "expected_output": expected_output,
                "actual_output": actual_output
            }
            return {
                "score": self.score,
                "success": self.success,
                "details": self.details
            }
        except Exception as e:
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }

# Example of additional end-to-end metrics:

# @MetricRegistry.register_end_to_end
# class F1ScoreMetric(BaseMetric):
#     """F1 score metric for evaluating classification tasks."""
#     
#     def __init__(self, threshold: float = 0.7):
#         super().__init__(threshold=threshold)
#     
#     def run(self, expected_output: str, actual_output: str) -> Dict[str, Any]:
#         # Implementation of F1 score calculation would go here
#         pass

# @MetricRegistry.register_end_to_end
# class RougeMetric(BaseMetric):
#     """ROUGE metric for evaluating summarization tasks."""
#     
#     def __init__(self, rouge_type: str = "rouge-l", threshold: float = 0.5):
#         super().__init__(threshold=threshold)
#         self.rouge_type = rouge_type
#     
#     def run(self, expected_output: str, actual_output: str) -> Dict[str, Any]:
#         # Implementation of ROUGE metric would go here
#         pass

# @MetricRegistry.register_end_to_end
# class BLEUMetric(BaseMetric):
#     """BLEU metric for evaluating translation or text generation tasks."""
#     
#     def __init__(self, threshold: float = 0.3):
#         super().__init__(threshold=threshold)
#     
#     def run(self, expected_output: str, actual_output: str) -> Dict[str, Any]:
#         # Implementation of BLEU metric would go here
#         pass
