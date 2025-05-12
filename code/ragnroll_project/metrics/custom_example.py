from ragnroll.metrics.base import MetricRegistry
from ragnroll.metrics.end2end import ClassificationBaseMetric
from typing import Dict, Any, List


@MetricRegistry.register_end_to_end
class ExactMatchMetric(ClassificationBaseMetric):
    """Custom example metric for binary classification tasks."""
    
    def run(self, expected_outputs: List[str], actual_outputs: List[str]) -> Dict[str, Any]:
        """
        Args:
            expected_outputs: List of ground truth labels
            actual_outputs: List of predicted labels

        Returns:
            Dict with accuracy score and success flag
        """
        try:
            # Transform the expected and actual outputs to binary values
            y_true, y_pred = self._process_predictions(expected_outputs, actual_outputs)

            # HERE YOUR METRIC LOGIC
            self.score = sum([int(label == prediction) for label, prediction in zip(y_true, y_pred)]) / len(y_true)


            self.success = self.score >= self.threshold
            
            return {
                "score": self.score,
                "success": self.success,
                "details": {
                    "num_examples": len(expected_outputs),
                }
            }
        
        except Exception as e:
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }
