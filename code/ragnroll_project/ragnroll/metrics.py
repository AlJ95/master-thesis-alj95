from typing import List, Dict, Any, Optional, Union, Tuple
from haystack import component
from haystack.utils import Secret
from haystack.components.evaluators import ContextRelevanceEvaluator

class BaseMetric:
    """
    Base class for all metrics.
    """
    def __init__(self, name: str, threshold: float = 0.5):
        """
        Initialize the base metric.
        
        Args:
            name: Name of the metric
            threshold: Threshold for determining success
        """
        self.name = name
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.error = None
        self.details = {}

    def measure(self, **kwargs) -> float:
        """
        Measure the metric.
        
        Args:
            **kwargs: Metric-specific inputs
            
        Returns:
            float: Score between 0 and 1
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def a_measure(self, **kwargs) -> float:
        """
        Asynchronous implementation of measure().
        
        Args:
            **kwargs: Metric-specific inputs
            
        Returns:
            float: Score between 0 and 1
        """
        return self.measure(**kwargs)
    
    def is_successful(self) -> bool:
        """
        Check if the metric is successful based on threshold.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.error is not None:
            self.success = False
        return self.success
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metric to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "name": self.name,
            "score": self.score,
            "threshold": self.threshold,
            "success": self.success,
            "error": str(self.error) if self.error else None,
            "details": self.details
        }


class ExactMatchMetric(BaseMetric):
    """
    Exact match metric that checks if the actual output exactly matches the expected output.
    """
    def __init__(self, threshold: float = 1.0):
        """
        Initialize the exact match metric.
        
        Args:
            threshold: Threshold for determining success (default is 1.0 for exact match)
        """
        super().__init__(name="exact_match", threshold=threshold)
    
    def run(self, expected_output: str, actual_output: str) -> Dict[str, Any]:
        """
        Run the exact match metric.
        
        Args:
            expected_output: Expected output
            actual_output: Actual output
            
        Returns:
            Dict[str, Any]: Results with score, success flag, and details
        """
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
    
    def measure(self, expected_output: str, actual_output: str) -> float:
        """
        Measure the exact match between expected and actual outputs.
        
        Args:
            expected_output: Expected output
            actual_output: Actual output
            
        Returns:
            float: 1.0 if exact match, 0.0 otherwise
        """
        results = self.run(expected_output=expected_output, actual_output=actual_output)
        return results["score"]
