from typing import Dict, Any, List, Type, Optional, Callable, Union
from abc import ABC, abstractmethod
import inspect
import logging

logger = logging.getLogger(__name__)

class MetricRegistry:
    """Registry for all available metrics."""
    
    _end_to_end_metrics: Dict[str, Type["BaseMetric"]] = {}
    _component_metrics: Dict[str, Dict[str, Type["BaseMetric"]]] = {}
    
    @classmethod
    def register_end_to_end(cls, metric_class: Type["BaseMetric"]) -> Type["BaseMetric"]:
        """Register an end-to-end metric."""
        cls._end_to_end_metrics[metric_class.__name__] = metric_class
        return metric_class
    
    @classmethod
    def register_component_metric(cls, component_type: str) -> Callable:
        """Register a component-specific metric.
        
        Args:
            component_type: The type of component to get metrics for (e.g. "retriever", "generator")
        """
        def decorator(metric_class: Type["BaseMetric"]) -> Type["BaseMetric"]:
            if component_type not in cls._component_metrics:
                cls._component_metrics[component_type] = {}
            cls._component_metrics[component_type][metric_class.__name__] = metric_class
            return metric_class
        return decorator
    
    @classmethod
    def get_end_to_end_metrics(cls) -> Dict[str, Type["BaseMetric"]]:
        """Get all registered end-to-end metrics."""
        return cls._end_to_end_metrics
    
    @classmethod
    def get_component_metrics(cls, component_type: Optional[str] = None) -> Dict[str, Dict[str, Type["BaseMetric"]]]:
        """Get all registered component metrics or metrics for a specific component.
        
        Args:
            component_type: The type of component to get metrics for (e.g. "retriever", "generator")
            
        Returns:
            Dict[str, Dict[str, Type["BaseMetric"]]]: A dictionary of metrics for the given component type
        """
        if component_type:
            return {component_type: cls._component_metrics.get(component_type, {})}
        return cls._component_metrics


class BaseMetric(ABC):
    """Base class for all metrics."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the base metric.
        
        Args:
            threshold: Threshold for determining success
        """
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.error = None
        self.details = {}

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the metric evaluation.
        
        Args:
            **kwargs: Metric-specific inputs
            
        Returns:
            Dict[str, Any]: Results with score, success flag, and details
        """
        pass
    
    def measure(self, **kwargs) -> float:
        """
        Measure the metric (Backward compatibility).
        
        Args:
            **kwargs: Metric-specific inputs
            
        Returns:
            float: Score between 0 and 1
        """
        results = self.run(**kwargs)
        return results["score"]
    
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
        """Check if the metric is successful based on threshold."""
        if self.error is not None:
            self.success = False
        return self.success
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.__class__.__name__,
            "score": self.score,
            "threshold": self.threshold,
            "success": self.success,
            "error": str(self.error) if self.error else None,
            "details": self.details
        }
