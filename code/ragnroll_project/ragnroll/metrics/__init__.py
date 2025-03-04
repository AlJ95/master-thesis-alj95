"""
Metrics package for evaluating RAG pipelines.

This package provides metrics for evaluating both end-to-end performance
and component-specific performance of RAG pipelines.
"""

from ragnroll.metrics.base import BaseMetric, MetricRegistry
from ragnroll.metrics.end2end import ExactMatchMetric

# Import component metrics
from ragnroll.metrics.component.retriever import RetrievalPrecisionMetric, RetrievalRecallMetric

# Create a convenience dictionary of all available metrics
AVAILABLE_METRICS = {
    "end-to-end": MetricRegistry.get_end_to_end_metrics(),
    "component": MetricRegistry.get_component_metrics()
}

__all__ = [
    "BaseMetric", 
    "MetricRegistry",
    "ExactMatchMetric",
    "RetrievalPrecisionMetric",
    "RetrievalRecallMetric",
    "AVAILABLE_METRICS"
]
