"""
Evaluation package for RAG pipelines.

This package provides tools for evaluating both end-to-end performance
and component-specific performance of RAG pipelines.
"""

from ragnroll.evaluation.eval import evaluate, print_scores, Evaluator, EvaluationDataset

__all__ = [
    "evaluate",
    "print_scores",
    "Evaluator", 
    "EvaluationDataset"
]
