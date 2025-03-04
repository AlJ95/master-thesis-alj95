#!/usr/bin/env python
"""
Example script demonstrating how to use the refactored metrics and evaluation system.
"""

import json
import os
from haystack import Pipeline
from ragnroll.metrics import (
    BaseMetric, 
    MetricRegistry, 
    ExactMatchMetric,
    RetrievalPrecisionMetric, 
    RetrievalRecallMetric,
    AVAILABLE_METRICS
)
from ragnroll.evaluation import evaluate, print_scores, Evaluator

def main():
    # Print available metrics
    print("Available end-to-end metrics:")
    for name in AVAILABLE_METRICS["end-to-end"]:
        print(f"  - {name}")
    
    print("\nAvailable component metrics:")
    for component, metrics in AVAILABLE_METRICS["component"].items():
        print(f"  {component}:")
        for name in metrics:
            print(f"    - {name}")
    
    # Load the example evaluation data with ground truth
    eval_data_path = os.path.join("examples", "evaluation_data_with_ground_truth.json")
    try:
        with open(eval_data_path, "r") as f:
            evaluation_data = json.load(f)
            print(f"\nLoaded evaluation data from {eval_data_path}")
    except FileNotFoundError:
        # Fall back to inline data if file not found
        print(f"\nFile {eval_data_path} not found, using inline example data")
        evaluation_data = {
            "test_cases": [
                {
                    "input": "What is the capital of France?",
                    "expected_output": "The capital of France is Paris.",
                    "ground_truth_docs": [
                        {
                            "id": "doc1",
                            "content": "Paris is the capital of France."
                        }
                    ]
                },
                {
                    "input": "Who wrote Romeo and Juliet?",
                    "expected_output": "William Shakespeare wrote Romeo and Juliet.",
                    "ground_truth_docs": [
                        {
                            "id": "doc2",
                            "content": "Shakespeare wrote Romeo and Juliet."
                        }
                    ]
                }
            ]
        }
    
    # Example of directly testing the retrieval metrics without a full pipeline
    print("\nDemonstrating direct usage of retrieval metrics:")
    
    # Create test data
    retrieved_docs = [
        {"id": "doc1", "content": "Paris is the capital of France."},
        {"id": "doc3", "content": "Berlin is the capital of Germany."}
    ]
    
    ground_truth_docs = [
        {"id": "doc1", "content": "Paris is the capital of France."},
        {"id": "doc2", "content": "France is in Europe."}
    ]
    
    # Test precision
    precision_metric = RetrievalPrecisionMetric()
    result = precision_metric.run(
        component_output={"documents": retrieved_docs},
        ground_truth_docs=ground_truth_docs
    )
    
    print(f"Precision: {result['score']:.2f}")
    print(f"Details: {result['details']}")
    
    # Test recall
    recall_metric = RetrievalRecallMetric()
    result = recall_metric.run(
        component_output={"documents": retrieved_docs},
        ground_truth_docs=ground_truth_docs
    )
    
    print(f"Recall: {result['score']:.2f}")
    print(f"Details: {result['details']}")
    
    # In a real scenario, you would use an actual pipeline
    print("\nIn a real scenario, you would run:")
    print("  pipeline = create_rag_pipeline()")
    print("  scores, component_metrics = evaluate(evaluation_data, pipeline)")
    print("  print_scores(scores)")

if __name__ == "__main__":
    main() 