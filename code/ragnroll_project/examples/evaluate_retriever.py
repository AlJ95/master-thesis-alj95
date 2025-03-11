#!/usr/bin/env python
"""
Example script for evaluating retriever components using the
HaystackContextRelevanceMetric and RagasContextPrecisionMetric metrics.

This script demonstrates how to:
1. Evaluate retriever outputs with both metrics
2. Use different model providers and configurations
3. Compare results between different evaluations
"""

import os
from typing import List, Dict, Any
from haystack import Document
from ragnroll.metrics.component.retriever import (
    HaystackContextRelevanceMetric,
    RagasContextPrecisionMetric
)

# Sample data for demonstration
sample_query = "What are the key features of Haystack?"
sample_documents = [
    Document(
        content="""Haystack is an open-source framework for building search systems that work with large document collections.
        Key features include: modular design with swappable components, support for various embedding models,
        multiple document stores including Elasticsearch and vector databases, retrieval methods like BM25 and 
        dense retrievers, and integration with LLMs for question answering."""
    ),
    Document(
        content="""This is an unrelated document about gardening. Plants need sunlight, water, and nutrients to grow.
        Different plants have different requirements for these resources. Some plants need full sunlight, 
        while others prefer shade. Soil quality is also important for plant health."""
    )
]

def main():
    # You can set API keys in environment variables or pass them directly
    # For OpenAI: OPENAI_API_KEY
    
    print("Evaluating retriever with different metrics and configurations...")
    
    # Sample component output for evaluation
    component_output = {
        "documents": sample_documents
    }
    
    # Create a list of metrics to evaluate
    metrics = [
        {
            "name": "Haystack Context Relevance",
            "metric": HaystackContextRelevanceMetric(
                threshold=0.7,
                model="gpt-4o-mini"
            )
        },
        {
            "name": "Haystack Context Relevance with Examples",
            "metric": HaystackContextRelevanceMetric(
                threshold=0.7,
                model="gpt-4o-mini",
                examples=[{
                    "inputs": {
                        "questions": "What is a database?", 
                        "contexts": ["A database is an organized collection of data stored and accessed electronically."]
                    },
                    "outputs": {
                        "statements": ["A database is an organized collection of data.", "Data in databases is stored and accessed electronically."],
                        "statement_scores": [1, 1]
                    }
                }]
            )
        },
        {
            "name": "RAGAS Context Precision (GPT-4o-mini)",
            "metric": RagasContextPrecisionMetric(
                threshold=0.7,
                model_name="gpt-4o-mini"
            )
        },
        {
            "name": "RAGAS Context Precision (GPT-3.5-turbo)",
            "metric": RagasContextPrecisionMetric(
                threshold=0.7,
                model_name="gpt-3.5-turbo"
            )
        }
    ]
    
    # Run evaluations
    results = []
    for metric_info in metrics:
        try:
            print(f"\nRunning evaluation with {metric_info['name']}...")
            result = metric_info["metric"].run(
                component_outputs=[component_output],
                queries=[sample_query]
            )
            
            results.append({
                "name": metric_info["name"],
                "score": result["score"],
                "details": result
            })
            
            print(f"Score: {result['score']:.4f}")
            print(f"Success: {result['score'] >= metric_info['metric'].threshold}")
            
            # Print additional details if available
            if "results" in result and result["results"]:
                for i, query_result in enumerate(result["results"]):
                    if "statements" in query_result and "statement_scores" in query_result:
                        print(f"Query {i+1} Statements:")
                        for stmt, score in zip(query_result["statements"], query_result["statement_scores"]):
                            print(f"  - {stmt} (Relevant: {'Yes' if score else 'No'})")
            
        except Exception as e:
            print(f"Error evaluating with {metric_info['name']}: {e}")
    
    # Compare results
    print("\n=== Comparison of Results ===")
    for result in results:
        print(f"{result['name']}: {result['score']:.4f}")

if __name__ == "__main__":
    main() 