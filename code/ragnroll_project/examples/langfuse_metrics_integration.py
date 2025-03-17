#!/usr/bin/env python3
"""
Example script demonstrating how to use Langfuse metrics integration.

This script shows how to:
1. Run a simple evaluation
2. Report metrics to Langfuse
3. View the metrics in the Langfuse UI

Make sure your environment variables for Langfuse are properly set:
- LANGFUSE_HOST (optional, defaults to https://cloud.langfuse.com)
- LANGFUSE_SECRET_KEY
- LANGFUSE_PUBLIC_KEY
"""

import os
import sys
import uuid
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Make sure environment variables are set
if not os.environ.get("LANGFUSE_SECRET_KEY") or not os.environ.get("LANGFUSE_PUBLIC_KEY"):
    print("Please set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY environment variables")
    print("You can find these in your Langfuse dashboard under Settings > API Keys")
    sys.exit(1)

# Import from ragnroll
from ragnroll.utils.pipeline import config_to_pipeline
from ragnroll.evaluation.eval import evaluate
from ragnroll.evaluation.data import load_evaluation_data
from ragnroll.utils.ingestion import index_documents
from ragnroll.evaluation.tracing import report_metrics_to_langfuse
from haystack_integrations.components.connectors.langfuse import LangfuseConnector

def main():
    # Generate a unique run ID
    run_id = str(uuid.uuid4())
    print(f"Run ID: {run_id}")
    
    # Define paths
    eval_data_path = "path/to/evaluation/data.json"  # Replace with actual path
    corpus_dir = "path/to/corpus"  # Replace with actual path
    config_file = "configs/baselines/llm_config.yaml"  # Replace with actual path
    
    # Load test data (simplified example)
    # If you don't have evaluation data, you can create a simple test dataset
    sample_data = {
        "test_cases": [
            {
                "input": "What is RAG?",
                "expected_output": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of documents with text generation to provide accurate, grounded answers."
            },
            {
                "input": "How does a vector database work?",
                "expected_output": "Vector databases store and index embeddings (numerical vector representations) of data, allowing for similarity search based on semantic meaning rather than keyword matching."
            }
        ]
    }
    
    try:
        # Create pipeline with Langfuse tracing enabled
        run_name = f"Langfuse-Test-{run_id}"
        
        # Configure the pipeline
        # For this example, we'll use a simple LLM pipeline
        pipeline = config_to_pipeline(config_file)
        
        # Add Langfuse connector to pipeline
        pipeline.add_component("tracer", LangfuseConnector(run_name))
        
        # Run evaluation
        print("Running evaluation...")
        result = evaluate(sample_data, pipeline, run_name=run_name)
        
        # Process and report metrics to Langfuse
        if isinstance(result, dict) and "trace_ids" in result:
            trace_ids = result["trace_ids"]
            
            if trace_ids:
                print(f"Found {len(trace_ids)} trace IDs")
                for trace_id in trace_ids:
                    print(f"Reporting metrics for trace: {trace_id}")
                    
                    # Report end-to-end metrics
                    if "metrics" in result:
                        report_metrics_to_langfuse(
                            trace_id,
                            result["metrics"],
                            metric_type="end2end"
                        )
                    
                    # Report component metrics
                    if "component_metrics" in result:
                        for component_type, metrics in result["component_metrics"].items():
                            report_metrics_to_langfuse(
                                trace_id,
                                metrics,
                                metric_type=component_type
                            )
                
                print("Metrics successfully reported to Langfuse")
                print("\nView your metrics in the Langfuse UI:")
                
                # Determine Langfuse host
                langfuse_host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
                org_name = "your-org"  # Replace with your org name
                
                print(f"{langfuse_host}/{org_name}/traces/{trace_ids[0]}")
            else:
                print("No trace IDs found. Make sure the LangfuseConnector is correctly added to your pipeline.")
        else:
            print("Evaluation result doesn't contain trace IDs")
            
        # Save evaluation results
        output_file = f"langfuse_metrics_test_{run_id}.csv"
        df = result["dataframe"] if isinstance(result, dict) else result
        df.to_csv(output_file)
        print(f"Evaluation results saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 