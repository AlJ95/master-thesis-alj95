"""
Functions for working with traces and metrics in Langfuse.
"""
import os
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from langfuse import Langfuse

def fetch_current_traces(run_name: str) -> pd.DataFrame:
    """
    Fetch traces for a specific run from Langfuse.
    
    Args:
        run_id: Run identifier to filter traces
        
    Returns:
        DataFrame containing trace information
    """
    try:
        langfuse = Langfuse(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],   
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=os.environ["LANGFUSE_HOST"],
        )

        all_traces = langfuse.fetch_traces().data
        all_observations = langfuse.fetch_observations().data

    except Exception as e:
        raise ValueError(f"Langfuse is not configured: {e}")

    current_traces = [trace for trace in all_traces if run_name == trace.name]
    trace_ids = [trace.id for trace in current_traces]
    current_observations = [observation for observation in all_observations if observation.trace_id in trace_ids]

    print(f"There are {len(current_traces)} traces and {len(current_observations)} observations")

    latencies = _extract_latencies(current_traces, current_observations, save_to_csv=True)

    if len(current_traces) == 0:
        raise ValueError(f"No traces found for run_id: {run_name}")
    
    return latencies

def _extract_latencies(traces, observations, save_to_csv=False):
    latencies = {}
    for trace in traces:
        for observation in observations:
            if observation.trace_id == trace.id:
                latencies[observation.id] = {
                    "type": observation.type,
                    "name": observation.name,
                    "latency": observation.latency,
                    "trace_id": trace.id,
                    "trace_name": trace.name,
                    "trace_latency": trace.latency,
                }
    df = pd.DataFrame(latencies).T

    df["latency"] = pd.to_numeric(df["latency"], errors="coerce")
    df["trace_latency"] = pd.to_numeric(df["trace_latency"], errors="coerce")

    aggregated_df = (
        df
        .drop(columns=["type"])
        .groupby(["name", "trace_name"])
        .mean(numeric_only=True)
        .reset_index()
        .set_index(["trace_name"])
        .pivot(columns="name", values="latency")
    )

    aggregated_df.loc[:, "trace_latency"] = df.groupby(["trace_name"]).mean(numeric_only=True)["trace_latency"]

    aggregated_df.columns = pd.MultiIndex.from_tuples([("LAT", col) for col in aggregated_df.columns])
    

    if save_to_csv:
        df.to_csv("latencies.csv", index=False)

    return aggregated_df

def report_metrics_to_langfuse(
    trace_id: str,
    metrics: Dict[str, Any],
    metric_type: str = "end2end",
    component_name: Optional[str] = None
) -> None:
    """
    Report metrics to Langfuse as scores.
    
    Args:
        trace_id: Langfuse trace ID to attach scores to
        metrics: Dictionary of metrics to report (name -> result)
        metric_type: Type of metrics being reported (end2end, retriever, generator)
        component_name: Optional component name for component-specific metrics
    """
    # Initialize Langfuse client from environment variables
    langfuse = Langfuse(
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        host=os.environ.get("LANGFUSE_HOST")
    )
    
    # Iterate through metrics and create scores
    for metric_name, metric_result in metrics.items():
        # Extract score value
        score_value = metric_result.get("score", 0.0)
        
        # Create score name with appropriate prefix
        score_name = f"{metric_type}.{metric_name}"
        if component_name:
            score_name = f"{metric_type}.{component_name}.{metric_name}"
        
        # Report score to Langfuse
        langfuse.score(
            trace_id=trace_id,
            name=score_name,
            value=score_value,  # Main score value
            comment=f"Success: {metric_result.get('success', False)}",
            # Include detailed results as metadata
            metadata={
                "success": metric_result.get("success", False),
                "threshold": metric_result.get("threshold", 0.0),
                "detailed_results": metric_result.get("detailed_results", {}),
                "individual_scores": metric_result.get("individual_scores", [])
            }
        )
    
    # Ensure scores are sent to Langfuse
    langfuse.flush()

def report_batch_metrics_to_langfuse(
    trace_ids: List[str],
    metrics_list: List[Dict[str, Any]],
    metric_type: str = "end2end",
    component_names: Optional[List[str]] = None
) -> None:
    """
    Report a batch of metrics to Langfuse as scores.
    
    Args:
        trace_ids: List of Langfuse trace IDs to attach scores to
        metrics_list: List of dictionaries of metrics to report
        metric_type: Type of metrics being reported (end2end, retriever, generator)
        component_names: Optional list of component names for component-specific metrics
    """
    # Initialize Langfuse client from environment variables
    langfuse = Langfuse(
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    
    # Process each trace and its metrics
    for i, (trace_id, metrics) in enumerate(zip(trace_ids, metrics_list)):
        component_name = component_names[i] if component_names and i < len(component_names) else None
        
        # Iterate through metrics and create scores
        for metric_name, metric_result in metrics.items():
            # Extract score value
            score_value = metric_result.get("score", 0.0)
            
            # Create score name with appropriate prefix
            score_name = f"{metric_type}.{metric_name}"
            if component_name:
                score_name = f"{metric_type}.{component_name}.{metric_name}"
            
            # Report score to Langfuse
            langfuse.score(
                trace_id=trace_id,
                name=score_name,
                value=score_value,
                comment=f"Success: {metric_result.get('success', False)}",
                metadata={
                    "success": metric_result.get("success", False),
                    "threshold": metric_result.get("threshold", 0.0),
                    "detailed_results": metric_result.get("detailed_results", {}),
                    "individual_scores": metric_result.get("individual_scores", [])
                }
            )
    
    # Ensure scores are sent to Langfuse
    langfuse.flush()
