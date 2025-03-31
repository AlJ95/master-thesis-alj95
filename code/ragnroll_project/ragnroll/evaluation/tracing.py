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
