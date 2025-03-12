import os
from langfuse import Langfuse
import pandas as pd

def fetch_current_traces(run_id):
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

    current_traces = [trace for trace in all_traces if run_id in trace.name]
    trace_ids = [trace.id for trace in current_traces]
    current_observations = [observation for observation in all_observations if observation.trace_id in trace_ids]

    print(f"There are {len(current_traces)} traces and {len(current_observations)} observations")

    latencies = _extract_latencies(current_traces, current_observations, save_to_csv=True)

    if len(current_traces) == 0:
        raise ValueError(f"No traces found for run_id: {run_id}")
    
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

    if save_to_csv:
        df.to_csv("latencies.csv", index=False)
    return df
