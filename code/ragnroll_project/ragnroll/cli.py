"""This module provides the RP To-Do CLI."""
# rptodo/cli.py

from typing import Optional, List
import os

import pandas as pd

import typer

from ragnroll import __app_name__, __version__

app = typer.Typer()

os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-6a6b4f2e-53bb-4381-8351-c1549b0b44db"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-e3818c59-351d-46ea-917f-d06cde587ac5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack import tracing
from haystack_integrations.components.connectors.langfuse import LangfuseConnector
from langfuse import Langfuse

tracing.tracer.is_content_tracing_enabled = True

@app.command()
def split_data(
    path: str = typer.Argument(..., help="Path to directory containing JSON/CSV files"),
    test_size: float = typer.Argument(..., help="Percentage of data for test set (0-100)"),
    random_state: int = typer.Option(42, help="Random seed for reproducibility"),
):
    """
    Split data into train, validation and test sets from JSON/CSV files.
    Creates three directories: train, val, test
    """
    from .utils.data import val_test_split
    val_test_split(path, test_size, random_state)
    typer.echo(f"Successfully split data into train/val/test sets in {path}")
    



@app.command()
def run_evaluations(
    configuration_file: str = typer.Argument(...),
    eval_data_path : str = typer.Argument(...),
    corpus_dir : str = typer.Argument(...),
    output_directory: str = typer.Argument(...),
    track_resources: bool = typer.Option(True, help="Track system resource usage during evaluation"),
    # ToDo: Baselines Options
):
    from .utils.pipeline import config_to_pipeline
    from .evaluation.eval import evaluate
    from .evaluation.data import load_evaluation_data
    from .utils.ingestion import index_documents
    from .evaluation.tracing import fetch_current_traces, report_metrics_to_langfuse
    from pathlib import Path
    import uuid
    import os
    from dvclive import Live

    # Setup Run-ID
    run_id = str(uuid.uuid4())
    print(f"Run-ID: {run_id}")

    baseline_configs = ["configs/baselines/llm_config.yaml", "configs/baselines/predefined_bm25.yaml"]

    gathered_results = []
    for config in baseline_configs + [configuration_file]:
        print(f"Running evaluation for {config}")
        
        try:
            # Load and prepare pipelines
            pipeline = config_to_pipeline(config)
            pipeline = index_documents(corpus_dir, pipeline)

        
            data = load_evaluation_data(eval_data_path)

            print("--------------------------------")
            run_name = f"{config.split('/')[-1]}-{run_id}"
            print(run_name)
            pipeline.add_component("tracer", LangfuseConnector(run_name))
            result = evaluate(data, pipeline, run_name=run_name, track_resources=track_resources)

        
            # Fetch traces from Langfuse
            traces = fetch_current_traces(run_name)

        except Exception as e:
            print(f"Warning: Failed to report metrics to Langfuse: {e}")

        # In parallel, report evaluation metrics as scores to Langfuse
        try:
            # Extract DataFrames for pandas reporting (original approach)
            df = result["dataframe"] if isinstance(result, dict) else result
            print(df)

            # Report end-to-end metrics for each pipeline
            if isinstance(result, dict) and "trace_ids" in result and "metrics" in result:
                trace_ids = result["trace_ids"]
                for trace_id in trace_ids:
                    report_metrics_to_langfuse(
                        trace_id, 
                        result["metrics"], 
                        metric_type="end2end", 
                    )
                    
                    # Report component metrics if available
                    if "component_metrics" in result:
                        for component_type, metrics in result["component_metrics"].items():
                            report_metrics_to_langfuse(
                                trace_id,
                                metrics,
                                metric_type=component_type,
                            )
            
            print("Metrics successfully reported to Langfuse as scores")
        except Exception as e:
            print(f"Warning: Failed to report metrics to Langfuse: {e}")
            
        results = pd.concat([df, traces], axis=1)
        gathered_results.append(results)

        # Save the combined results
    pd.concat(gathered_results).T.to_csv(output_directory)
    print(f"Evaluation results saved to {output_directory}")

    return

def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    return
