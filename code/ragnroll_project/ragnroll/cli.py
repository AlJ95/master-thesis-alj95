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
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import json
    from pathlib import Path
    
    path = Path(path)
    
    # Validate input parameters
    if not (0 < test_size < 100):
        raise typer.BadParameter("test_size must be between 0 and 100")
    
    # Create output directories
    val_path = path / "val" 
    test_path = path / "test"
    val_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)
    
    # Process files
    if path.is_dir():
        # Get only JSON/CSV files
        files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in ['.json', '.csv']]
        
        if not files:
            raise typer.BadParameter("No JSON or CSV files found in directory")
            
        # Process each file
        for file in files:
            # Load data
            if file.suffix.lower() == '.json':
                with open(file) as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:  # CSV
                df = pd.read_csv(file)
                
            # Split data
            val_df, test_df = train_test_split(
                df,
                test_size=test_size/100,
                random_state=random_state
            )
            
            # Save splits
            base_name = file.stem
            val_df.to_json(val_path / f"{base_name}_val.json", orient='records')
            test_df.to_json(test_path / f"{base_name}_test.json", orient='records')
            
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
    from .evaluation.system_metrics import SystemResourceTracker
    from pathlib import Path
    import uuid
    import os

    # Setup Run-ID
    run_id = str(uuid.uuid4())
    print(f"Run-ID: {run_id}")

    # Initialize the resource tracker if tracking is enabled
    resource_tracker = None
    if track_resources:
        resource_tracker = SystemResourceTracker()
        resource_tracker.start_tracking()
        print(f"System resource tracking started")

    try:
        # Load and prepare pipelines
        llm_pipeline = config_to_pipeline("configs/baselines/llm_config.yaml")

        naive_rag_pipeline = config_to_pipeline("configs/baselines/predefined_bm25.yaml")
        naive_rag_pipeline = index_documents(corpus_dir, naive_rag_pipeline)

        rag_pipeline = config_to_pipeline(configuration_file)
        rag_pipeline = index_documents(corpus_dir, rag_pipeline)
        
        data = load_evaluation_data(eval_data_path)

        print("--------------------------------")
        run_name = f"LLM-Baseline-{run_id}"
        print(run_name)
        llm_pipeline.add_component("tracer", LangfuseConnector(run_name))
        result_baseline_llm = evaluate(data, llm_pipeline, run_name=run_name)

        print("--------------------------------")
        print("Baseline Naive RAG")
        # run_name = f"Naive-RAG-Baseline-{run_id}"
        # print(run_name)
        # naive_rag_pipeline.add_component("tracer", LangfuseConnector(run_name))
        # result_baseline_naive_rag = evaluate(data, naive_rag_pipeline, run_name=run_name)
        # print("--------------------------------")
        print("RAG")
        # run_name = f"RAG-Pipeline-{run_id}"
        # print(run_name)
        # rag_pipeline.add_component("tracer", LangfuseConnector(run_name))
        # result_rag = evaluate(data, rag_pipeline, run_name=run_name)
        # print("--------------------------------")

        from .evaluation.tracing import fetch_current_traces
        traces = fetch_current_traces(run_id)

        # Combine the evaluation results and traces
        results = pd.concat([result_baseline_llm, traces], axis=1)

        # If resource tracking is enabled, add resource metrics to results
        if track_resources and resource_tracker:
            # Get resource metrics
            metrics_summary = resource_tracker.get_metrics_summary()
            
            # Flatten the nested dictionaries
            flat_metrics = {}
            
            # Duration and samples
            flat_metrics["duration_seconds"] = float(metrics_summary.get("duration_seconds", 0))
            flat_metrics["samples_count"] = float(metrics_summary.get("samples_count", 0))
            
            # CPU metrics
            if "cpu" in metrics_summary and isinstance(metrics_summary["cpu"], dict):
                cpu = metrics_summary["cpu"]
                # System CPU
                if "system" in cpu and isinstance(cpu["system"], dict):
                    system = cpu["system"]
                    flat_metrics["cpu_system_mean"] = float(system.get("mean", 0))
                    flat_metrics["cpu_system_max"] = float(system.get("max", 0))
                    flat_metrics["cpu_system_min"] = float(system.get("min", 0))
                # Process CPU
                if "process" in cpu and isinstance(cpu["process"], dict):
                    process = cpu["process"]
                    flat_metrics["cpu_process_mean"] = float(process.get("mean", 0))
                    flat_metrics["cpu_process_max"] = float(process.get("max", 0))
                    flat_metrics["cpu_process_min"] = float(process.get("min", 0))
            
            # Memory metrics
            if "memory" in metrics_summary and isinstance(metrics_summary["memory"], dict):
                memory = metrics_summary["memory"]
                # System percentage
                if "system_percent" in memory and isinstance(memory["system_percent"], dict):
                    sys_percent = memory["system_percent"]
                    flat_metrics["memory_system_percent_mean"] = float(sys_percent.get("mean", 0))
                    flat_metrics["memory_system_percent_max"] = float(sys_percent.get("max", 0))
                    flat_metrics["memory_system_percent_min"] = float(sys_percent.get("min", 0))
                # System used GB
                if "system_used_gb" in memory and isinstance(memory["system_used_gb"], dict):
                    sys_used = memory["system_used_gb"]
                    flat_metrics["memory_system_used_gb_mean"] = float(sys_used.get("mean", 0))
                    flat_metrics["memory_system_used_gb_max"] = float(sys_used.get("max", 0))
                    flat_metrics["memory_system_used_gb_min"] = float(sys_used.get("min", 0))
                # Process MB
                if "process_mb" in memory and isinstance(memory["process_mb"], dict):
                    proc_mb = memory["process_mb"]
                    flat_metrics["memory_process_mb_mean"] = float(proc_mb.get("mean", 0))
                    flat_metrics["memory_process_mb_max"] = float(proc_mb.get("max", 0))
                    flat_metrics["memory_process_mb_min"] = float(proc_mb.get("min", 0))
            
            # Create flat resource metrics DataFrame
            resource_df = pd.DataFrame([flat_metrics])
            resource_df.loc[:, "run_id"] = run_name
            resource_df.set_index("run_id", inplace=True)
            
            # Prefix resource metrics columns to avoid conflicts
            resource_df.columns = pd.MultiIndex.from_tuples([("SYS", col) for col in resource_df.columns])
            
            # Add to results
            results = pd.concat([results, resource_df], axis=1)

        # Save the combined results
        results.T.to_csv(output_directory)
        print(f"Evaluation results saved to {output_directory}")

    finally:
        # Stop resource tracking if it was started
        if track_resources and resource_tracker:
            resource_tracker.stop_tracking()
            resource_tracker.print_summary()

    
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
