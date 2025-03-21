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
    

def test_generalization_error(
    path: str = typer.Argument(..., help="Path to directory containing JSON/CSV files"),
):
    """
    Test generalization error of a model.
    """
    # TODO: Get all configurations from output.csv
    # TODO: Run evaluation for each configuration on test set
    # TODO: Mark test set as used
    # TODO: If user wants to run on used test set, raise warning


@app.command()
def run_evaluations(
    configuration_file: str = typer.Argument(...), 
    eval_data_file : str = typer.Argument(...),
    corpus_dir : str = typer.Argument(...),
    output_directory: str = typer.Argument(...), # TODO This must be removed to get consistent output name for test set run
    track_resources: bool = typer.Option(True, help="Track system resource usage during evaluation"), # TODO Remove this option
    baselines: bool = typer.Option(True, help="Run baselines"), 
    experiment_name: str = typer.Option("RAG Experimentation", help="Experiment name"),
):
    from .utils.pipeline import config_to_pipeline
    from .evaluation.eval import evaluate
    from .evaluation.data import load_evaluation_data
    from .utils.ingestion import index_documents
    from .evaluation.tracing import fetch_current_traces
    from .utils.data import val_test_split
    from .utils.config import extract_run_params
    from pathlib import Path
    import uuid
    import warnings
    import mlflow

    mlflow.set_tracking_uri(uri="http://localhost:8080")
    eval_data_path = Path(eval_data_file)

    # Split the evaluation data into val, test sets based on Simon et al. (2024) 
    val_test_split(eval_data_path)

    if not eval_data_path.exists():
        warnings.warn(f"Evaluation data path {eval_data_path} does not exist")
        typer.Abort()
    if eval_data_path.is_dir():
        warnings.warn(f"Evaluation data path {eval_data_path} is a directory")
        typer.Abort()

    val_data_path = eval_data_path.parent / "val" / eval_data_path.name
    assert val_data_path.exists(), f"Validation data path {val_data_path} does not exist"

    # Setup Run-ID
    mlflow.set_experiment(experiment_name=experiment_name)

    if baselines:
        baseline_configs = ["configs/baselines/llm_config.yaml", "configs/baselines/predefined_bm25.yaml"]
    else:
        baseline_configs = []

    gathered_results = []
    
    for config in baseline_configs + [configuration_file]:
        print(f"Running evaluation for {config}")

        run_name = ".".join(config.split("/")[-2:])
        with mlflow.start_run(run_name=run_name):
        
                # Load and prepare pipelines
                pipeline = config_to_pipeline(config)

                params = extract_run_params(config)
                mlflow.log_params(params)

                pipeline = index_documents(corpus_dir, pipeline)
                pipeline.add_component("tracer", LangfuseConnector(run_name))
                data = load_evaluation_data(val_data_path)

                result = evaluate(data, pipeline, run_name=run_name, track_resources=track_resources)
                df = result["dataframe"]

                traces = fetch_current_traces(run_name)
                
                for col in df.columns:
                    mlflow.log_metrics({".".join(col): df[col].values[0]})

                mlflow.log_table(data=df, artifact_file="evaluation_results.json")

                results = pd.concat([df, traces], axis=1)
                gathered_results.append(results)

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
