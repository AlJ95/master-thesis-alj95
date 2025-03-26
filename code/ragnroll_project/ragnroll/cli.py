"""This module provides the RP To-Do CLI."""
# rptodo/cli.py

from typing import Optional, List
import os

import pandas as pd
from pathlib import Path
import typer
from dotenv import load_dotenv

from ragnroll import __app_name__, __version__

app = typer.Typer()

os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-6a6b4f2e-53bb-4381-8351-c1549b0b44db"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-e3818c59-351d-46ea-917f-d06cde587ac5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack import tracing
from haystack_integrations.components.connectors.langfuse import LangfuseConnector

tracing.tracer.is_content_tracing_enabled = True

CONFIG_PATH = Path(__file__).parent.parent / "configs"
BASELINES_PATH = CONFIG_PATH / "baselines"
PIPELINES_PATH = Path(__file__).parent.parent / "pipelines"
FROM_PIPELINE_PATH = CONFIG_PATH / "from_pipeline"

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
    config_source: str = typer.Argument(...), 
    eval_data_file : str = typer.Argument(...),
    corpus_dir : str = typer.Argument(...),
    output_directory: str = typer.Argument(...), # TODO This must be removed to get consistent output name for test set run
    track_resources: bool = typer.Option(True, help="Track system resource usage during evaluation"),
    baselines: bool = typer.Option(True, help="Run baselines"), 
    experiment_name: str = typer.Option("RAG Experimentation", help="Experiment name"),
):
    from .utils.pipeline import config_to_pipeline, validate_pipeline
    from .evaluation.eval import Evaluator
    from .evaluation.data import load_evaluation_data
    from .utils.ingestion import index_documents
    from .evaluation.tracing import fetch_current_traces
    from .utils.data import val_test_split
    from .utils.config import extract_run_params
    from pathlib import Path
    import warnings
    import mlflow

    if os.getenv("MLFLOW_TRACKING_URI"):
        uri = os.getenv("MLFLOW_TRACKING_URI")
    else:
        uri = "http://localhost:8080"

    # check if uri is accessible
    try:
        mlflow.set_tracking_uri(uri=uri)
    except Exception as e:
        raise ValueError(f"Failed to set tracking URI: {e}")

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

    config_source = Path(config_source)

    if baselines:
        baseline_paths = [
            BASELINES_PATH / "llm_config.yaml", 
            BASELINES_PATH / "predefined_bm25.yaml"
        ]
    else:
        baseline_paths = []

    gathered_results = []
    
    for config_path in baseline_paths + [config_source]:
        print(f"Running evaluation for {config_path}")

        run_name = f"{config_path.parent.name}.{config_path.name}"
        with mlflow.start_run(run_name=run_name):
            
            # TODO: Outsource into utils.pipeline
            # Load and prepare pipelines
            if config_path.suffix == ".yaml":
                pipeline = config_to_pipeline(config_path)
                params = extract_run_params(config_path)
            elif config_path.suffix == ".py":
                pipeline = None
                try:
                    import importlib
                    import sys

                    # Absoluten Pfad zum Projekt-Root-Verzeichnis hinzufügen (wo pipelines/ liegt)
                    project_root = Path(__file__).parent.parent  # Das sollte zum ragnroll_project Verzeichnis zeigen
                    sys.path.insert(0, str(project_root))
                    
                    # Importieren des Moduls aus pipelines/ direkt
                    module_name = f"pipelines.{config_path.stem}"
                    module = importlib.import_module(module_name)
                    pipeline = module.pipeline
                except ImportError as e:
                    raise ImportError(f"Import error: {e}")
                
                yaml_path = FROM_PIPELINE_PATH / f"{config_path.stem}.yaml"
                pipeline.dump(open(yaml_path, "w"))
                params = extract_run_params(yaml_path)
            else:
                raise ValueError(f"Invalid config file type: {config_path}")
                
            mlflow.log_params(params)
            validate_pipeline(pipeline)

            pipeline = index_documents(corpus_dir, pipeline)
            pipeline.add_component("tracer", LangfuseConnector(run_name))
            data = load_evaluation_data(val_data_path)

            evaluator = Evaluator(pipeline)
            result = evaluator.evaluate(evaluation_data=data, run_name=run_name, track_resources=track_resources)

            traces = fetch_current_traces(run_name)
            
            for col in result.columns:
                mlflow.log_metrics({".".join(col): result[col].values[0]})

            mlflow.log_table(data=result, artifact_file="evaluation_results.json")

            results = pd.concat([result, traces], axis=1)
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
