"""This module provides the RP To-Do CLI."""
# rptodo/cli.py

from typing import Optional, List
import os
import json
import pandas as pd
from pathlib import Path
import typer
from dotenv import load_dotenv

from ragnroll import __app_name__, __version__

app = typer.Typer()

CONFIG_PATH = Path(__file__).parent.parent / "configs"
BASELINES_PATH = CONFIG_PATH / "baselines"
ENV_PATH = Path(__file__).parent.parent / ".env"

load_dotenv(ENV_PATH)

os.environ["LANGFUSE_SECRET_KEY"] = os.environ["LANGFUSE_INIT_PROJECT_SECRET_KEY"]
os.environ["LANGFUSE_PUBLIC_KEY"] = os.environ["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack import tracing
from haystack_integrations.components.connectors.langfuse import LangfuseConnector

tracing.tracer.is_content_tracing_enabled = True

# Import necessary functions
from .utils.pipeline import config_to_pipeline, draw_pipeline

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
def test_generalization_error(
    eval_data_file: str = typer.Argument(..., help="Path to directory containing JSON/CSV files"),
    corpus_dir: str = typer.Argument(..., help="Path to directory containing corpus"),
    output_directory: str = typer.Argument(..., help="Path to directory containing JSON/CSV files"),
    experiment_name: str = typer.Option("RAG Experimentation", help="Experiment name"),
    strict: bool = typer.Option(True, help="Do not use the same config twice."),
):
    """
    Test generalization error of a model.
    """
    import mlflow
    from .utils.pipeline import config_to_pipeline, validate_pipeline
    from .utils.ingestion import index_documents
    from .evaluation.eval import Evaluator
    from .evaluation.data import load_evaluation_data
    from .evaluation.tracing import fetch_current_traces

    eval_data_path = Path(eval_data_file)
    test_data_path = eval_data_path.parent / "test" / eval_data_path.name


    mlflow.set_tracking_uri("http://localhost:8080")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        mlflow.set_experiment(experiment_id=experiment.experiment_id)
    else:
        raise ValueError(f"Experiment {experiment_name} not found")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if runs.empty:
        raise ValueError(f"No runs found for experiment {experiment_name}. Create a new evaluation dataset or use --no-strict (not recommended)")

    if "params.used_test_sets" in runs.columns and strict:
        # Check if the testset path is already in the params
        already_used_configs = runs[
            runs['params.used_test_sets'].str.contains(str(test_data_path))
            ]['params.config'].unique()
        
        runs = runs[~runs['params.config'].isin(already_used_configs)]

    gathered_results = []
    for _, run in runs.iterrows():

        with mlflow.start_run(run_id=run.run_id):
            run_name = run["tags.mlflow.runName"]

            pipeline = config_to_pipeline(configuration_dict=eval(run["params.config"]))
            validate_pipeline(pipeline)

            pipeline = index_documents(corpus_dir, pipeline)
            pipeline.add_component("tracer", LangfuseConnector(run_name))
            data = load_evaluation_data(test_data_path)

            evaluator = Evaluator(pipeline)
            result = evaluator.evaluate(evaluation_data=data, run_name=run_name, track_resources=False)

            traces = fetch_current_traces(run_name)
            
            for col in result.columns:
                metric_name = ".".join(("TEST",) + col)
                mlflow.log_metrics({metric_name: result[col].values[0]})
            
            if "params.used_test_sets" in run:
                used_test_sets = run["params.used_test_sets"]
            else:
                used_test_sets = []

            used_test_sets.append(str(test_data_path))
            mlflow.log_param("used_test_sets", used_test_sets)

            results = pd.concat([result, traces], axis=1)
            gathered_results.append(results)

            if Path(output_directory).exists():
                pd.concat(gathered_results).T.to_csv(output_directory, mode="a", header=False)
            else:
                pd.concat(gathered_results).T.to_csv(output_directory)
                
            print(f"Evaluation results saved to {output_directory}")


@app.command()
def run_evaluations(
    config_sources: str = typer.Argument(...), 
    eval_data_file : str = typer.Argument(...),
    corpus_dir : str = typer.Argument(...),
    output_directory: str = typer.Argument(...), # TODO This must be removed to get consistent output name for test set run
    track_resources: bool = typer.Option(True, help="Track system resource usage during evaluation"),
    baselines: bool = typer.Option(True, help="Run baselines"), 
    experiment_name: str = typer.Option("RAG Experimentation", help="Experiment name"),
):
    from .utils.pipeline import gather_config_paths, config_to_pipeline, validate_pipeline
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

    if baselines:
        baseline_paths = [
            BASELINES_PATH / "llm_config.yaml", 
            BASELINES_PATH / "predefined_bm25.yaml"
        ]
    else:
        baseline_paths = []

    # Gather all config paths from the source file (YAML, MATRIX-YAML, PYTHON)
    config_sources = gather_config_paths(Path(config_sources))

    gathered_results = []
    
    for config_path in baseline_paths + config_sources:
        print(f"Running evaluation for {config_path}")

        run_name = f"{config_path.parent.name}.{config_path.name}"
        with mlflow.start_run(run_name=run_name):
            
            # Load and prepare pipelines
            pipeline = config_to_pipeline(configuration_file_path=config_path)
            validate_pipeline(pipeline)

            params = extract_run_params(config_path)
            
            mlflow.log_params(params)

            pipeline = index_documents(corpus_dir, pipeline)
            pipeline.add_component("tracer", LangfuseConnector(run_name))
            data = load_evaluation_data(val_data_path)

            evaluator = Evaluator(pipeline)
            result = evaluator.evaluate(evaluation_data=data, run_name=run_name, track_resources=track_resources)

            traces = fetch_current_traces(run_name)
            
            for col in result.columns:
                metric_name = ".".join(("VAL",) + col)
                mlflow.log_metrics({metric_name: result[col].values[0]})

            results = pd.concat([result, traces], axis=1)
            gathered_results.append(results)

            if Path(output_directory).exists():
                pd.concat(gathered_results).T.to_csv(output_directory, mode="a", header=False)
            else:
                pd.concat(gathered_results).T.to_csv(output_directory)

            print(f"Evaluation results saved to {output_directory}")


    return

@app.command()
def draw_pipeline(
    config_file: str = typer.Argument(..., help="Path to the pipeline configuration file (YAML or Python)."),
    output_file: str = typer.Option("pipeline.png", "-o", help="Path to save the output PNG image.")
):
    """
    Loads a pipeline from a config file and draws it to a PNG image.
    """
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            typer.echo(f"Error: Config file not found at {config_path}", err=True)
            raise typer.Exit(code=1)
            
        pipeline = config_to_pipeline(configuration_file_path=config_path)
        draw_pipeline(pipeline, output_file)
        typer.echo(f"Pipeline drawn successfully to {output_file}")
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"An unexpected error occurred: {e}", err=True)
        raise typer.Exit(code=1)

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

if __name__ == "__main__":
    # invoke run_evaluations

    # invoke run_evaluations
    run_evaluations(
        config_sources="configs/from_pipeline/sample.yaml",
        eval_data_file="data/processed/dev_data/synthetic_rag_evaluation.json",
        corpus_dir="data/processed/dev_data/corpus",
        output_directory="output.csv",
        track_resources=True,
        baselines=True,
        experiment_name="RAG Experimentation"
    )
