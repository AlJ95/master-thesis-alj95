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
    from pathlib import Path
    import uuid
    import os

    # Setup Run-ID
    run_id = str(uuid.uuid4())
    print(f"Run-ID: {run_id}")

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
        result_baseline_llm = evaluate(data, llm_pipeline, run_name=run_name, track_resources=track_resources)

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

        # Save the combined results
        results.T.to_csv(output_directory)
        print(f"Evaluation results saved to {output_directory}")

    finally:
        pass
    
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
