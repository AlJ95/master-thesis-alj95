"""This module provides the RP To-Do CLI."""
# rptodo/cli.py

from typing import Optional, List

import typer

from ragnroll import __app_name__, __version__

app = typer.Typer()

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
):
    from .utils.pipeline import config_to_pipeline
    from .evaluation.eval import evaluate
    from .evaluation.data import load_evaluation_data
    from .utils.ingestion import index_documents
    llm_pipeline = config_to_pipeline("configs/baselines/llm_config.yaml")

    naive_rag_pipeline = config_to_pipeline("configs/baselines/predefined_bm25.yaml")
    naive_rag_pipeline = index_documents(corpus_dir, naive_rag_pipeline)

    rag_pipeline = config_to_pipeline(configuration_file)
    rag_pipeline = index_documents(corpus_dir, rag_pipeline)
    
    data = load_evaluation_data(eval_data_path)

    print("--------------------------------")
    print("Baseline LLM")   
    result_baseline_llm = evaluate(data, llm_pipeline)
    print("--------------------------------")
    print("Baseline Naive RAG")
    result_baseline_naive_rag = evaluate(data, naive_rag_pipeline)
    print("--------------------------------")
    print("RAG")
    result_rag = evaluate(data, rag_pipeline)
    print("--------------------------------")

    return result_baseline_llm, result_baseline_naive_rag, result_rag


@app.command()
def draw_pipeline(
    configuration_file: str = typer.Argument(...),
    output_file: str = typer.Option(
        None,
        "--output-file",
        "-o",
        help="The name of the output file.",
    ),
):
    from .utils.pipeline import config_to_pipeline
    pipeline = config_to_pipeline(configuration_file)
    pipeline.draw(path=output_file)

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
