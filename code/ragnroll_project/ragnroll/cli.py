"""This module provides the RP To-Do CLI."""
# rptodo/cli.py

from genericpath import exists
from typing import Optional, List

import typer
from pathlib import Path
from random import sample

from ragnroll import __app_name__, __version__

app = typer.Typer()

@app.command()
def split_data(
    path: str = typer.Argument(...),
    testset_percentage: int = typer.Argument(...),
):
    """
    Process to split data manually
    """
    path = Path(path)
    test_data_path = path / "holdout_testset"
    test_data_path.mkdir(exist_ok=True)

    if path.is_dir():
        files_to_split = list(path.iterdir())

        sample_size = testset_percentage * len(files_to_split) // 100
        test_data = sample(files_to_split, sample_size)

        for test_file in test_data:
            test_file.rename(test_data_path / test_file.name)
        
    else:
        pass
        # ToDo Handle Single File CSVs / JSONs / ...
    



@app.command()
def run_evaluations(
    configuration_file: str = typer.Argument(...),
    data_path : str = typer.Argument(...),
    output_directory: str = typer.Argument(...),
):
    from .pipeline import config_to_pipeline
    pipeline = config_to_pipeline(configuration_file)
    
    pipeline.run()


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
    from .pipeline import config_to_pipeline
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