"""This module provides the RP To-Do CLI."""
# rptodo/cli.py

from typing import Optional, List

import typer

from ragnroll import __app_name__, __version__

app = typer.Typer()

@app.command()
def split_data(
    path: List[str] = typer.Argument(...),
):
    """
    Process to split data manually
    """

@app.command()
def run_evaluations(
    configuration_file: str = typer.Argument(...),
    output_directory: str = typer.Argument(...),
):
    pass

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
    config_to_pipeline(configuration_file, output_file)

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