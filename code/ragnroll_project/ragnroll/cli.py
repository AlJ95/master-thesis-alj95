"""This module provides the RP To-Do CLI."""
# rptodo/cli.py

from typing import Optional, List

import typer

from ragnroll import __app_name__, __version__

app = typer.Typer()

@app.command()
def split_data(
    path: List[str] | str = typer.Argument(...),
):
    """
    Process to split data manually
    """

@app.command()
def run_evaluations(
    configuration_file: List[str] | str = typer.Argument(...),
    output_directory: str = typer.Argument(...),
):
    pass

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