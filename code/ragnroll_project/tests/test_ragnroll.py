# tests/test_rptodo.py

from typer.testing import CliRunner

from ragnroll import __app_name__, __version__, cli

runner = CliRunner()

def test_version():
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} v{__version__}\n" in result.stdout

def test_custom_component():
    result = runner.invoke(cli.app, ["draw-pipeline", "../configs/example_component.yaml", "-o", "test_output.png"])
    assert result.exit_code == 0
    assert "Pipeline drawn to test_output.png" in result.stdout
