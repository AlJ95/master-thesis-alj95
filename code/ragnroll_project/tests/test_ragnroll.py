# tests/test_rptodo.py

from typer.testing import CliRunner
from unittest.mock import patch
from pathlib import Path

from ragnroll import __app_name__, __version__, cli

runner = CliRunner()

def test_version():
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} v{__version__}\n" in result.stdout

def test_custom_component():
    # Patch the functions called internally by the draw-pipeline command
    with patch("ragnroll.cli.config_to_pipeline") as mock_load, \
         patch("ragnroll.cli.draw_pipeline") as mock_draw:
        
        # Correct the path relative to the project root (where pytest runs)
        result = runner.invoke(cli.app, ["draw-pipeline", "configs/example_component.yaml", "-o", "test_output.png"])
        
        # Check that the command exited successfully
        assert result.exit_code == 0
        
        # Check that the internal functions were called with expected args
        mock_load.assert_called_once()
        
        # Check the kwargs passed to config_to_pipeline
        call_args_list = mock_load.call_args_list
        assert len(call_args_list) == 1
        args, kwargs = call_args_list[0]
        assert "configuration_file_path" in kwargs
        assert isinstance(kwargs["configuration_file_path"], Path)
        assert str(kwargs["configuration_file_path"]).endswith("configs\\example_component.yaml")

        mock_draw.assert_called_once_with(mock_load.return_value, "test_output.png")
        
        # Check for the success message added in the CLI command
        assert "Pipeline drawn successfully to test_output.png" in result.stdout
