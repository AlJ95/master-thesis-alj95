from haystack import Pipeline
from pathlib import Path

def config_to_pipeline(configuration_file_path: str) -> None:
    """
    Load a pipeline from a configuration file and draw it to a PNG file.

    Args:
        config_path (str): The path to the configuration file.
        config_name (str): The name of the configuration file.
    """
    configuration_file_path = Path(configuration_file_path)

    if not configuration_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {configuration_file_path.resolve()}")
    
    if configuration_file_path.suffix not in [".yaml", ".yml"]:
        raise ValueError("Configuration file must be a YAML file.")
    
    if output_file_name is None:
        output_file_name = configuration_file_path.with_suffix(".png")
    else:
        if not output_file_name.endswith(".png"):
            raise ValueError("Output file must be a PNG file.")
    
        output_file_name = Path(output_file_name)

    pipeline = Pipeline.load(open(configuration_file_path, "r"))
    return pipeline