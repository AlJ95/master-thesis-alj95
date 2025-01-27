from haystack import Pipeline
from pathlib import Path
from dotenv import load_dotenv

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
    
    load_dotenv()
    pipeline = Pipeline.load(open(configuration_file_path, "r"))
    return pipeline


def draw_pipeline(pipeline: Pipeline, output_file: str) -> None:
    """
    Draw a pipeline to a PNG file.

    Args:
        pipeline (Pipeline): The pipeline to draw.
        output_file (str): The name of the output file.
    """
    
    if not output_file.endswith(".png"):
        raise ValueError("Output file must be a PNG file.")

    pipeline.draw(output_file)
    print(f"Pipeline drawn to {output_file}")
