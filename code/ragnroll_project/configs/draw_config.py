from haystack import Pipeline
from pathlib import Path
import os
import sys

print(Path(__file__).parent.parent / "ragnroll")
sys.path.append(str(Path(__file__).parent.parent / "ragnroll"))
from utils.pipeline import config_to_pipeline, gather_config_paths

# CONFIG_PATH = "configs/configuration_validation/configuration_phase/llm_config_vllm.yaml"

# config_paths = gather_config_paths(Path(CONFIG_PATH))

os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-0000000000000000000000000000000000000000000000000000000000000000"
os.environ["OPENAI_API_KEY"] = "sk-proj-0000000000000000000000000000000000000000000000000000000000000000"

# pipeline = config_to_pipeline(configuration_file_path=config_paths[0])
# pipeline.draw(Path(CONFIG_PATH).parent / "pipeline.png", server_url="http://localhost:3001")


def draw_all_pipelines(config_dir: Path):
    """
    Recursively find all YAML config files and draw their pipeline diagrams
    """
    # Create output directory for drawings if it doesn't exist
    output_dir = config_dir / "pipeline_drawings"
    output_dir.mkdir(exist_ok=True)

    # Find all yaml files recursively
    for root, _, files in os.walk(config_dir):
        for file in files:
            if file.endswith(('.yaml', '.yml')):
                print(f"Drawing pipeline for {file}")
                config_path = Path(root) / file
                try:
                    # Load and draw pipeline
                    config_paths = gather_config_paths(config_path)
                    pipeline = config_to_pipeline(configuration_file_path=config_paths[0])
                    
                    # Create output filename based on config path
                    rel_path = config_path.relative_to(config_dir)
                    output_path = output_dir / f"{rel_path.stem}.pdf"
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    
                    # Draw pipeline
                    pipeline.draw(output_path, server_url="http://localhost:3001", params=dict(format="pdf", bgColor="!white", fit=True, paper="a4"))
                    print(f"Drew pipeline for {config_path} to {output_path}")
                    
                except Exception as e:
                    print(f"Error processing {config_path}: {str(e)}")

# Get configs directory path
configs_dir = Path(__file__).parent

# Draw all pipelines
draw_all_pipelines(configs_dir)
