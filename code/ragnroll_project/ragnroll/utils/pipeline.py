from haystack import Pipeline
from haystack.core.component import InputSocket, OutputSocket
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple
import yaml
from itertools import product
try:
    from .components import *
except ImportError:
    import sys
    sys.path.append(Path(__file__).parent.parent)
    from components import *
import haystack.dataclasses


def extract_matrix_configuration(configuration_file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract the matrix configuration from a configuration file.
    """
    with open(configuration_file_path, "r") as file:
        configuration = yaml.safe_load(file)

    matrix_parameters = {}
    def find_lists_in_yaml(data, path=""):
        """Recursively search through YAML data for any list values."""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                if isinstance(value, list) and key not in ["connections", "env_vars"] and value:
                    matrix_parameters[new_path] = value
                find_lists_in_yaml(value, new_path)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                find_lists_in_yaml(item, new_path)

    # Search through entire configuration for lists
    find_lists_in_yaml(configuration)

    if not matrix_parameters:
        return [configuration]

    print("\nMatrix parameters:")
    for key, value in matrix_parameters.items():
        print(f"{key}: {value}")
    
    # create all possible combinations of the matrix parameters
    combinations = list(product(*matrix_parameters.values()))
    print(f"\nAll possible combinations ({len(combinations)}):")
    for combination in combinations:
        # Create key-value pairs using parameter names and values
        param_dict = dict(zip(matrix_parameters.keys(), combination))
        for key, value in param_dict.items():
            print(f"  {key}: {value}")
        print()

    # replace the matrix parameters in the configuration with the combinations
    configurations = []
    for combination in combinations:
        for key, value in matrix_parameters.items():
            keys = key.split(".")
            current_dict = configuration["components"]
            for k in keys[:-1]:
                current_dict = current_dict[k]
            current_dict[keys[-1]] = value
        configurations.append(configuration)

    return configuration

def config_to_pipeline(configuration_file_path: Path) -> Pipeline:
    """
    Load a pipeline from a configuration file and draw it to a PNG file.

    Args:
        config_path (str): The path to the configuration file.
        config_name (str): The name of the configuration file.
    """

    configurations = extract_matrix_configuration(configuration_file_path)

    if not configuration_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {configuration_file_path.resolve()}")
    
    if configuration_file_path.suffix not in [".yaml", ".yml"]:
        raise ValueError("Configuration file must be a YAML file.")
    
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    pipelines = []
    for configuration in configurations:
        pipeline = Pipeline.load(open(configuration, "r"))
        pipelines.append(pipeline)
    return pipelines


def extract_component_structure(pipeline: Pipeline) -> Dict[str, Any]:
    """
    Extract the structure of a pipeline.
    """
    pipeline_dict = pipeline.to_dict()
    components = pipeline_dict["components"]
    connections = pipeline_dict["connections"]

    structure = {}
    for name, component in components.items():

        input_connections = [
            con 
            for con in connections 
            if name == con["receiver"].split(".")[0]
            ]
        
        output_connections = [
            con for 
            con in connections 
            if name == con["sender"].split(".")[0]
        ]

        structure[name] = {
            "type": component["type"],
            "input_connections": input_connections,
            "input_vars": [con["receiver"].split(".")[1] for con in input_connections],
            "output_connections": output_connections,
            "output_vars": [con["sender"].split(".")[1] for con in output_connections],
        }

        return structure


def get_predecessor_connection_mappings(pipeline: Pipeline, component_name: str) -> List[Tuple[str, InputSocket, OutputSocket]]:
    """
    Get the predecessor connection mappings of a given component.
    """
    receivers_per_component = [
        pipeline._find_receivers_from(component_name)
        for component_name in pipeline.to_dict()["components"].keys()
        ]
    
    predecessors = []

    for receivers in receivers_per_component:
        if not receivers:
            continue
        for receiver_mapping in receivers:
            # Check if the component name is in the OutputSocket of the receiver_mapping which is the second element of the tuple
            if component_name in receiver_mapping[1].receivers:
                for sender in receiver_mapping[2].senders:
                    predecessors.append((sender, receiver_mapping[1], receiver_mapping[2]))

    return predecessors


def get_last_component_with_documents(pipeline: Pipeline, component_name: str) -> str:
    """
    Get the name of the last component that has a documents input socket.
    In Haystack, usually a builder-type component comes before a generator-type component.
    Prompt builder can take multiple components as input, so we need to check all predecessors if they have a documents input socket.
    """
    component_type = pipeline.to_dict()["components"][component_name]["type"]

    if ".generators." not in component_type:
        raise ValueError("Component is not a generator.")
    
    prompt_builder = get_predecessor_connection_mappings(pipeline, component_name)[0]
    predecessors = get_predecessor_connection_mappings(pipeline, prompt_builder[0])
    predecessor_with_documents = [
        predecessor
        for predecessor in predecessors
        if predecessor[1].type == List[haystack.dataclasses.document.Document]
        ]
    
    if len(predecessor_with_documents) == 0:
        return None

    return predecessor_with_documents[0][0]



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


def validate_pipeline(pipeline: Pipeline) -> None:
    """
    Validate a pipeline.
    """
    if not "answer_builder" in pipeline.to_dict()["components"]:
        raise ValueError("Pipeline must have an answer builder.")

    

if __name__ == "__main__":
    if False:
        config_file = Path(__file__).parent.parent.parent / "configs" / "baselines" / "predefined_bm25.yaml"
            
        import os
        assert os.path.exists(config_file), "Path does not exist"

        pipeline = config_to_pipeline(config_file)
        draw_pipeline(pipeline, config_file[:-5] + ".png")

        respone = pipeline.run(data=dict(example_component=dict(input="Hello World!")))

        print(respone)
    
    if True:
        config_file = Path(__file__).parent.parent.parent / "configs" / "matrix_example.yaml"
        extract_matrix_configuration(config_file)

