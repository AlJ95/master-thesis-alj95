from haystack import Pipeline
from haystack.core.component import InputSocket, OutputSocket
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple
from .components import *
import haystack.dataclasses

def config_to_pipeline(configuration_file_path: str) -> Pipeline:
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


if __name__ == "__main__":
    config_file = "../configs/example_component.yaml"
    
    import os
    assert os.path.exists(config_file), "Path does not exist"

    pipeline = config_to_pipeline(config_file)
    draw_pipeline(pipeline, config_file[:-5] + ".png")

    respone = pipeline.run(data=dict(example_component=dict(input="Hello World!")))

    print(respone)