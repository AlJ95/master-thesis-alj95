from pathlib import Path
from warnings import warn
import json
import pandas as pd

def load_evaluation_data(data_path: str):
    """
    Loads data from a directory containing datasets or directly from a single file
    """
    data_path = Path(data_path)
    
    # Check if the path is a file or directory
    if data_path.is_file():
        # Direct file handling
        if data_path.suffix.lower() not in [".csv", ".json"]:
            raise ValueError(f"File {data_path} is not a CSV or JSON file.")
        
        if data_path.suffix.lower() == ".json":
            with open(data_path, "r") as f:
                data = json.load(f)
        else:
            data = pd.read_csv(data_path).to_dict(orient="records")
    else:
        raise ValueError(f"Path {data_path} is a directory.")

    if all("expected_output" not in data["test_cases"][i].keys() 
           for i in range(len(data["test_cases"]))
           ):
        raise ValueError("expected_output column not found in the dataset. Please make sure that the expected_output column is named 'expected_output'.")
    
    if all("expected_retrieval" not in data["test_cases"][i].keys() 
           for i in range(len(data["test_cases"]))
           ):
        warn("expected_retrieval column not found in the dataset. Please make sure that the expected_retrieval column is named 'expected_retrieval'.")

    return data

if __name__ == "__main__":
    path = "C:/Users/Besitzer/Projekte/master-thesis-alj95/code/ragnroll_project/data/val/"
    data = load_evaluation_data(path)
    print(data)
