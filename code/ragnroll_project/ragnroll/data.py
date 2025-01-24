from pathlib import Path
from datasets import load_dataset
from warnings import warn
from deepeval.test_case import LLMTestCase

def load_data_from_directory(data_path: str):
    """
    Loads data as single files from a directory
    """
    data_path = Path(data_path)

    all_files = list(data_path.iterdir())

    # Transforming .c, .py, ... files to txt files
    for f in all_files:
        if f.suffix not in [".csv", ".json"]:
            raise ValueError(f"File {f} is not a CSV or JSON file. Please keep only relevant JSON or CSV files in the directory.")
        
    data = load_dataset(path=str(data_path))

    if "label" not in data["validation"].features.keys():
        raise ValueError("Label column not found in the dataset. Please make sure that the label column is named 'label'.")
    
    if "content" not in data["validation"].features.keys():
        warn("Content column not found in the dataset. Please make sure that the content column is named 'content'.")

    return data

if __name__ == "__main__":
    path = "C:/Users/Besitzer/Projekte/master-thesis-alj95/code/ragnroll_project/data/val/"
    data = load_data_from_directory(path)
    print(data.data)
