from pathlib import Path
from datasets import load_dataset
from warnings import warn

from h11 import Data

def load_data_from_directory(data_path: str):
    """
    Loads data as single files from a directory
    """
    data_path = Path(data_path)

    label_dirs = list(data_path.iterdir())

    all_files = []
    for label_dir in label_dirs:
        all_files.extend(label_dir.iterdir())

    # Transforming .c, .py, ... files to txt files
    for f in all_files:
        if f.suffix not in [".csv", ".json"]:
            f.rename(f.parent / (f.stem + ".txt"))
        else:
            warn(f"Multiple data files are not supported right now. Skipping {f.name}")
    
    data = load_dataset(path=str(data_path))

    # for label_dir in label_dirs:
    #     for f in label_dir.iterdir():
    #         text = open(f).read()
    #         label = label_dir.name
    #         data.add_item(dict(text=text, label=label))

    return data


if __name__ == "__main__":
    path = "/Users/alj95/Projekte/master-thesis-alj95/code/ragnroll_project/data/se_vulnerabilities_test"
    data = load_data_from_directory(path)
    print(data)
