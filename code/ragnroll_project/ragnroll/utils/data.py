from sklearn.model_selection import train_test_split
import pandas as pd
import json
from pathlib import Path

import typer

def val_test_split(path: str = "data/processed", test_size: float = 20, random_state: int = 42):
    """
    Split data into validation and test sets from JSON/CSV files.
    Creates two directories: val, test
    """
    path = Path(path)

    # Validate input parameters
    if not (0 < test_size < 100):
        raise typer.BadParameter("test_size must be between 0 and 100")

    # Process files
    if path.is_dir():
        # Create output directories
        val_path = path / "val" 
        test_path = path / "test"
        val_path.mkdir(exist_ok=True)
        test_path.mkdir(exist_ok=True)

        # Get only JSON/CSV files
        files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in ['.json', '.csv']]
        
        if not files:
            raise typer.BadParameter("No JSON or CSV files found in directory")
            
        # Process each file
        for file in files:
            # Load data
            if file.suffix.lower() == '.json':
                with open(file) as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:  # CSV
                df = pd.read_csv(file)
                
            # Split data
            val_df, test_df = train_test_split(
                df,
                test_size=test_size/100,
                random_state=random_state
            )
            
            # Save splits
            base_name = file.stem
            with open(val_path / f"{base_name}_val.json", 'w') as f:
                json.dump({"test_cases": [test_case for test_case in val_df.test_cases.values]}, f)
            with open(test_path / f"{base_name}_test.json", 'w') as f:
                json.dump({"test_cases": [test_case for test_case in test_df.test_cases.values]}, f)
