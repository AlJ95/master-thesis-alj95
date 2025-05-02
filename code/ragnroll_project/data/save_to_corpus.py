import os
import shutil
from pathlib import Path

def copy_html_files():
    # Get script location
    script_dir = Path(__file__).parent
    
    # Define source and target directories
    source_dir = script_dir / "raw" / "clean_html"
    target_dir = script_dir / "processed" / "config_val" / "corpus_filtered_cleaned"
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Walk through source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.html'):
                # Get source file path
                source_file = Path(root) / file
                
                # Create target file path
                target_file = target_dir / file
                
                # Copy file
                shutil.copy2(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")

if __name__ == "__main__":
    copy_html_files()
