import os
from pathlib import Path
from dotenv import load_dotenv

def test_env_loading():
    # Load the .env file
    load_dotenv(Path(__file__).parent.parent / ".env")
    
    # Check if a specific environment variable is loaded
    assert os.getenv("OPENAI_API_KEY") is not None, "Environment variable not loaded" 