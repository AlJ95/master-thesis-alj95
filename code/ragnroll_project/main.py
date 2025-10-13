"""
Main entry point for the RAGnRoll framework using configuration-based approach.
"""

import sys
import os
from pathlib import Path

# Add the ragnroll package to the path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point for the RAGnRoll framework."""
    print("RAGnRoll Framework - Configuration-Based Approach")
    print("=" * 50)
    
    # Check if config.py exists
    config_path = Path("config.py")
    if not config_path.exists():
        print("Error: config.py not found!")
        print("Please create a config.py file with your configuration settings.")
        sys.exit(1)
    
    # Import the configuration
    try:
        import config
    except ImportError as e:
        print(f"Error importing config.py: {e}")
        sys.exit(1)
    
    # Process the configuration
    process_config(config)

def process_config(config):
    """Process the configuration and execute the appropriate functions."""
    print("Processing configuration...")
    
    # Check if we have profiles
    if hasattr(config, 'profiles') and isinstance(config.profiles, list):
        print(f"Found {len(config.profiles)} configuration profiles")
        for i, profile in enumerate(config.profiles):
            print(f"Processing profile {i+1}...")
            execute_profile(profile)
    else:
        print("No profiles found in config. Processing single configuration...")
        # Try to execute as a single config dict
        if hasattr(config, '__dict__'):
            execute_profile(config.__dict__)
        else:
            print("Invalid configuration format.")

def execute_profile(profile):
    """Execute a single configuration profile."""
    # This is where we would call the actual evaluation functions
    # For now, we'll just print the profile settings
    print(f"Profile settings:")
    
    # Handle both object and dict configurations
    if isinstance(profile, dict):
        for key, value in profile.items():
            print(f"  {key}: {value}")
    else:
        for key, value in profile.__dict__.items():
            if not key.startswith('__'):
                print(f"  {key}: {value}")
    print()

if __name__ == "__main__":
    main()
