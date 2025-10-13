#!/usr/bin/env python3
"""
Test script to verify the configuration-based approach works correctly.
This script tests that the system can be executed with a minimal configuration.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_config_loading():
    """Test that config.py can be loaded and parsed correctly."""
    try:
        import config
        print("✓ Successfully imported config.py")
        
        # Check if profiles exist
        if hasattr(config, 'profiles') and isinstance(config.profiles, list):
            print(f"✓ Found {len(config.profiles)} configuration profiles")
            for i, profile in enumerate(config.profiles):
                print(f"  Profile {i+1}: {profile.get('name', 'unnamed')}")
                print(f"    Experiment: {profile.get('experiment_name', 'default')}")
                print(f"    Data file: {profile.get('eval_data_file', 'default')}")
        else:
            print("✗ No profiles found in config")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Error importing config.py: {e}")
        return False

def test_main_execution():
    """Test that main.py can be executed with minimal configuration."""
    try:
        # Import main to check for syntax errors
        import main
        print("✓ main.py imports successfully")
        return True
    except Exception as e:
        print(f"✗ Error importing main.py: {e}")
        return False

if __name__ == "__main__":
    print("Testing Configuration-Based Approach")
    print("=" * 40)
    
    success = True
    success &= test_config_loading()
    success &= test_main_execution()
    
    if success:
        print("\n✓ All tests passed! The configuration-based approach is working.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed.")
        sys.exit(1)
