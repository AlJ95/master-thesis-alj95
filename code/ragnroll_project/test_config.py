"""
Test configuration file for the RAGnRoll framework.
This file contains a minimal configuration for testing the new system.
"""

# Minimal test configuration
profiles = [
    {
        # Basic configuration
        "name": "test_profile",
        "experiment_name": "Test Experiment",
        
        # Data paths - using minimal test data
        "config_sources": "configs/examples/predefined.yaml",
        "eval_data_file": "data/processed/dev_data/synthetic_rag_evaluation.json",
        "corpus_dir": "data/processed/dev_data/corpus",
        
        # Evaluation parameters
        "track_resources": False,  # Disable resource tracking for test
        "baselines": False,  # Skip baselines for faster testing
        "test_size": 10,  # Smaller test size for testing
        "random_state": 42,
        "positive_label": "valid",
        "negative_label": "invalid",
        
        # Test generalization error parameters
        "run_id": None,
        "strict": True,
        
        # Split data parameters
        "split_data_path": "data/raw",  # Path for split_data command
        "split_test_size": 10,  # Percentage for test set
        
        # Draw pipeline parameters
        "draw_config_file": "configs/examples/predefined.yaml",
        "draw_output_file": "test_pipeline.png",
        
        # Mode control - determines which function to execute
        "mode": "run_evaluations"
    }
]
