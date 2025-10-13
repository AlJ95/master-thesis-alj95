"""
Configuration file for the RAGnRoll framework.
This file contains all the settings and parameters for the evaluation.
"""

# Comprehensive configuration structure with all CLI parameters
profiles = [
    {
        # Basic configuration
        "name": "example_profile",
        "experiment_name": "RAG Experimentation",
        
        # Data paths
        "config_sources": "configs/from_pipeline/sample.yaml",
        "eval_data_file": "data/processed/dev_data/synthetic_rag_evaluation.json",
        "corpus_dir": "data/processed/dev_data/corpus",
        
        # Evaluation parameters
        "track_resources": True,
        "baselines": True,
        "test_size": 20,
        "random_state": 42,
        "positive_label": "valid",
        "negative_label": "invalid",
        
        # Test generalization error parameters
        "run_id": None,  # Optional
        "strict": True,
        
        # Split data parameters
        "split_data_path": "data/raw",  # Path for split_data command
        "split_test_size": 20,  # Percentage for test set
        
        # Draw pipeline parameters
        "draw_config_file": "configs/from_pipeline/sample.yaml",
        "draw_output_file": "pipeline.png",
        
        # Mode control - determines which function to execute
        # Options: "run_evaluations", "split_data", "test_generalization_error", "draw_pipeline"
        "mode": "run_evaluations"
    }
]
