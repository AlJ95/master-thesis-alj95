"""
Configuration file for the RAGnRoll framework.
This file contains all the settings and parameters for the evaluation.
"""

# Comprehensive configuration structure with all parameters needed for evaluation
profiles = [
    {
        # Basic configuration
        "name": "minimal_test_profile",
        "experiment_name": "Minimal Test Experiment",
        
        # Data paths
        "config_sources": "configs/from_pipeline/sample.yaml",
        "eval_data_file": "data/processed/dev_data/synthetic_rag_evaluation.json",
        "corpus_dir": "data/processed/dev_data/corpus",
        
        # Evaluation parameters
        "track_resources": True,
        "baselines": True,
        "test_size": 10,
        "random_state": 42,
        "positive_label": "valid",
        "negative_label": "invalid",
    }
]
