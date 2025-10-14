"""
Configuration file for the RAGnRoll framework.
This file contains all the settings and parameters for the evaluation.
"""

# Comprehensive configuration structure with all parameters needed for evaluation
profiles = [
    {
        # Basic configuration
        "name": "parallel_test_profile",
        "experiment_name": "Parallel Execution Test",
        
        # Data paths
        "config_sources": "configs/examples/predefined.yaml",
        "eval_data_file": "data/processed/dev_data/synthetic_rag_evaluation.json",
        "corpus_dir": "data/processed/dev_data/corpus",
        
        # Evaluation parameters
        "track_resources": False,
        "baselines": False,
        "test_size": 10,
        "random_state": 42,
        "positive_label": "valid",
        "negative_label": "invalid",
        
        # Parallel execution configuration
        "data_parallel_execution": True,
        "num_data_processes": 12,
        "parallel_chunk_size": 50,
    }
]
