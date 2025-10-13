"""
Main entry point for the RAGnRoll framework using configuration-based approach.
"""

import sys
import os
from pathlib import Path

# Add the ragnroll package to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import the necessary modules for the RAGnRoll functionality
# This is a simplified version - in practice, we'll need to import the actual functions
try:
    from ragnroll.utils.data import val_test_split
    from ragnroll.utils.pipeline import config_to_pipeline, draw_pipeline, gather_config_paths
    from ragnroll.evaluation.eval import Evaluator
    from ragnroll.evaluation.data import load_evaluation_data
    from ragnroll.utils.ingestion import index_documents
    from ragnroll.evaluation.tracing import fetch_current_traces
    from ragnroll.utils.config import extract_run_params
    import mlflow
    import warnings
    from pathlib import Path
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

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
    print(f"Profile settings:")
    
    # Handle both object and dict configurations
    if isinstance(profile, dict):
        for key, value in profile.items():
            print(f"  {key}: {value}")
    else:
        for key, value in profile.__dict__.items():
            if not key.startswith('__'):
                print(f"  {key}: {value}")
    
    # Determine which function to execute based on the mode
    mode = profile.get('mode', 'run_evaluations')
    print(f"Executing mode: {mode}")
    
    if mode == 'run_evaluations':
        run_evaluations(profile)
    elif mode == 'split_data':
        split_data(profile)
    elif mode == 'test_generalization_error':
        test_generalization_error(profile)
    elif mode == 'draw_pipeline':
        draw_pipeline(profile)
    else:
        print(f"Unknown mode: {mode}")

def run_evaluations(profile):
    """Execute the run_evaluations functionality using configuration parameters."""
    print("Running evaluations...")
    
    # Extract parameters from profile
    config_sources = profile.get('config_sources', 'configs/from_pipeline/sample.yaml')
    eval_data_file = profile.get('eval_data_file', 'data/processed/dev_data/synthetic_rag_evaluation.json')
    corpus_dir = profile.get('corpus_dir', 'data/processed/dev_data/corpus')
    track_resources = profile.get('track_resources', True)
    baselines = profile.get('baselines', True)
    test_size = profile.get('test_size', 20)
    random_state = profile.get('random_state', 42)
    positive_label = profile.get('positive_label', 'valid')
    negative_label = profile.get('negative_label', 'invalid')
    experiment_name = profile.get('experiment_name', 'RAG Experimentation')
    
    try:
        # Import required modules
        from ragnroll.utils.pipeline import gather_config_paths, config_to_pipeline, validate_pipeline
        from ragnroll.evaluation.eval import Evaluator
        from ragnroll.evaluation.data import load_evaluation_data
        from ragnroll.utils.ingestion import index_documents
        from ragnroll.evaluation.tracing import fetch_current_traces
        from ragnroll.utils.config import extract_run_params
        import mlflow
        import warnings
        from pathlib import Path
        
        # Set up MLflow
        if os.getenv("MLFLOW_TRACKING_URI"):
            uri = os.getenv("MLFLOW_TRACKING_URI")
        else:
            uri = "http://localhost:8080"

        # check if uri is accessible
        try:
            mlflow.set_tracking_uri(uri=uri)
        except Exception as e:
            raise ValueError(f"Failed to set tracking URI: {e}")

        eval_data_path = Path(eval_data_file)

        # Split the evaluation data into val, test sets based on Simon et al. (2024) 
        val_test_split(eval_data_path, test_size=test_size, random_state=random_state)

        if not eval_data_path.exists():
            warnings.warn(f"Evaluation data path {eval_data_path} does not exist")
            return
        if eval_data_path.is_dir():
            warnings.warn(f"Evaluation data path {eval_data_path} is a directory")
            return

        val_data_path = eval_data_path.parent / "val" / eval_data_path.name
        assert val_data_path.exists(), f"Validation data path {val_data_path} does not exist"

        # Setup Run-ID
        mlflow.set_experiment(experiment_name=experiment_name)

        # Prepare baseline paths
        BASELINES_PATH = Path(__file__).parent / "configs" / "baselines"
        if baselines:
            baseline_paths = [
                BASELINES_PATH / "llm_config.yaml", 
                BASELINES_PATH / "predefined_bm25.yaml"
            ]
        else:
            baseline_paths = []

        # Gather all config paths from the source file (YAML, MATRIX-YAML, PYTHON)
        config_sources = gather_config_paths(Path(config_sources))

        for config_path in baseline_paths + config_sources:
            print(f"Running evaluation for {config_path}")

            run_name = f"{config_path.parent.name}.{config_path.name}"
            with mlflow.start_run(run_name=run_name):
                
                # Load and prepare pipelines
                pipeline = config_to_pipeline(configuration_file_path=config_path)
                validate_pipeline(pipeline)

                params = extract_run_params(config_path)
                params["corpus_dir"] = corpus_dir
                params["val_data_path"] = val_data_path
                params["commit_hash"] = os.system("git rev-parse HEAD")
                
                mlflow.log_params(params)

                pipeline, indexing_duration = index_documents(corpus_dir, pipeline)
                mlflow.log_metrics({"indexing_duration": indexing_duration})
                
                # pipeline.add_component("tracer", LangfuseConnector(run_name))
                data = load_evaluation_data(val_data_path)

                evaluator = Evaluator(pipeline, positive_label=positive_label, negative_label=negative_label)
                result = evaluator.evaluate(evaluation_data=data, run_name=run_name, track_resources=track_resources)

                fetch_current_traces(run_name)
                
                for col in result.columns:
                    metric_name = ".".join(("VAL",) + col)
                    mlflow.log_metrics({metric_name: result[col].values[0]})

        print("Evaluation completed")
        
    except Exception as e:
        print(f"Error in run_evaluations: {e}")
        import traceback
        traceback.print_exc()

def split_data(profile):
    """Execute the split_data functionality using configuration parameters."""
    print("Splitting data...")
    
    # Extract parameters from profile
    path = profile.get('split_data_path', 'data/raw')
    test_size = profile.get('split_test_size', 20)
    random_state = profile.get('random_state', 42)
    
    try:
        from ragnroll.utils.data import val_test_split
        from pathlib import Path
        
        eval_data_path = Path(path)
        val_test_split(eval_data_path, test_size=test_size, random_state=random_state)
        print(f"Successfully split data into train/val/test sets in {path}")
        
    except Exception as e:
        print(f"Error in split_data: {e}")
        import traceback
        traceback.print_exc()

def test_generalization_error(profile):
    """Execute the test_generalization_error functionality using configuration parameters."""
    print("Testing generalization error...")
    
    # Extract parameters from profile
    eval_data_file = profile.get('eval_data_file', 'data/processed/dev_data/synthetic_rag_evaluation.json')
    corpus_dir = profile.get('corpus_dir', 'data/processed/dev_data/corpus')
    experiment_name = profile.get('experiment_name', 'RAG Experimentation')
    run_id = profile.get('run_id', None)
    strict = profile.get('strict', True)
    positive_label = profile.get('positive_label', 'valid')
    negative_label = profile.get('negative_label', 'invalid')
    
    try:
        import mlflow
        from ragnroll.utils.pipeline import config_to_pipeline, validate_pipeline
        from ragnroll.utils.ingestion import index_documents
        from ragnroll.evaluation.eval import Evaluator
        from ragnroll.evaluation.data import load_evaluation_data
        from ragnroll.evaluation.tracing import fetch_current_traces
        from pathlib import Path
        
        eval_data_path = Path(eval_data_file)
        test_data_path = eval_data_path.parent / "test" / eval_data_path.name

        if os.getenv("MLFLOW_TRACKING_URI"):
            uri = os.getenv("MLFLOW_TRACKING_URI")
        else:
            uri = "http://localhost:8080"

        # check if uri is accessible
        try:
            mlflow.set_tracking_uri(uri=uri)
        except Exception as e:
            raise ValueError(f"Failed to set tracking URI: {e}")
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            mlflow.set_experiment(experiment_id=experiment.experiment_id)
        else:
            raise ValueError(f"Experiment {experiment_name} not found")

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if run_id:
            runs = runs[runs['run_id'] == run_id]
        
        if "params.used_test_sets" in runs.columns and strict:
            # Check if the testset path empty
            runs = runs[runs['params.used_test_sets'].isna()]
        
        if runs.empty:
            raise ValueError(f"No runs found for experiment {experiment_name} ({run_id if run_id else 'all runs'}). Create a new evaluation dataset or use --no-strict (not recommended)")

        for _, run in runs.iterrows():
            with mlflow.start_run(run_id=run.run_id):
                run_name = run["tags.mlflow.runName"]

                pipeline = config_to_pipeline(configuration_dict=eval(run["params.config"]))
                validate_pipeline(pipeline)

                pipeline, indexing_duration = index_documents(corpus_dir, pipeline)
                # pipeline.add_component("tracer", LangfuseConnector(run_name))
                data = load_evaluation_data(test_data_path)

                evaluator = Evaluator(pipeline, positive_label=positive_label, negative_label=negative_label)
                result = evaluator.evaluate(evaluation_data=data, run_name=run_name, track_resources=False)

                fetch_current_traces(run_name)
                
                for col in result.columns:
                    metric_name = ".".join(("TEST",) + col)
                    mlflow.log_metrics({metric_name: result[col].values[0]})
                
                mlflow.log_param("used_test_sets", str(test_data_path))

        print("Evaluation completed")
        
    except Exception as e:
        print(f"Error in test_generalization_error: {e}")
        import traceback
        traceback.print_exc()

def draw_pipeline(profile):
    """Execute the draw_pipeline functionality using configuration parameters."""
    print("Drawing pipeline...")
    
    # Extract parameters from profile
    config_file = profile.get('draw_config_file', 'configs/from_pipeline/sample.yaml')
    output_file = profile.get('draw_output_file', 'pipeline.png')
    
    try:
        from ragnroll.utils.pipeline import config_to_pipeline, draw_pipeline
        from pathlib import Path
        
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"Error: Config file not found at {config_path}")
            return
            
        pipeline = config_to_pipeline(configuration_file_path=config_path)
        draw_pipeline(pipeline, output_file)
        print(f"Pipeline drawn successfully to {output_file}")
        
    except Exception as e:
        print(f"Error in draw_pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
