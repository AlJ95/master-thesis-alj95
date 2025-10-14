"""
Main entry point for the RAGnRoll framework using configuration-based approach.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import asyncio

ENV_PATH = Path(__file__).parent / ".env"

load_dotenv(ENV_PATH)

# Initialize logging first
from ragnroll.utils.logging_config import init_logging
logger = init_logging()

logger.info(f"Langfuse Host: {os.environ.get('LANGFUSE_HOST', 'Not set')}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack import tracing
from haystack_integrations.components.connectors.langfuse import LangfuseConnector

# Add the ragnroll package to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import the necessary modules for the RAGnRoll functionality
try:
    from ragnroll.utils.pipeline import config_to_pipeline, gather_config_paths
    from ragnroll.evaluation.eval import Evaluator, ParallelExecutionStrategy
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

    try:
        import config
    except ImportError as e:
        print(f"Error importing config.py: {e}")
        sys.exit(1)

    # Process the configuration
    process_config(config)

def process_config(config):
    """Process the configuration and execute the evaluation for all profiles."""
    logger.info("Processing configuration...")

    # Check if we have profiles
    if hasattr(config, 'profiles') and isinstance(config.profiles, list):
        logger.info(f"Found {len(config.profiles)} configuration profiles")
        for i, profile in enumerate(config.profiles):
            logger.info(f"Processing profile {i+1}...")
            run_evaluations(profile)
    else:
        logger.info("No profiles found in config. Processing single configuration...")
        # Try to execute as a single config dict
        if hasattr(config, '__dict__'):
            run_evaluations(config.__dict__)
        else:
            logger.error("Invalid configuration format.")

def run_evaluations(profile):
    """Execute the run_evaluations functionality using configuration parameters."""
    logger.info("Running evaluations...")

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
        # Set up MLflow
        if os.getenv("MLFLOW_TRACKING_URI"):
            uri = os.getenv("MLFLOW_TRACKING_URI")
        else:
            uri = "http://localhost:8080"

        # check if uri is accessible
        try:
            mlflow.set_tracking_uri(uri=uri)
        except Exception as e:
            logger.error(f"Failed to set tracking URI: {e}")
            raise ValueError(f"Failed to set tracking URI: {e}")

        eval_data_path = Path(eval_data_file)

        # Split the evaluation data into val, test sets based on Simon et al. (2024)
        # Note: This is a simplified version - in a real implementation, we'd want to
        # check if the data has already been split to avoid errors
        from ragnroll.utils.data import val_test_split
        try:
            val_test_split(eval_data_path, test_size=test_size, random_state=random_state)
        except Exception as e:
            # If data is already split, continue with existing files
            logger.warning(f"Could not split data (may already be split): {e}")

        if not eval_data_path.exists():
            logger.warning(f"Evaluation data path {eval_data_path} does not exist")
            return
        if eval_data_path.is_dir():
            logger.warning(f"Evaluation data path {eval_data_path} is a directory")
            return

        val_data_path = eval_data_path.parent / "val" / eval_data_path.name
        if not val_data_path.exists():
            logger.error(f"Validation data path {val_data_path} does not exist")
            return

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
            logger.info(f"Running evaluation for {config_path}")

            run_name = f"{config_path.parent.name}.{config_path.name}"
            with mlflow.start_run(run_name=run_name):

                # Load and prepare pipelines
                pipeline = config_to_pipeline(configuration_file_path=config_path)
                # validate_pipeline(pipeline)  # Commented out for now to avoid potential issues

                params = extract_run_params(str(config_path))
                params["corpus_dir"] = corpus_dir
                params["val_data_path"] = str(val_data_path)
                import subprocess
                try:
                    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent)
                    params["commit_hash"] = result.stdout.strip() if result.returncode == 0 else "unknown"
                except Exception:
                    params["commit_hash"] = "unknown"

                mlflow.log_params(params)

                pipeline, indexing_duration = index_documents(corpus_dir, pipeline)
                mlflow.log_metrics({"indexing_duration": indexing_duration})

                pipeline.add_component("tracer", LangfuseConnector(run_name))
                data = load_evaluation_data(val_data_path)

                # Always use parallel execution strategy
                execution_strategy = ParallelExecutionStrategy()
                evaluator = Evaluator(
                    pipeline,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    execution_strategy=execution_strategy
                )
                result = evaluator.evaluate(evaluation_data=data, run_name=run_name, track_resources=track_resources)

                fetch_current_traces(run_name)

                for col in result.columns:
                    # Clean the column name for MLflow compatibility
                    clean_col = str(col).replace("(", "").replace(")", "").replace("'", "").replace(",", "_")
                    metric_name = ".".join(("VAL", clean_col))
                    mlflow.log_metrics({metric_name: result[col].values[0]})

        logger.info("Evaluation completed")

    except Exception as e:
        logger.error(f"Error in run_evaluations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
