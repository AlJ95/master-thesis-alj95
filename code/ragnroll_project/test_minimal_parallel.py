"""
Minimal test file for parallel execution functionality.
This file tests the new parallel execution strategy with a small dataset.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def create_minimal_test_data():
    """Create a minimal test dataset for parallel execution testing."""
    test_data = {
        "test_cases": [
            {
                "input": "What is the capital of France?",
                "expected_output": "valid"
            },
            {
                "input": "What is 2 + 2?",
                "expected_output": "valid"
            },
            {
                "input": "Tell me a joke.",
                "expected_output": "invalid"
            }
        ]
    }
    return test_data

async def test_sequential_execution():
    """Test that sequential execution still works."""
    try:
        from ragnroll.evaluation.eval import Evaluator, EvaluationDataset
        from ragnroll.utils.pipeline import config_to_pipeline
        from ragnroll.utils.ingestion import index_documents
        
        # Load a simple pipeline configuration
        config_path = Path("configs/examples/predefined.yaml")
        if not config_path.exists():
            print("Warning: predefined.yaml not found, using sample.yaml")
            config_path = Path("configs/from_pipeline/sample.yaml")
        
        pipeline = config_to_pipeline(config_path)
        
        # Create minimal corpus for indexing
        corpus_dir = Path("data/processed/dev_data/corpus")
        if corpus_dir.exists():
            pipeline, _ = index_documents(str(corpus_dir), pipeline)
        
        # Create test data
        test_data = create_minimal_test_data()
        
        # Test sequential evaluation
        evaluator = Evaluator(pipeline)
        dataset = EvaluationDataset(test_data)
        dataset.generate_predictions(pipeline)
        
        print("✓ Sequential execution works with minimal data")
        return True
        
    except Exception as e:
        print(f"✗ Sequential execution failed: {e}")
        return False

def test_parallel_execution_import():
    """Test that parallel execution classes can be imported."""
    try:
        from ragnroll.evaluation.eval import ParallelExecutionStrategy
        print("✓ Parallel execution classes can be imported")
        return True
    except ImportError as e:
        print(f"✗ Parallel execution classes import failed: {e}")
        return False

def test_parallel_execution_strategy():
    """Test that parallel execution strategy can be created and used."""
    try:
        from ragnroll.evaluation.eval import ParallelExecutionStrategy

        # Test parallel strategy creation
        parallel_strategy = ParallelExecutionStrategy(num_processes=2)
        assert parallel_strategy is not None
        print("✓ Parallel execution strategy created successfully")

        return True
    except Exception as e:
        print(f"✗ Parallel execution strategy test failed: {e}")
        return False

async def test_parallel_evaluation():
    """Test parallel evaluation with minimal data."""
    try:
        from ragnroll.evaluation.eval import Evaluator, EvaluationDataset, ParallelExecutionStrategy
        from ragnroll.utils.pipeline import config_to_pipeline
        from ragnroll.utils.ingestion import index_documents

        # Load a simple pipeline configuration
        config_path = Path("configs/examples/predefined.yaml")
        if not config_path.exists():
            print("Warning: predefined.yaml not found, using sample.yaml")
            config_path = Path("configs/from_pipeline/sample.yaml")

        pipeline = config_to_pipeline(config_path)

        # Create minimal corpus for indexing
        corpus_dir = Path("data/processed/dev_data/corpus")
        if corpus_dir.exists():
            pipeline, _ = index_documents(str(corpus_dir), pipeline)

        # Create test data
        test_data = create_minimal_test_data()

        # Test parallel evaluation
        execution_strategy = ParallelExecutionStrategy(num_processes=2)
        evaluator = Evaluator(pipeline, execution_strategy=execution_strategy)
        dataset = EvaluationDataset(test_data, execution_strategy)
        dataset.generate_predictions(pipeline)

        print("✓ Parallel evaluation works with minimal data")
        return True

    except Exception as e:
        print(f"✗ Parallel evaluation failed: {e}")
        return False

async def main():
    print("Testing Minimal Parallel Execution Setup")
    print("=" * 40)
    
    success = True
    success &= await test_sequential_execution()
    success &= test_parallel_execution_import()
    success &= test_parallel_execution_strategy()
    success &= await test_parallel_evaluation()
    
    if success:
        print("\n✓ All minimal tests passed! Parallel implementation is ready.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
