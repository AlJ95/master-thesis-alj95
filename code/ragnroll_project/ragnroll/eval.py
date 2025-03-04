from ragnroll.metrics import ExactMatchMetric
from haystack import Pipeline
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def generate_dataset(pipeline: Pipeline, evaluation_data: dict) -> Dict[str, Any]:
    """
    Generates the evaluation dataset with predictions from the pipeline
    
    Args:
        pipeline: Haystack pipeline to evaluate
        evaluation_data: Dictionary containing test cases with inputs and expected outputs
        
    Returns:
        Dict[str, Any]: Dataset with predictions and expected values
    """
    dataset = {
        "test_cases": []
    }
    
    for test_case in evaluation_data["test_cases"]:
        try:
            input_text = test_case["input"]
            expected_output = test_case["expected_output"]
            
            # Generate the answer using the pipeline
            response = generate_answer(pipeline, input_text)
            actual_output = extract_answer_from_pipeline(response)
            
            # Add to dataset
            dataset["test_cases"].append({
                "input": input_text,
                "expected_output": expected_output,
                "actual_output": actual_output,
                "component_outputs": response
            })
        except Exception as e:
            logger.error(f"Error generating answer for test case: {e}")
            
    return dataset


def generate_answer(pipeline: Pipeline, input_text: str) -> str:
    """
    Generates an answer using the pipeline
    
    Args:
        pipeline: Haystack pipeline to use
        input_text: Input text/query
        
    Returns:
        str: Generated answer
    """
    components = list(pipeline.to_dict()["components"].keys())
    return pipeline.run(data=dict(query=input_text), include_outputs_from=components)


def extract_answer_from_pipeline(response: Dict[str, Any]) -> str:
    """
    Extracts the answer from the pipeline response
    """
    if "answer_builder" in response:
        return response["answer_builder"]["answer"]
    elif "llm" in response:
        return response["llm"]["replies"][0]
    else:
        raise ValueError(f"Could not extract answer from pipeline response: {response}. Make sure the pipeline has an answer_builder component.")


def evaluate(data: Dict[str, Any], pipeline: Pipeline) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluates the pipeline on the test cases
    
    Args:
        data: Evaluation data
        pipeline: Haystack pipeline to evaluate
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Results for the metrics
    """
    # Generate dataset with predictions
    evaluation_dataset = generate_dataset(pipeline, data)
    
    # Initialize end-to-end metrics
    em_metric = ExactMatchMetric()

    # initialize scores
    scores = {
        "end-to-end": {
            "exact_match": []
        },
        "component-wise": {
            
        }
    }

    # Evaluate each test case for end-to-end metrics
    for test_case in evaluation_dataset["test_cases"]:
        # Evaluate exact match
        em_result = em_metric.run(
            expected_output=test_case["expected_output"],
            actual_output=test_case["actual_output"]
        )
        scores["end-to-end"]["exact_match"].append(em_result["score"])

    calculated_scores = calculate_scores(scores)
    print_scores(calculated_scores)
    return calculated_scores


def calculate_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates the scores for the metrics
    """
    return {
        "end-to-end": {
            "exact_match": sum(scores["end-to-end"]["exact_match"]) / len(scores["end-to-end"]["exact_match"])
        },
        "component-wise": {
        }
    }

def print_scores(scores: Dict[str, Any]) -> None:
    """
    Prints the scores for the metrics
    """
    print("===== Evaluation Results =====")
    print("=== End-to-End Metrics ===")
    for metric, score in scores["end-to-end"].items():
        print(f"{metric}: {score:.4f}")
    print("=== Component-Wise Metrics ===")
    for component, metrics in scores["component-wise"].items():
        print(f"{component}:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.4f}")
