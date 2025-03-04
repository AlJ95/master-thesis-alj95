from ragnroll.metrics import MetricRegistry, BaseMetric
from haystack import Pipeline, Document
from typing import List, Dict, Any, Tuple, Type, Optional, Union
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EvaluationDataset:
    """Class to handle evaluation datasets."""
    
    def __init__(self, evaluation_data: Dict[str, Any]):
        """Initialize with raw evaluation data."""
        self.evaluation_data = evaluation_data
        self.processed_data = {
            "test_cases": []
        }
    
    def generate_predictions(self, pipeline: Pipeline) -> None:
        """Generate predictions for all test cases."""
        for test_case in self.evaluation_data["test_cases"]:
            try:
                input_text = test_case["input"]
                expected_output = test_case["expected_output"]
                
                # Get ground truth documents if available
                expected_retrieval = test_case.get("expected_retrieval", None)
                
                # Generate the answer using the pipeline
                response = self._generate_answer(pipeline, input_text)
                actual_output = self._extract_answer_from_pipeline(response)
                
                # Add to dataset
                self.processed_data["test_cases"].append({
                    "input": input_text,
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                    "component_outputs": response,
                    "expected_retrieval": expected_retrieval
                })
            except Exception as e:
                logger.error(f"Error generating answer for test case: {e}")
    
    def _generate_answer(self, pipeline: Pipeline, input_text: str) -> Dict[str, Any]:
        """Generate an answer using the pipeline."""
        components = list(pipeline.to_dict()["components"].keys())
        return pipeline.run(data=dict(query=input_text), include_outputs_from=components)
    
    def _extract_answer_from_pipeline(self, response: Dict[str, Any]) -> str:
        """Extract the answer from the pipeline response."""
        if "answer_builder" in response:
            return response["answer_builder"]["answer"]
        elif "llm" in response:
            return response["llm"]["replies"][0]
        else:
            raise ValueError(f"Could not extract answer from pipeline response: {response}. Make sure the pipeline has an answer_builder component.")
    
    def get_processed_data(self) -> Dict[str, Any]:
        """Get the processed data with predictions."""
        return self.processed_data


class Evaluator:
    """Main evaluator class for running metrics on evaluation data."""
    
    def __init__(self, pipeline: Pipeline):
        """Initialize with a pipeline to evaluate."""
        self.pipeline = pipeline
        self.end_to_end_metrics = self._instantiate_end_to_end_metrics()
        self.component_metrics = self._instantiate_component_metrics()
    
    def _instantiate_end_to_end_metrics(self) -> Dict[str, BaseMetric]:
        """Create instances of all registered end-to-end metrics."""
        return {
            name: metric_cls() 
            for name, metric_cls in MetricRegistry.get_end_to_end_metrics().items()
        }
    
    def _instantiate_component_metrics(self) -> Dict[str, Dict[str, BaseMetric]]:
        """Create instances of component metrics relevant to this pipeline."""
        pipeline_components = set(self.pipeline.to_dict()["components"].keys())
        metrics = {}
        
        for component_type, metric_classes in MetricRegistry.get_component_metrics().items():
            # Only include metrics for components that exist in the pipeline
            if any(comp.startswith(component_type) for comp in pipeline_components):
                metrics[component_type] = {
                    name: metric_cls() for name, metric_cls in metric_classes.items()
                }
        
        return metrics
    
    def evaluate(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the evaluation on the provided data.
        
        Args:
            evaluation_data: Test cases to evaluate
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Generate dataset with predictions
        dataset = EvaluationDataset(evaluation_data)
        dataset.generate_predictions(self.pipeline)
        processed_data = dataset.get_processed_data()
        
        # Initialize scores structure
        scores = {
            "end-to-end": defaultdict(list),
            "component-wise": defaultdict(lambda: defaultdict(list))
        }
        
        # Evaluate each test case for end-to-end metrics
        for test_case in processed_data["test_cases"]:
            for metric_name, metric in self.end_to_end_metrics.items():
                metric_result = metric.run(
                    expected_output=test_case["expected_output"],
                    actual_output=test_case["actual_output"]
                )
                scores["end-to-end"][metric_name].append(metric_result["score"])
            
            # Component-wise metrics
            component_outputs = test_case["component_outputs"]
            expected_retrieval = test_case.get("expected_retrieval", None)
            query = test_case["input"]
            
            # Convert expected_retrieval to Document objects if they're not already
            if expected_retrieval and not all(isinstance(doc, Document) for doc in expected_retrieval):
                expected_retrieval = [
                    Document(content=doc["content"]) if isinstance(doc, dict) else Document(content=doc)
                    for doc in expected_retrieval
                ]
            
            for component_type, component_metrics in self.component_metrics.items():
                if component_type in component_outputs:
                    for metric_name, metric in component_metrics.items():
                        try:
                            # Build the parameters for the metric
                            metric_params = {
                                "component_output": component_outputs[component_type],
                                "expected_output": test_case["expected_output"],
                                "input_text": test_case["input"],
                                "query": query
                            }
                            
                            # Add ground truth documents if available
                            if expected_retrieval:
                                metric_params["expected_retrieval"] = expected_retrieval
                            
                            metric_result = metric.run(**metric_params)
                            scores["component-wise"][component_type][metric_name].append(metric_result["score"])
                        except Exception as e:
                            logger.error(f"Error evaluating {component_type} with {metric_name}: {e}")
        
        calculated_scores = self._calculate_scores(scores)
        print_scores(calculated_scores)
        return calculated_scores
    
    def _calculate_scores(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregated scores from individual test case scores."""
        result = {
            "end-to-end": {},
            "component-wise": {}
        }
        
        # Calculate end-to-end metric averages
        for metric_name, scores_list in scores["end-to-end"].items():
            if scores_list:
                result["end-to-end"][metric_name] = sum(scores_list) / len(scores_list)
        
        # Calculate component-wise metric averages
        for component_type, metrics in scores["component-wise"].items():
            result["component-wise"][component_type] = {}
            for metric_name, scores_list in metrics.items():
                if scores_list:
                    result["component-wise"][component_type][metric_name] = sum(scores_list) / len(scores_list)
        
        return result


def evaluate(data: Dict[str, Any], pipeline: Pipeline) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluates the pipeline on the test cases.
    
    Args:
        data: Evaluation data
        pipeline: Haystack pipeline to evaluate
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Results for the metrics
    """
    evaluator = Evaluator(pipeline)
    scores = evaluator.evaluate(data)
    return scores, evaluator.component_metrics


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
