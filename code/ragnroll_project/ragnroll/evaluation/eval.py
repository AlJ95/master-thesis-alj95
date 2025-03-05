from ragnroll.metrics import MetricRegistry, BaseMetric
from haystack import Pipeline, Document
from typing import List, Dict, Any, Tuple, Type, Optional, Union
from collections import defaultdict
import logging
import inspect

# Standard-Werte für Klassifikations-Labels
# Diese können später in eine Konfigurationsdatei ausgelagert werden
DEFAULT_POSITIVE_LABEL = "valid"
DEFAULT_NEGATIVE_LABEL = "invalid"

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
    
    def __init__(self, pipeline: Pipeline, positive_label: str = DEFAULT_POSITIVE_LABEL, 
                 negative_label: str = DEFAULT_NEGATIVE_LABEL):
        """
        Initialize with a pipeline to evaluate.
        
        Args:
            pipeline: The pipeline to evaluate
            positive_label: The label to consider as positive class (default: "valid")
            negative_label: The label to consider as negative class (default: "invalid")
        """
        self.pipeline = pipeline
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.end_to_end_metrics = self._instantiate_end_to_end_metrics()
        self.component_metrics = self._instantiate_component_metrics()
        
        logger.info(f"Evaluator initialisiert mit Labels: positiv='{self.positive_label}', negativ='{self.negative_label}'")
    
    def _instantiate_end_to_end_metrics(self) -> Dict[str, BaseMetric]:
        """Create instances of all registered end-to-end metrics."""
        metrics = {}
        
        for name, metric_cls in MetricRegistry.get_end_to_end_metrics().items():
            # Überprüfen, ob es sich um eine Klassifikationsmetrik handelt
            if 'ClassificationBaseMetric' in [cls.__name__ for cls in inspect.getmro(metric_cls)]:
                # Für Klassifikationsmetriken die konfigurierten Label-Werte setzen
                metrics[name] = metric_cls(
                    positive_label=self.positive_label,
                    negative_label=self.negative_label
                )
            else:
                # Für andere Metriken die Standardinitialisierung verwenden
                metrics[name] = metric_cls()
                
        return metrics
    
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
        
        # Generiere Vorhersagen
        dataset.generate_predictions(self.pipeline)
        processed_data = dataset.get_processed_data()
        
        # Run end-to-end evaluations
        end_to_end_results = self._evaluate_end_to_end(processed_data["test_cases"])
        
        # Run component-wise evaluations
        component_results = self._evaluate_components(processed_data["test_cases"])
        
        # Combine results
        results = {
            "end-to-end": end_to_end_results,
            "component-wise": component_results
        }
        
        print_scores(results)
        return results
    
    def _evaluate_end_to_end(self, test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate all end-to-end metrics on the complete set of test cases.
        
        Args:
            test_cases: List of test cases with predictions
            
        Returns:
            Dict[str, float]: Metric name to score mapping
        """
        results = {}
        
        # Extract expected and actual outputs from all test cases
        expected_outputs = [tc["expected_output"] for tc in test_cases]
        actual_outputs = [tc["actual_output"] for tc in test_cases]
        
        # Apply each metric to the entire set of outputs
        for metric_name, metric in self.end_to_end_metrics.items():
            try:
                metric_result = metric.run(
                    expected_outputs=expected_outputs,
                    actual_outputs=actual_outputs
                )
                results[metric_name] = metric_result["score"]
            except Exception as e:
                logger.error(f"Error evaluating with {metric_name}: {e}")
                results[metric_name] = 0.0
        
        return results
    
    def _evaluate_components(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all component metrics on the complete set of test cases.
        
        Args:
            test_cases: List of test cases with predictions
            
        Returns:
            Dict[str, Dict[str, float]]: Component type to metric results mapping
        """
        results = {}
        
        # Group test cases by component
        component_data = defaultdict(lambda: defaultdict(list))
        
        # Gather all data needed for each component
        for test_case in test_cases:
            component_outputs = test_case["component_outputs"]
            
            for component_type in component_outputs:
                component_data[component_type]["component_outputs"].append(component_outputs[component_type])
                component_data[component_type]["expected_outputs"].append(test_case["expected_output"])
                component_data[component_type]["input_texts"].append(test_case["input"])
                component_data[component_type]["queries"].append(test_case["input"])
                
                # Add expected retrieval documents if available
                expected_retrieval = test_case.get("expected_retrieval", None)
                if expected_retrieval:
                    if not all(isinstance(doc, Document) for doc in expected_retrieval):
                        expected_retrieval = [
                            Document(content=doc["content"]) if isinstance(doc, dict) else Document(content=doc)
                            for doc in expected_retrieval
                        ]
                    
                    if "expected_retrievals" not in component_data[component_type]:
                        component_data[component_type]["expected_retrievals"] = []
                    
                    component_data[component_type]["expected_retrievals"].append(expected_retrieval)
        
        # Evaluate each component
        for component_type, component_metrics in self.component_metrics.items():
            if component_type not in component_data:
                continue
                
            results[component_type] = {}
            component_inputs = component_data[component_type]
            
            for metric_name, metric in component_metrics.items():
                try:
                    # Build the parameters for the batch metric
                    metric_params = {
                        "component_outputs": component_inputs["component_outputs"],
                        "expected_outputs": component_inputs["expected_outputs"],
                        "input_texts": component_inputs["input_texts"],
                        "queries": component_inputs["queries"]
                    }
                    
                    # Add expected retrievals if available
                    if "expected_retrievals" in component_inputs:
                        metric_params["expected_retrievals"] = component_inputs["expected_retrievals"]
                    
                    # Run metric
                    metric_result = metric.run(**metric_params)
                    results[component_type][metric_name] = metric_result["score"]
                except Exception as e:
                    logger.error(f"Error evaluating {component_type} with {metric_name}: {e}")
                    results[component_type][metric_name] = 0.0
        
        return results


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
