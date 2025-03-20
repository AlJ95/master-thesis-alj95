from ragnroll.metrics import MetricRegistry, BaseMetric
from haystack import Pipeline, Document
from typing import List, Dict, Any, Tuple, Type, Optional, Union
from collections import defaultdict
import logging
import inspect
import pandas as pd
import os

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
        pipeline_components = self.pipeline.to_dict()["components"]
        metrics = {}
        
        # Map component names to their types from the pipeline
        component_types = {}
        for comp_name, comp_details in pipeline_components.items():
            comp_type = comp_details.get("type", "")
            if ".generator." in comp_type:
                component_types[comp_name] = "generator"
            elif ".retriever." in comp_type:
                component_types[comp_name] = "retriever"
            else:
                # Fallback to component name-based detection for compatibility
                if comp_name.startswith("retriever"):
                    component_types[comp_name] = "retriever"
                elif comp_name.startswith("generator") or comp_name.startswith("llm"):
                    component_types[comp_name] = "generator"
        
        for component_type, metric_classes in MetricRegistry.get_component_metrics().items():
            # Only include metrics for components that exist in the pipeline
            if component_type in component_types.values():
                metrics[component_type] = {
                    name: metric_cls() for name, metric_cls in metric_classes.items()
                }
                
        return metrics
    
    def evaluate(self, evaluation_data: Dict[str, Any], run_id: str) -> pd.DataFrame:
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

        # Convert results to pandas DataFrames
        results_df = self._results_to_df(end_to_end_results, component_results, run_id)
        
        return results_df

    def _results_to_df(self, end_to_end_results: Dict[str, float], component_results: Dict[str, Dict[str, float]], run_id: str) -> pd.DataFrame:
        """
        Convert results to pandas DataFrames
        """
        if "generator" in component_results:
            generator_results = pd.DataFrame([component_results["generator"]])
            generator_results.columns = pd.MultiIndex.from_tuples([("GEN", col) for col in generator_results.columns])
        else:
            generator_results = pd.DataFrame()

        if "retriever" in component_results:
            retriever_results = pd.DataFrame([component_results["retriever"]])
            retriever_results.columns = pd.MultiIndex.from_tuples([("RET", col) for col in retriever_results.columns])
        else:
            retriever_results = pd.DataFrame()

        end_to_end_results = pd.DataFrame([end_to_end_results])
        end_to_end_results.columns = pd.MultiIndex.from_tuples([("E2E", col) for col in end_to_end_results.columns])

        results_df = pd.concat([
            end_to_end_results,
            generator_results,
            retriever_results
        ], axis=1)

        results_df.loc[:, "run_id"] = run_id
        results_df.set_index("run_id", inplace=True)
        
        return results_df
        

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
        
        # Get component name to type mapping
        component_types = {}
        pipeline_components = self.pipeline.to_dict()["components"]
        for comp_name, comp_details in pipeline_components.items():
            comp_type = comp_details.get("type", "")
            if ".generator." in comp_type:
                component_types[comp_name] = "generator"
            elif ".retriever." in comp_type:
                component_types[comp_name] = "retriever"
            else:
                # Fallback to component name-based detection for compatibility
                if comp_name.startswith("retriever"):
                    component_types[comp_name] = "retriever"
                elif comp_name.startswith("generator") or comp_name.startswith("llm"):
                    component_types[comp_name] = "generator"
        
        # Group test cases by component
        component_data = defaultdict(lambda: defaultdict(list))
        
        # Gather all data needed for each component
        for test_case in test_cases:
            component_outputs = test_case["component_outputs"]
            
            for component_name in component_outputs:
                # Map component name to its standardized type (retriever, generator)
                if component_name in component_types:
                    component_type = component_types[component_name]
                    # Store data under the standardized component type
                    component_data[component_type]["component_outputs"].append(component_outputs[component_name])
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


def evaluate(data: Dict[str, Any], pipeline: Pipeline, run_name: str,
           positive_label: str = DEFAULT_POSITIVE_LABEL, 
           negative_label: str = DEFAULT_NEGATIVE_LABEL,
           track_resources: bool = False,
           resource_output_prefix: Optional[str] = None) -> pd.DataFrame:
    """
    Evaluates the pipeline on the test cases.
    
    Args:
        data: Evaluation data
        pipeline: Haystack pipeline to evaluate
        run_name: Identifier for this evaluation run
        positive_label: The label to consider as positive class (default: "valid")
        negative_label: The label to consider as negative class (default: "invalid")
        track_resources: Whether to track system resource usage during evaluation
        resource_output_prefix: Prefix for resource metric files (if track_resources is True)
        
    Returns:
        pd.DataFrame: Results for the metrics and additional metadata
    """
    # Initialize resource tracking if requested
    resource_tracker = None
    if track_resources:
        try:
            from ragnroll.metrics.system import SystemResourceTracker
            resource_tracker = SystemResourceTracker()
            resource_tracker.start_tracking()
            logger.info(f"System resource tracking enabled for evaluation run: {run_name}")
        except ImportError:
            logger.warning("Could not import SystemResourceTracker. Resource tracking disabled.")
            track_resources = False
    
    # Store full metric results for Langfuse reporting
    end_to_end_metrics_details = {}
    component_metrics_details = {}
    trace_ids = []
    
    try:
        # Create EvaluationDataset to process the data
        dataset = EvaluationDataset(data)
        
        # Generate predictions
        dataset.generate_predictions(pipeline)
        processed_data = dataset.get_processed_data()
        
        # Extract trace IDs from component outputs if available
        for test_case in processed_data["test_cases"]:
            component_outputs = test_case.get("component_outputs", {})
            # Check for tracer component with trace URL
            if "tracer" in component_outputs and "trace_url" in component_outputs["tracer"]:
                # Extract trace ID from URL
                trace_url = component_outputs["tracer"]["trace_url"]
                # The URL format is typically: https://langfuse.com/[org]/traces/[trace_id]
                # Extract trace ID from the last part of the URL
                trace_id = trace_url.split("/")[-1] if trace_url else None
                if trace_id:
                    trace_ids.append(trace_id)
        
        # Run the evaluation with the original evaluator
        evaluator = Evaluator(pipeline, positive_label=positive_label, negative_label=negative_label)
        
        # Run end-to-end evaluations and capture detailed results
        test_cases = processed_data["test_cases"]
        
        # Extract expected and actual outputs
        expected_outputs = [tc["expected_output"] for tc in test_cases]
        actual_outputs = [tc["actual_output"] for tc in test_cases]
        
        # Run each end-to-end metric
        end_to_end_results = {}
        for metric_name, metric in evaluator.end_to_end_metrics.items():
            try:
                metric_result = metric.run(
                    expected_outputs=expected_outputs,
                    actual_outputs=actual_outputs
                )
                end_to_end_results[metric_name] = metric_result["score"]
                # Store full metric details for Langfuse
                end_to_end_metrics_details[metric_name] = metric_result
            except Exception as e:
                logger.error(f"Error evaluating with {metric_name}: {e}")
                end_to_end_results[metric_name] = 0.0
                end_to_end_metrics_details[metric_name] = {
                    "score": 0.0,
                    "success": False,
                    "details": {"error": str(e)}
                }
        
        # Run component-wise evaluations
        component_results = evaluator._evaluate_components(test_cases)
        
        # Capture detailed component metric results for Langfuse
        for component_type, metrics in evaluator.component_metrics.items():
            component_metrics_details[component_type] = {}
            
            if component_type not in component_results:
                continue
                
            for metric_name, metric in metrics.items():
                if metric_name in component_results[component_type]:
                    # Get the metric result - we need to re-run to get the details
                    try:
                        # Get component-specific test data
                        if component_type == "retriever":
                            # For retrievers, we need queries and component outputs
                            queries = [tc["input"] for tc in test_cases]
                            component_outputs = [
                                tc["component_outputs"].get("retriever", {}) 
                                for tc in test_cases
                            ]
                            metric_result = metric.run(component_outputs=component_outputs, queries=queries)
                        elif component_type == "generator":
                            # For generators, we need expected outputs and component outputs
                            component_outputs = [
                                tc["component_outputs"].get("llm", tc["component_outputs"].get("generator", {})) 
                                for tc in test_cases
                            ]
                            metric_result = metric.run(
                                component_outputs=component_outputs,
                                expected_outputs=expected_outputs
                            )
                        else:
                            # Default case - just use the score from the results
                            metric_result = {
                                "score": component_results[component_type][metric_name],
                                "success": component_results[component_type][metric_name] >= metric.threshold
                            }
                            
                        component_metrics_details[component_type][metric_name] = metric_result
                    except Exception as e:
                        logger.error(f"Error capturing details for {component_type}.{metric_name}: {e}")
                        component_metrics_details[component_type][metric_name] = {
                            "score": component_results[component_type][metric_name],
                            "success": component_results[component_type][metric_name] >= metric.threshold,
                            "details": {"error": str(e)}
                        }
        
        # Convert results to DataFrame
        scores = evaluator._results_to_df(end_to_end_results, component_results, run_name)
        
        # Print scores to console
        print_scores({
            "end-to-end": end_to_end_results,
            "component-wise": component_results
        })
        
        # Add resource metrics if tracking was enabled
        if track_resources and resource_tracker:
            # Get metrics summary
            metrics_summary = resource_tracker.get_metrics_summary()
            
            # Flatten the nested dictionaries
            flat_metrics = {}
            
            # Duration and samples
            flat_metrics["duration_seconds"] = float(metrics_summary.get("duration_seconds", 0))
            flat_metrics["samples_count"] = float(metrics_summary.get("samples_count", 0))
            
            # CPU metrics
            if "cpu" in metrics_summary and isinstance(metrics_summary["cpu"], dict):
                cpu = metrics_summary["cpu"]
                # System CPU
                if "system" in cpu and isinstance(cpu["system"], dict):
                    system = cpu["system"]
                    flat_metrics["cpu_system_mean"] = float(system.get("mean", 0))
                    flat_metrics["cpu_system_max"] = float(system.get("max", 0))
                    flat_metrics["cpu_system_min"] = float(system.get("min", 0))
                # Process CPU
                if "process" in cpu and isinstance(cpu["process"], dict):
                    process = cpu["process"]
                    flat_metrics["cpu_process_mean"] = float(process.get("mean", 0))
                    flat_metrics["cpu_process_max"] = float(process.get("max", 0))
                    flat_metrics["cpu_process_min"] = float(process.get("min", 0))
            
            # Memory metrics
            if "memory" in metrics_summary and isinstance(metrics_summary["memory"], dict):
                memory = metrics_summary["memory"]
                if "process" in memory and isinstance(memory["process"], dict):
                    process = memory["process"]
                    flat_metrics["memory_process_mean_mb"] = float(process.get("mean", 0))
                    flat_metrics["memory_process_max_mb"] = float(process.get("max", 0))
                    flat_metrics["memory_process_min_mb"] = float(process.get("min", 0))
                
                if "system" in memory and isinstance(memory["system"], dict):
                    system = memory["system"]
                    if "used" in system and isinstance(system["used"], dict):
                        used = system["used"]
                        flat_metrics["memory_system_used_mean_mb"] = float(used.get("mean", 0))
                        flat_metrics["memory_system_used_max_mb"] = float(used.get("max", 0))
                        flat_metrics["memory_system_used_min_mb"] = float(used.get("min", 0))
            
            # Add system metrics to the scores DataFrame
            for metric_name, value in flat_metrics.items():
                scores.loc[run_name, ("SYS", metric_name)] = value
            
            # Save raw metrics to file if output prefix is provided
            if resource_output_prefix:
                try:
                    import json
                    output_file = f"{resource_output_prefix}_{run_name}.json"
                    with open(output_file, 'w') as f:
                        json.dump(metrics_summary, f, indent=2)
                    logger.info(f"Raw system metrics saved to {output_file}")
                except Exception as e:
                    logger.error(f"Error saving raw metrics to file: {e}")
    
    finally:
        # Stop resource tracking if it was started
        if track_resources and resource_tracker:
            resource_tracker.stop_tracking()
    
    # Add trace IDs and metrics details to the returned result
    # This information will be used by the CLI to report to Langfuse
    result = {
        "dataframe": scores,
        "trace_ids": trace_ids,
        "metrics": end_to_end_metrics_details,
        "component_metrics": component_metrics_details
    }
    
    # For backward compatibility, ensure the DataFrame is directly accessible
    for key, value in result.items():
        scores.attrs[key] = value
    
    return result


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
