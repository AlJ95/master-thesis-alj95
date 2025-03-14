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
            elif ".reranker." in comp_type:
                component_types[comp_name] = "reranker"
            else:
                # Fallback to component name-based detection for compatibility
                if comp_name.startswith("retriever"):
                    component_types[comp_name] = "retriever"
                elif comp_name.startswith("reranker"):
                    component_types[comp_name] = "reranker"
                elif comp_name.startswith("generator") or comp_name.startswith("llm"):
                    component_types[comp_name] = "generator"
        
        for component_type, metric_classes in MetricRegistry.get_component_metrics().items():
            # Only include metrics for components that exist in the pipeline
            if component_type in component_types.values():
                metrics[component_type] = {
                    name: metric_cls() for name, metric_cls in metric_classes.items()
                }
                
        # Special case: If there's a reranker, use retriever metrics for it too
        if "reranker" in component_types.values():
            retriever_metrics = MetricRegistry.get_component_metrics().get("retriever", {})
            if retriever_metrics:
                metrics["reranker"] = {
                    name: metric_cls() for name, metric_cls in retriever_metrics.items()
                }
                logger.info("Reranker-Komponente erkannt. Verwende Retriever-Metriken für die Evaluierung.")
        
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
            elif ".reranker." in comp_type:
                component_types[comp_name] = "reranker"
            else:
                # Fallback to component name-based detection for compatibility
                if comp_name.startswith("retriever"):
                    component_types[comp_name] = "retriever"
                elif comp_name.startswith("reranker"):
                    component_types[comp_name] = "reranker"
                elif comp_name.startswith("generator") or comp_name.startswith("llm"):
                    component_types[comp_name] = "generator"
        
        # Group test cases by component
        component_data = defaultdict(lambda: defaultdict(list))
        
        # Gather all data needed for each component
        for test_case in test_cases:
            component_outputs = test_case["component_outputs"]
            
            for component_name in component_outputs:
                # Map component name to its standardized type (retriever, reranker, generator)
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
        
        # Special case: If there's a reranker, we also need to extract its data
        # The reranker receives documents from the retriever, so we need to use the retriever output data
        if "reranker" in self.component_metrics and "retriever" in component_data:
            # Copy the relevant data from the retriever to the reranker if the reranker itself has no outputs
            if "reranker" not in component_data:
                logger.info("Extracting reranker data from retriever outputs.")
                component_data["reranker"] = defaultdict(list)
                component_data["reranker"]["expected_outputs"] = component_data["retriever"]["expected_outputs"].copy()
                component_data["reranker"]["input_texts"] = component_data["retriever"]["input_texts"].copy()
                component_data["reranker"]["queries"] = component_data["retriever"]["queries"].copy()
                if "expected_retrievals" in component_data["retriever"]:
                    component_data["reranker"]["expected_retrievals"] = component_data["retriever"]["expected_retrievals"].copy()
                
                # Extract reranker outputs from pipeline components if available
                reranker_outputs = []
                for tc in test_cases:
                    # Look for reranker component in the outputs
                    reranker_output = None
                    for comp_name, comp_output in tc["component_outputs"].items():
                        if comp_name in component_types and component_types[comp_name] == "reranker":
                            reranker_output = comp_output
                            break
                    
                    # If no explicit reranker output was found, take the last set of documents before the LLM
                    if reranker_output is None:
                        # Determine the order of components in the pipeline
                        component_order = list(self.pipeline.to_dict()["components"].keys())
                        # Find LLM/generator component index
                        llm_idx = next((i for i, comp in enumerate(component_order) 
                                      if comp in component_types and component_types[comp] == "generator"), 
                                     len(component_order))
                        
                        # Look backwards from the LLM to find the last retriever/reranker
                        for i in range(llm_idx - 1, -1, -1):
                            comp_name = component_order[i]
                            if comp_name in tc["component_outputs"]:
                                if comp_name in component_types and component_types[comp_name] != "generator":
                                    reranker_output = tc["component_outputs"][comp_name]
                                    break
                    
                    reranker_outputs.append(reranker_output if reranker_output is not None else [])
                
                component_data["reranker"]["component_outputs"] = reranker_outputs
        
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
        pd.DataFrame: Results for the metrics
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
    
    try:
        # Run the evaluation
        evaluator = Evaluator(pipeline, positive_label=positive_label, negative_label=negative_label)
        scores = evaluator.evaluate(data, run_name)
        
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
                # System percentage
                if "system_percent" in memory and isinstance(memory["system_percent"], dict):
                    sys_percent = memory["system_percent"]
                    flat_metrics["memory_system_percent_mean"] = float(sys_percent.get("mean", 0))
                    flat_metrics["memory_system_percent_max"] = float(sys_percent.get("max", 0))
                    flat_metrics["memory_system_percent_min"] = float(sys_percent.get("min", 0))
                # System used GB
                if "system_used_gb" in memory and isinstance(memory["system_used_gb"], dict):
                    sys_used = memory["system_used_gb"]
                    flat_metrics["memory_system_used_gb_mean"] = float(sys_used.get("mean", 0))
                    flat_metrics["memory_system_used_gb_max"] = float(sys_used.get("max", 0))
                    flat_metrics["memory_system_used_gb_min"] = float(sys_used.get("min", 0))
                # Process MB
                if "process_mb" in memory and isinstance(memory["process_mb"], dict):
                    proc_mb = memory["process_mb"]
                    flat_metrics["memory_process_mb_mean"] = float(proc_mb.get("mean", 0))
                    flat_metrics["memory_process_mb_max"] = float(proc_mb.get("max", 0))
                    flat_metrics["memory_process_mb_min"] = float(proc_mb.get("min", 0))
            
            # Create flat resource metrics DataFrame
            resource_df = pd.DataFrame([flat_metrics])
            resource_df.loc[:, "run_id"] = run_name
            resource_df.set_index("run_id", inplace=True)
            
            # Prefix resource metrics columns to avoid conflicts
            resource_df.columns = pd.MultiIndex.from_tuples([("SYS", col) for col in resource_df.columns])
            
            # Add to results
            scores = pd.concat([scores, resource_df], axis=1)
            
            # Save detailed metrics if output prefix was provided
            if resource_output_prefix:
                try:
                    # Save CSV
                    csv_path = f"{resource_output_prefix}_metrics.csv"
                    resource_tracker.to_csv(csv_path)
                    logger.info(f"Resource metrics saved to {csv_path}")
                except Exception as e:
                    logger.error(f"Error saving resource metrics: {e}")
        
        return scores
    
    finally:
        # Stop resource tracking if it was started
        if track_resources and resource_tracker:
            resource_tracker.stop_tracking()
            logger.info("Resource tracking stopped")
            
            # Print summary
            try:
                resource_tracker.print_summary()
            except:
                pass


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
