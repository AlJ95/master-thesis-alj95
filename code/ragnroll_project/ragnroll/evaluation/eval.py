import sys, os
try:
    from ragnroll.metrics import MetricRegistry, BaseMetric
    from ragnroll.metrics.system import SystemResourceTracker
    from ragnroll.metrics.end2end import ClassificationBaseMetric
    from ragnroll.utils.config import get_components_from_config_by_classes
    from ragnroll.utils.pipeline import get_last_component_with_documents
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from ragnroll.metrics import MetricRegistry, BaseMetric
    from ragnroll.metrics.end2end import ClassificationBaseMetric
    from ragnroll.metrics.system import SystemResourceTracker
    from ragnroll.utils.config import get_components_from_config_by_classes
    from ragnroll.utils.pipeline import get_last_component_with_documents
from haystack import Pipeline
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
import os

DEFAULT_POSITIVE_LABEL = "valid"
DEFAULT_NEGATIVE_LABEL = "invalid"

logger = logging.getLogger(__name__)

class EvaluationDataset:
    """Class to handle evaluation datasets."""
    
    def __init__(self, evaluation_data: Dict[str, Any]):
        """Initialize with raw evaluation data."""
        self.evaluation_data = evaluation_data
        self.processed_data = []
    
    def generate_predictions(self, pipeline: Pipeline) -> None:
        """Generate predictions for all test cases."""
        for test_case in self.evaluation_data["test_cases"]:
            try:
                input_text = test_case["input"]
                expected_output = test_case["expected_output"]
                # Generate the answer using the pipeline
                response = self._generate_answer(pipeline, input_text)
                actual_output = self._extract_answer_from_pipeline(response)
                
                # Add to dataset
                self.processed_data.append({
                    "input": input_text,
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                    "component_outputs": response,
                })
            except Exception as e:
                logger.error(f"Error generating answer for test case: {e}")
    
    def _generate_answer(self, pipeline: Pipeline, input_text: str) -> Dict[str, Any]:
        """Generate an answer using the pipeline."""
        components = list(pipeline.to_dict()["components"].keys())
        data = dict(query=input_text)

        # # Add text to data if embedding retriever is present
        # if get_components_from_config_by_classes(pipeline.to_dict(), ".embedding_retriever."):
        #     data["text"] = input_text

        return pipeline.run(data=data, include_outputs_from=components)
    
    def _extract_answer_from_pipeline(self, response: Dict[str, Any]) -> str:
        """Extract the answer from the pipeline response."""
        if "answer_builder" in response:
            return response["answer_builder"]["answers"][0].data
        else:
            raise ValueError(f"Could not extract answer from pipeline response: {response}. Make sure the pipeline has an answer_builder component.")
    
    def get_processed_data(self) -> Dict[str, Any]:
        """Get the processed data with predictions."""
        return self.processed_data

    def get_trace_ids(self) -> List[str]:
        """
        Get the trace IDs from the processed data.
        
        Trace URLs are stored in the tracer component.
        The URL format is typically: https://langfuse.com/[org]/traces/[trace_id]
        We extract the trace_id from the URL and return a list of trace IDs.
        """
        try:
            trace_ids = []
            for test_case in self.processed_data:
                if "tracer" in test_case["component_outputs"] and "trace_url" in test_case["component_outputs"]["tracer"]:
                    trace_ids.append(test_case["component_outputs"]["tracer"]["trace_url"].split("/")[-1])
            return trace_ids
        except Exception as e:
            logger.error(f"Error getting trace IDs: {e}")
            return []

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
        self.individual_scores = {}

        logger.info(f"Evaluator initialisiert mit Labels: positiv='{self.positive_label}', negativ='{self.negative_label}'")
    
    def _instantiate_end_to_end_metrics(self) -> Dict[str, BaseMetric]:
        """Create instances of all registered end-to-end metrics."""
        metrics = {}
        
        for name, metric_cls in MetricRegistry.get_end_to_end_metrics().items():
            # Überprüfen, ob es sich um eine Klassifikationsmetrik handelt
            metrics[name] = metric_cls(
                positive_label=self.positive_label,
                negative_label=self.negative_label
            )
                
        return metrics
    
    def _instantiate_component_metrics(self) -> Dict[str, Dict[str, BaseMetric]]:
        """Create instances of component metrics relevant to this pipeline."""
        pipeline_components = self.pipeline.to_dict()["components"]
        metrics = {}

        # Map component names to their types from the pipeline
        component_types = {}
        for component_name, component_dict in pipeline_components.items():
            expected_component_type = component_dict["type"]
            
            if ".generators." in expected_component_type:
                component_types[component_name] = "generator"
            elif ".retrievers." in expected_component_type:
                component_types[component_name] = "retriever"
        
        for expected_component_type, metric_classes in MetricRegistry.get_component_metrics().items():
            # Only include metrics for components that exist in the pipeline
            if expected_component_type in component_types.values():
                metrics[expected_component_type] = {
                    name: metric_cls() for name, metric_cls in metric_classes.items()
                }
                
        return metrics
    
    def _track_resources(self, track_resources: bool, run_name: str):
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

        return resource_tracker

    def gather_resource_metrics(self, resource_tracker: SystemResourceTracker) -> Dict[str, Any]:
        """
        Gather resource metrics from the resource tracker.
        """
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
        system_metrics = pd.DataFrame(flat_metrics, index=[0])
        return system_metrics

    def _get_component_types_mapping(self) -> Dict[str, str]:
        """
        Get a mapping of component names to their types.
        {'component_name': 'component_type'}

        Example:
        {'llm': 'generator', 'retriever': 'retriever'}
        """
        component_types = {}
        pipeline_components = self.pipeline.to_dict()["components"]
        for component_name, component_details in pipeline_components.items():
            component_type = component_details.get("type", "")
            if ".generators." in component_type:
                component_types[component_name] = "generator"
            elif ".retrievers." in component_type:
                component_types[component_name] = "retriever"
        return component_types
    
    def evaluate(self, evaluation_data: Dict[str, Any], run_name: str, track_resources: bool = False) -> pd.DataFrame:
        """
        Run the evaluation on the provided data.
        
        Args:
            evaluation_data: Test cases to evaluate
            
        Returns:
            Dict[str, Any]: Evaluation results
        """

        if track_resources:
            resource_tracker = self._track_resources(track_resources, run_name)

        try:

            # TODO Das sollte die FUnktion sein, die ausgeführt wird.
            # Generate dataset with predictions
            dataset = EvaluationDataset(evaluation_data)
            # Generiere Vorhersagen
            dataset.generate_predictions(self.pipeline)
            processed_data = dataset.get_processed_data()
            trace_ids = dataset.get_trace_ids()
            # Run end-to-end evaluations
            end_to_end_results = self._evaluate_end_to_end(processed_data, trace_ids)
            
            # Run component-wise evaluations
            component_results = self._evaluate_components(processed_data)
            
            # Combine results
            results = {
                "end-to-end": end_to_end_results,
                "component-wise": component_results
            }

            print_scores(results)

            # Convert results to pandas DataFrames
            results_df = self._results_to_df(end_to_end_results, component_results, run_name)
            

            # Add resource metrics if tracking was enabled
            if track_resources and resource_tracker:
                # Get metrics summary
                resource_metrics = self.gather_resource_metrics(resource_tracker)
                resource_metrics.loc[:, "run_name"] = run_name
                resource_metrics.set_index("run_name", inplace=True)
                
                # apply MultiIndex to resource_metrics
                resource_metrics.columns = pd.MultiIndex.from_tuples([("SYS","", col) for col in resource_metrics.columns])
                results_df = pd.concat([results_df, resource_metrics], axis=1)

            return results_df
        finally:
            if track_resources and resource_tracker:
                resource_tracker.stop_tracking()

    def _results_to_df(self, end_to_end_results: Dict[str, float], component_results: Dict[str, Dict[str, float]], run_name: str) -> pd.DataFrame:
        """
        Convert results to pandas DataFrames
        """
        gathered_retriever_results = pd.DataFrame()
        gathered_generator_results = pd.DataFrame()

        if "generator" in component_results:
            for generator_name, generator_results in component_results["generator"].items():
                generator_results = pd.DataFrame([generator_results])
                generator_results.columns = pd.MultiIndex.from_tuples([("GEN", generator_name, col) for col in generator_results.columns])
                gathered_generator_results = pd.concat([gathered_generator_results, generator_results], axis=1)
        else:
            gathered_generator_results = pd.DataFrame()

        if "retriever" in component_results:
            for retriever_name, retriever_results in component_results["retriever"].items():
                retriever_results = pd.DataFrame([retriever_results])
                retriever_results.columns = pd.MultiIndex.from_tuples([("RET", retriever_name, col) for col in retriever_results.columns])
                gathered_retriever_results = pd.concat([gathered_retriever_results, retriever_results], axis=1)
        else:
            gathered_retriever_results = pd.DataFrame()

        end_to_end_results = pd.DataFrame([end_to_end_results])
        end_to_end_results.columns = pd.MultiIndex.from_tuples([("E2E", "", col) for col in end_to_end_results.columns])

        results_df = pd.concat([
            end_to_end_results,
            gathered_generator_results,
            gathered_retriever_results
        ], axis=1)

        results_df.loc[:, "run_name"] = run_name
        results_df.set_index("run_name", inplace=True)

        print(results_df.T)
        
        return results_df
        
    def _evaluate_end_to_end(self, test_cases: List[Dict[str, Any]], trace_ids: List[str]) -> Dict[str, float]:
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
        
        # Get exact match scores for each trace ID and post them to Langfuse
        ClassificationBaseMetric.return_exact_match_scores(
            expected_outputs=expected_outputs,
            actual_outputs=actual_outputs,
            trace_ids=trace_ids,
            callback=self._individual_score_callback,
            positive_label=self.positive_label,
            negative_label=self.negative_label
        )
        self._post_langfuse_scores()

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
    
    def _evaluate_retriever(self, retriever_name: List[str], test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate all retriever metrics on the complete set of test cases.
        """
        results = {}
        
        # Extract queries and retriever outputs from test cases
        queries = [tc["input"] for tc in test_cases]
        retriever_outputs = [
            tc["component_outputs"].get(retriever_name, {})
            for tc in test_cases
        ]
        
        # Apply each retriever metric
        for metric_name, metric in self.component_metrics.get("retriever", {}).items():
            try:
                metric_result = metric.run(
                    component_outputs=retriever_outputs,
                    queries=queries
                )
                results[metric_name] = metric_result["score"]
            except Exception as e:
                logger.error(f"Error evaluating retriever {retriever_name} with {metric_name}: {e}")
                results[metric_name] = 0.0
                
        return results
        
    def _evaluate_generator(self, generator_name: str, test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate all generator metrics on the complete set of test cases.
        """
        results = {}
        
        # Extract queries and generator outputs from test cases
        queries = [tc["input"] for tc in test_cases]
        generator_outputs = [
            tc["component_outputs"].get(generator_name, {})
            for tc in test_cases
        ]
        component_with_documents = get_last_component_with_documents(self.pipeline, generator_name)
        if component_with_documents is None:
            logger.warning(f"No component with documents found for {generator_name}")
            contexts = None
        else:
            contexts = [
                tc["component_outputs"][component_with_documents]["documents"]
                for tc in test_cases
            ]
        
        # Apply each generator metric
        for metric_name, metric in self.component_metrics.get("generator", {}).items():
            try:
                metric_result = metric.run(
                    component_outputs=generator_outputs,
                    queries=queries,
                    contexts=contexts
                )
                results[metric_name] = metric_result["score"]
            except Exception as e:
                logger.error(f"Error evaluating generator {generator_name} with {metric_name}: {e}")
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
        component_types = self._get_component_types_mapping()
        retriever_names = [name for name, t in component_types.items() if t == "retriever"]
        generator_names = [name for name, t in component_types.items() if t == "generator"]

        retriever_results = {}
        # Evaluate retrievers
        for retriever_name in retriever_names:
            retriever_results[retriever_name] = self._evaluate_retriever(retriever_name, test_cases)

        # Evaluate generators
        generator_results = {}  
        for generator_name in generator_names:
            generator_results[generator_name] = self._evaluate_generator(generator_name, test_cases)

        # Combine results
        results = {
            "retriever": retriever_results,
            "generator": generator_results
        }

        return results
    
    def _individual_score_callback(self, trace_id: str, metric_name: str, score: float) -> None:
        """
        Callback function to store individual scores for each trace URL.
        """
        self.individual_scores[trace_id] = {metric_name: score}

    def _post_langfuse_scores(self) -> None:
        """
        Post the individual scores to Langfuse.
        """
        for i, (trace_id, scores) in enumerate(self.individual_scores.items()):
            self.pipeline.get_component("tracer").tracer._tracer.score(
                id=f"{trace_id}-{i}",
                trace_id=trace_id,
                name=list(scores.keys())[0],
                value=list(scores.values())[0],
                data_type="BOOLEAN"
            )


def print_scores(scores: Dict[str, Any]) -> None:
    """
    Prints the scores for the metrics
    """
    print("\n===== Evaluation Results =====")
    print("\n=== End-to-End Metrics ===")
    for metric, score in scores["end-to-end"].items():
        print(f"{metric}: {score:.4f}")
    print("\n=== Component-Wise Metrics ===")
    for component_type, component_type_metrics in scores["component-wise"].items():
        print(f"Component Type: {component_type}")
        for component, metrics in component_type_metrics.items():
            print(f"  Component: {component}")
            for metric, score in metrics.items():
                print(f"    {metric}: {score:.4f}")


if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    import sys

    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logger.warning(f"No .env file found at {env_path}")
        exit(1)

    config_path = Path(__file__).parent.parent.parent / "configs" / "predefined_4r.yaml"
    assert config_path.exists(), f"Config file {config_path} does not exist"

    # Load the test cases
    from ragnroll.utils.pipeline import config_to_pipeline
    run_name = "test_run"
    pipeline = config_to_pipeline(config_path)

    from ragnroll.utils.ingestion import index_documents
    corpus_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "dev_data" / "corpus"
    pipeline = index_documents(corpus_dir, pipeline)

    from ragnroll.evaluation.data import load_evaluation_data
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "dev_data" / "val" / "synthetic_rag_evaluation.json"
    assert data_path.exists(), f"Data file {data_path} does not exist"
    data = load_evaluation_data(data_path)

    evaluator = Evaluator(pipeline)
    result = evaluator.evaluate(data, run_name, track_resources=True)
    print(result)
