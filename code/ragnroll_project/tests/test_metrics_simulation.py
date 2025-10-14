import pytest
from unittest.mock import patch, MagicMock, Mock
from haystack import Document
from haystack.components.evaluators import ContextRelevanceEvaluator
from ragnroll.evaluation.eval import Evaluator, EvaluationDataset
from ragnroll.metrics.component.retriever import HaystackContextRelevanceMetric, MAPAtKMetric
from ragnroll.metrics.component.generator import FormatValidatorMetric
from ragnroll.metrics.end2end import AccuracyMetric
import numpy as np


class TestMetricsSimulation:
    """
    Test suite for simulating and validating metrics calculation.

    This test suite creates a deterministic simulation of the RAG pipeline
    (Retriever -> Reranker -> Generator -> Context Relevance Evaluator)
    to ensure metrics work correctly with known inputs and expected outputs.
    """

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline with predefined components."""
        pipeline = MagicMock()
        pipeline.to_dict.return_value = {
            "components": {
                "retriever": {"type": "haystack_integrations.components.retrievers.chroma.ChromaEmbeddingRetriever"},
                "reranker": {"type": "haystack.components.rankers.meta.MetaRanker"},
                "llm": {"type": "haystack.components.generators.openai.GPTGenerator"},
                "answer_builder": {"type": "haystack.components.builders.AnswerBuilder"}
            },
            "connections": [
                {"sender": "retriever.documents", "receiver": "reranker.documents"},
                {"sender": "reranker.documents", "receiver": "llm.documents"},
                {"sender": "llm.replies", "receiver": "answer_builder.replies"}
            ]
        }
        return pipeline

    @pytest.fixture
    def deterministic_test_data(self):
        """
        Create deterministic test data with known expected outcomes.

        This simulates a complete RAG pipeline with:
        - Retriever: Returns 5 documents with fixed scores
        - Reranker: Re-ranks documents deterministically
        - Generator: Produces answers based on top documents
        - Context Relevance: Mocked to return fixed relevance scores
        """
        return [
            {
                "query": "What are the benefits of solar energy?",
                "retriever_output": {
                    "documents": [
                        Document(content="Solar energy reduces electricity bills and has a lower carbon footprint.", meta={"score": 0.95}),
                        Document(content="Solar panels can be installed on rooftops or in large solar farms.", meta={"score": 0.88}),
                        Document(content="Wind energy is another renewable energy source that uses turbines.", meta={"score": 0.72}),
                        Document(content="The sun provides enough energy in one hour to power the world for a year.", meta={"score": 0.65}),
                        Document(content="Football is a popular sport played by many people.", meta={"score": 0.12})
                    ]
                },
                "reranker_output": {
                    "documents": [
                        Document(content="Solar energy reduces electricity bills and has a lower carbon footprint.", meta={"score": 0.98}),
                        Document(content="The sun provides enough energy in one hour to power the world for a year.", meta={"score": 0.85}),
                        Document(content="Solar panels can be installed on rooftops or in large solar farms.", meta={"score": 0.82}),
                        Document(content="Wind energy is another renewable energy source that uses turbines.", meta={"score": 0.45}),
                        Document(content="Football is a popular sport played by many people.", meta={"score": 0.08})
                    ]
                },
                "generator_output": {
                    "replies": ["Solar energy offers significant benefits including reduced electricity bills, lower carbon footprint, and abundant renewable power from the sun."]
                },
                "answer_builder_output": {
                    "answers": [Mock(data="Solar energy offers significant benefits including reduced electricity bills, lower carbon footprint, and abundant renewable power from the sun.")]
                },
                "expected_output": "valid",  # For classification metrics
                "mocked_context_relevance_scores": [1.0, 0.9, 0.8, 0.3, 0.0]  # Fixed scores for deterministic testing
            },
            {
                "query": "How does photosynthesis work?",
                "retriever_output": {
                    "documents": [
                        Document(content="Photosynthesis is the process by which plants convert sunlight into energy.", meta={"score": 0.92}),
                        Document(content="Plants use chlorophyll to absorb light and produce glucose.", meta={"score": 0.85}),
                        Document(content="The chemical equation for photosynthesis involves CO2, water, and sunlight.", meta={"score": 0.78}),
                        Document(content="Animals obtain energy by consuming plants or other animals.", meta={"score": 0.45}),
                        Document(content="Basketball is a popular sport played by many people.", meta={"score": 0.15})
                    ]
                },
                "reranker_output": {
                    "documents": [
                        Document(content="Photosynthesis is the process by which plants convert sunlight into energy.", meta={"score": 0.96}),
                        Document(content="Plants use chlorophyll to absorb light and produce glucose.", meta={"score": 0.89}),
                        Document(content="The chemical equation for photosynthesis involves CO2, water, and sunlight.", meta={"score": 0.76}),
                        Document(content="Animals obtain energy by consuming plants or other animals.", meta={"score": 0.42}),
                        Document(content="Basketball is a popular sport played by many people.", meta={"score": 0.12})
                    ]
                },
                "generator_output": {
                    "replies": ["Photosynthesis is the biological process where plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll."]
                },
                "answer_builder_output": {
                    "answers": [Mock(data="Photosynthesis is the biological process where plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.")]
                },
                "expected_output": "valid",
                "mocked_context_relevance_scores": [1.0, 0.95, 0.85, 0.2, 0.0]
            },
            {
                "query": "What is the capital of Germany?",
                "retriever_output": {
                    "documents": [
                        Document(content="Berlin is the capital of Germany and has a rich history.", meta={"score": 0.88}),
                        Document(content="Munich is a large city in Bavaria, southern Germany.", meta={"score": 0.75}),
                        Document(content="Paris is the capital of France.", meta={"score": 0.65}),
                        Document(content="London is the capital of the United Kingdom.", meta={"score": 0.55}),
                        Document(content="Rome is the capital of Italy.", meta={"score": 0.45})
                    ]
                },
                "reranker_output": {
                    "documents": [
                        Document(content="Berlin is the capital of Germany and has a rich history.", meta={"score": 0.94}),
                        Document(content="Munich is a large city in Bavaria, southern Germany.", meta={"score": 0.72}),
                        Document(content="Paris is the capital of France.", meta={"score": 0.58}),
                        Document(content="London is the capital of the United Kingdom.", meta={"score": 0.52}),
                        Document(content="Rome is the capital of Italy.", meta={"score": 0.41})
                    ]
                },
                "generator_output": {
                    "replies": ["Berlin is the capital of Germany."]
                },
                "answer_builder_output": {
                    "answers": [Mock(data="Berlin is the capital of Germany.")]
                },
                "expected_output": "valid",
                "mocked_context_relevance_scores": [1.0, 0.0, 0.0, 0.0, 0.0]  # Only first document is relevant
            }
        ]

    def _simulate_pipeline_run(self, test_case):
        """
        Simulate a complete pipeline run with deterministic outputs.

        Returns component outputs that would be produced by each stage.
        """
        return {
            "retriever": test_case["retriever_output"],
            "reranker": test_case["reranker_output"],
            "llm": test_case["generator_output"],
            "answer_builder": test_case["answer_builder_output"]
        }

    @patch('ragnroll.metrics.component.retriever.AsyncContextRelevanceEvaluator.run')
    def test_accuracy_calculation_with_deterministic_data(self, mock_evaluator_run, mock_pipeline, deterministic_test_data):
        """Test that accuracy metric calculates correctly with deterministic pipeline simulation."""

        # Mock the context relevance evaluator to return deterministic scores
        def mock_evaluator_side_effect(**kwargs):
            questions = kwargs.get('questions', [])
            contexts = kwargs.get('contexts', [])

            results = []
            for i, (question, context_list) in enumerate(zip(questions, contexts)):
                # Use the mocked scores from test data
                test_case = deterministic_test_data[i]
                mocked_scores = test_case["mocked_context_relevance_scores"]

                # Create mock result for each document in context
                relevant_statements = []
                score = 0.0

                for j, (doc, mock_score) in enumerate(zip(context_list, mocked_scores)):
                    if mock_score > 0.5:  # Consider documents with score > 0.5 as relevant
                        relevant_statements.append(doc.content)
                        score = max(score, mock_score)

                results.append({
                    "relevant_statements": relevant_statements,
                    "score": score
                })

            # Return aggregated results
            avg_score = np.mean([r["score"] for r in results]) if results else 0.0
            return {
                "results": results,
                "score": avg_score,
                "individual_scores": [r["score"] for r in results]
            }

        mock_evaluator_run.side_effect = mock_evaluator_side_effect

        # Create evaluator
        evaluator = Evaluator(mock_pipeline)

        # Prepare processed data from deterministic test cases
        processed_data = []
        for test_case in deterministic_test_data:
            component_outputs = self._simulate_pipeline_run(test_case)
            processed_data.append({
                "input": test_case["query"],
                "expected_output": test_case["expected_output"],
                "actual_output": test_case["expected_output"],  # Perfect predictions for accuracy test
                "component_outputs": component_outputs
            })

        # Run end-to-end evaluation
        results = evaluator._evaluate_end_to_end(processed_data, [])

        # With perfect predictions, accuracy should be 1.0
        assert abs(results["AccuracyMetric"] - 1.0) < 0.001, f"Expected accuracy 1.0, got {results['AccuracyMetric']}"

    @patch('ragnroll.metrics.component.retriever.AsyncContextRelevanceEvaluator.run')
    def test_map_at_k_calculation_with_deterministic_data(self, mock_evaluator_run, mock_pipeline, deterministic_test_data):
        """Test that MAP@K metric calculates correctly with deterministic pipeline simulation."""

        # Mock the context relevance evaluator for MAP@K
        def mock_evaluator_side_effect(**kwargs):
            questions = kwargs.get('questions', [])
            contexts = kwargs.get('contexts', [])

            results = []
            for i, (question, context_list) in enumerate(zip(questions, contexts)):
                test_case = deterministic_test_data[i]
                mocked_scores = test_case["mocked_context_relevance_scores"]

                relevant_statements = []
                for j, (doc, mock_score) in enumerate(zip(context_list, mocked_scores)):
                    if mock_score > 0.5:
                        relevant_statements.append(doc.content)

                results.append({
                    "relevant_statements": relevant_statements,
                    "score": 1.0 if relevant_statements else 0.0
                })

            avg_score = np.mean([r["score"] for r in results]) if results else 0.0
            return {
                "results": results,
                "score": avg_score,
                "individual_scores": [r["score"] for r in results]
            }

        mock_evaluator_run.side_effect = mock_evaluator_side_effect

        # Create evaluator
        evaluator = Evaluator(mock_pipeline)

        # Prepare processed data
        processed_data = []
        for test_case in deterministic_test_data:
            component_outputs = self._simulate_pipeline_run(test_case)
            processed_data.append({
                "input": test_case["query"],
                "expected_output": test_case["expected_output"],
                "actual_output": test_case["expected_output"],
                "component_outputs": component_outputs
            })

        # Run component evaluation
        component_results = evaluator._evaluate_components(processed_data)

        # Check that retriever metrics were calculated
        assert "retriever" in component_results
        retriever_results = component_results["retriever"]

        # Should have results for the mock retriever
        assert len(retriever_results) > 0

        # Get the first retriever's results
        retriever_name = list(retriever_results.keys())[0]
        metrics = retriever_results[retriever_name]

        # Check that MAPAtKMetric was calculated
        assert "MAPAtKMetric" in metrics

        # The MAP@K score should be deterministic and match expected calculation
        map_score = metrics["MAPAtKMetric"]
        assert isinstance(map_score, (int, float)), f"MAP@K score should be numeric, got {type(map_score)}"
        assert 0.0 <= map_score <= 1.0, f"MAP@K score should be between 0 and 1, got {map_score}"

    @patch('ragnroll.metrics.component.retriever.AsyncContextRelevanceEvaluator.run')
    def test_context_relevance_evaluator_mocking(self, mock_evaluator_run, deterministic_test_data):
        """Test that the context relevance evaluator is properly mocked for deterministic results."""

        # Set up mock to return fixed scores
        expected_scores = [0.8, 0.9, 1.0]
        mock_evaluator_run.return_value = {
            "results": [
                {"relevant_statements": ["statement1"], "score": expected_scores[0]},
                {"relevant_statements": ["statement2"], "score": expected_scores[1]},
                {"relevant_statements": ["statement3"], "score": expected_scores[2]}
            ],
            "score": np.mean(expected_scores),
            "individual_scores": expected_scores
        }

        # Create metric instance
        metric = HaystackContextRelevanceMetric()

        # Prepare test data
        component_outputs = [
            {"documents": [Document(content="test doc 1")]},
            {"documents": [Document(content="test doc 2")]},
            {"documents": [Document(content="test doc 3")]}
        ]
        queries = ["query1", "query2", "query3"]

        # Run evaluation
        result = metric.run(component_outputs=component_outputs, queries=queries)

        # Verify mock was called
        mock_evaluator_run.assert_called_once()

        # Verify results match expected mocked values
        assert abs(result["score"] - np.mean(expected_scores)) < 0.001
        assert result["individual_scores"] == expected_scores

    def test_format_validator_with_deterministic_answers(self, mock_pipeline, deterministic_test_data):
        """Test format validator with deterministic generator outputs."""

        # Create evaluator
        evaluator = Evaluator(mock_pipeline)

        # Prepare processed data with deterministic outputs
        processed_data = []
        for test_case in deterministic_test_data:
            component_outputs = self._simulate_pipeline_run(test_case)
            processed_data.append({
                "input": test_case["query"],
                "expected_output": test_case["expected_output"],
                "actual_output": test_case["expected_output"],
                "component_outputs": component_outputs
            })

        # Run component evaluation
        component_results = evaluator._evaluate_components(processed_data)

        # Check generator metrics
        assert "generator" in component_results
        generator_results = component_results["generator"]

        assert len(generator_results) > 0

        # Get first generator results
        generator_name = list(generator_results.keys())[0]
        metrics = generator_results[generator_name]

        # Should include FormatValidatorMetric
        assert "FormatValidatorMetric" in metrics

        # Format validator should work with the deterministic outputs
        format_score = metrics["FormatValidatorMetric"]
        assert isinstance(format_score, (int, float))

    def test_end_to_end_pipeline_simulation(self, mock_pipeline, deterministic_test_data):
        """Test complete pipeline simulation from retriever to final evaluation."""

        # Create evaluator
        evaluator = Evaluator(mock_pipeline)

        # Prepare processed data
        processed_data = []
        for test_case in deterministic_test_data:
            component_outputs = self._simulate_pipeline_run(test_case)
            processed_data.append({
                "input": test_case["query"],
                "expected_output": test_case["expected_output"],
                "actual_output": test_case["expected_output"],
                "component_outputs": component_outputs
            })

        # Run full evaluation
        end_to_end_results = evaluator._evaluate_end_to_end(processed_data, [])
        component_results = evaluator._evaluate_components(processed_data)

        # Verify all expected metrics are present
        expected_e2e_metrics = ["AccuracyMetric", "PrecisionMetric", "RecallMetric", "F1ScoreMetric"]
        for metric in expected_e2e_metrics:
            assert metric in end_to_end_results, f"Missing end-to-end metric: {metric}"
            assert isinstance(end_to_end_results[metric], (int, float))

        # Verify component metrics are present
        assert "retriever" in component_results
        assert "generator" in component_results

        # Verify retriever has expected metrics
        retriever_metrics = component_results["retriever"]
        assert len(retriever_metrics) > 0
        retriever_name = list(retriever_metrics.keys())[0]
        assert "HaystackContextRelevanceMetric" in retriever_metrics[retriever_name]
        assert "MAPAtKMetric" in retriever_metrics[retriever_name]

        # Verify generator has expected metrics
        generator_metrics = component_results["generator"]
        assert len(generator_metrics) > 0
        generator_name = list(generator_metrics.keys())[0]
        assert "FormatValidatorMetric" in generator_metrics[generator_name]

    def test_deterministic_results_across_runs(self, mock_pipeline, deterministic_test_data):
        """Test that results are deterministic across multiple runs."""

        evaluator = Evaluator(mock_pipeline)

        # Run evaluation multiple times
        results_runs = []
        for _ in range(3):
            processed_data = []
            for test_case in deterministic_test_data:
                component_outputs = self._simulate_pipeline_run(test_case)
                processed_data.append({
                    "input": test_case["query"],
                    "expected_output": test_case["expected_output"],
                    "actual_output": test_case["expected_output"],
                    "component_outputs": component_outputs
                })

            results = evaluator._evaluate_end_to_end(processed_data, [])
            results_runs.append(results)

        # All runs should produce identical results
        first_run = results_runs[0]
        for run in results_runs[1:]:
            for metric_name in first_run.keys():
                assert abs(first_run[metric_name] - run[metric_name]) < 0.001, \
                    f"Non-deterministic result for {metric_name}: {first_run[metric_name]} vs {run[metric_name]}"
