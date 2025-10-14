from typing import Dict, Any, List, Union, Callable, Optional
import logging
import numpy as np
import asyncio
from haystack import Document
from haystack.components.evaluators import ContextRelevanceEvaluator
from haystack.components.evaluators.llm_evaluator import LLMEvaluator
from haystack.components.generators.chat.types import ChatGenerator


try:
    from ragnroll.metrics.base import BaseMetric, MetricRegistry
except ImportError:
    import sys , os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base import BaseMetric, MetricRegistry

# Import RAGAS components
logger = logging.getLogger(__name__)


class AsyncContextRelevanceEvaluator(LLMEvaluator):
    """
    Asynchronous version of ContextRelevanceEvaluator that processes evaluations in parallel.

    This evaluator inherits from LLMEvaluator but overrides the run method to use
    asyncio.gather() for parallel processing of multiple evaluation requests.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        examples: Optional[list[dict[str, Any]]] = None,
        progress_bar: bool = True,
        raise_on_failure: bool = True,
        chat_generator: Optional[ChatGenerator] = None,
        num_processes: Optional[int] = None,
    ):
        """
        Creates an instance of AsyncContextRelevanceEvaluator.

        :param examples:
            Optional few-shot examples. Default examples will be used if none are provided.
        :param progress_bar:
            Whether to show a progress bar during the evaluation.
        :param raise_on_failure:
            Whether to raise an exception if the API call fails.
        :param chat_generator:
            a ChatGenerator instance which represents the LLM.
        :param num_processes:
            Number of parallel processes to use. If None, uses config value.
        """
        # Use default examples if none provided (same as ContextRelevanceEvaluator)
        _DEFAULT_EXAMPLES = [
            {
                "inputs": {
                    "questions": "What is the capital of Germany?",
                    "contexts": ["Berlin is the capital of Germany. Berlin and was founded in 1244."],
                },
                "outputs": {"relevant_statements": ["Berlin is the capital of Germany."]},
            },
            {
                "inputs": {
                    "questions": "What is the capital of France?",
                    "contexts": [
                        "Berlin is the capital of Germany and was founded in 1244.",
                        "Europe is a continent with 44 countries.",
                        "Madrid is the capital of Spain.",
                    ],
                },
                "outputs": {"relevant_statements": []},
            },
            {
                "inputs": {"questions": "What is the capital of Italy?", "contexts": ["Rome is the capital of Italy."]},
                "outputs": {"relevant_statements": ["Rome is the capital of Italy."]},
            },
        ]

        self.instructions = (
            "Please extract only sentences from the provided context which are absolutely relevant and "
            "required to answer the following question. If no relevant sentences are found, or if you "
            "believe the question cannot be answered from the given context, return an empty list, example: []"
        )
        self.inputs = [("questions", list[str]), ("contexts", list[list[str]])]
        self.outputs = ["relevant_statements"]
        self.examples = examples or _DEFAULT_EXAMPLES
        self.num_processes = num_processes

        super().__init__(
            instructions=self.instructions,
            inputs=self.inputs,
            outputs=self.outputs,
            examples=self.examples,
            chat_generator=chat_generator,
            raise_on_failure=raise_on_failure,
            progress_bar=progress_bar,
        )

    async def _run_single_evaluation(self, input_names_to_values: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Run a single evaluation asynchronously.

        :param input_names_to_values: Input values for a single evaluation
        :returns: Evaluation result or None if failed
        """
        from haystack.components.builders import PromptBuilder
        from haystack.dataclasses.chat_message import ChatMessage
        import json

        # Build prompt
        template = self.prepare_template()
        builder = PromptBuilder(template=template)
        prompt = builder.run(**input_names_to_values)

        messages = [ChatMessage.from_user(prompt["prompt"])]

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._chat_generator.run, messages
            )
        except Exception as e:
            if self.raise_on_failure:
                raise ValueError(f"Error while generating response for prompt: {prompt}. Error: {e}")
            logger.warning("Error while generating response for prompt: {prompt}. Error: {e}", prompt=prompt, e=e)
            return None

        if self.is_valid_json_and_has_expected_keys(expected=self.outputs, received=result["replies"][0].text):
            parsed_result = json.loads(result["replies"][0].text)
            return parsed_result
        else:
            return None

    def run(self, **inputs) -> dict[str, Any]:
        """
        Run the async LLM evaluator with parallel processing.

        :param questions: A list of questions.
        :param contexts: A list of lists of contexts. Each list of contexts corresponds to one question.
        :returns: A dictionary with evaluation results.
        """
        from statistics import mean

        self.validate_input_parameters(dict(self.inputs), inputs)

        # Prepare input data
        input_names, values = inputs.keys(), list(zip(*inputs.values()))
        list_of_input_names_to_values = [dict(zip(input_names, v)) for v in values]

        # Get number of processes from config if not specified
        if self.num_processes is None:
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                import config
                self.num_processes = config.profiles[0].get("num_data_processes", 4)
            except (ImportError, AttributeError, IndexError):
                self.num_processes = 4  # fallback

        # Limit concurrency to prevent overwhelming the API
        semaphore = asyncio.Semaphore(self.num_processes)

        async def run_with_semaphore(input_data: dict[str, Any]) -> Optional[dict[str, Any]]:
            async with semaphore:
                return await self._run_single_evaluation(input_data)

        async def run_all_evaluations():
            tasks = [run_with_semaphore(input_data) for input_data in list_of_input_names_to_values]
            return await asyncio.gather(*tasks, return_exceptions=True)

        # Run evaluations in parallel
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            raw_results = loop.run_until_complete(run_all_evaluations())
        finally:
            loop.close()

        # Process results
        results = []
        errors = 0
        metadata = []

        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                logger.warning(f"Exception in evaluation {i}: {res}")
                results.append(None)
                errors += 1
            elif res is None:
                results.append(None)
                errors += 1
            else:
                results.append(res)
                # Note: metadata handling would need to be adapted for async

        if errors > 0:
            logger.warning(
                "Async LLM evaluator failed for {errors} out of {len(list_of_input_names_to_values)} inputs.",
                errors=errors,
                len=len(list_of_input_names_to_values),
            )

        # Post-process results (same as ContextRelevanceEvaluator)
        for idx, res in enumerate(results):
            if res is None:
                results[idx] = {"relevant_statements": [], "score": float("nan")}
                continue
            if len(res["relevant_statements"]) > 0:
                res["score"] = 1
            else:
                res["score"] = 0

        # Calculate average context relevance score over all queries
        valid_scores = [res["score"] for res in results if not np.isnan(res["score"])]
        score = mean(valid_scores) if valid_scores else float("nan")
        individual_scores = [res["score"] for res in results]

        return {
            "results": results,
            "score": score,
            "individual_scores": individual_scores,
            "meta": metadata or None
        }


@MetricRegistry.register_component_metric("retriever")
class HaystackContextRelevanceMetric(BaseMetric):
    """
    Wrapper for Haystack's ContextRelevanceEvaluator.
    
    This metric uses Haystack's built-in evaluator to measure how relevant the 
    retrieved documents are to the query.
    """
    
    def __init__(
        self, 
        threshold: float = 0.5, 
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_params: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        raise_on_failure: bool = False,
        progress_bar: bool = False
    ):
        """
        Initialize the ContextRelevanceMetric.
        
        Args:
            threshold: Minimum score for the evaluation to be considered successful
            api_key: OpenAI API key (will use env var OPENAI_API_KEY if None)
            model: Model to use (defaults to gpt-4o-mini)
            api_params: Additional parameters for the API call
            examples: Few-shot examples to improve evaluation quality
            raise_on_failure: Whether to raise an exception on API call failure
            progress_bar: Whether to show a progress bar during evaluation
        """
        super().__init__(threshold=threshold)

        # Initialize the async Haystack evaluator
        self.evaluator = AsyncContextRelevanceEvaluator(
            examples=examples,
            raise_on_failure=raise_on_failure,
            progress_bar=progress_bar
        )
    
    def run(
        self, 
        component_outputs: List[Dict[str, Any]], 
        queries: List[str] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the evaluation.
        
        Args:
            component_outputs: Outputs from the retriever component
            queries: List of queries corresponding to the outputs
            **kwargs: Additional arguments
            
        Returns:
            Evaluation results
        """
        if not queries or len(queries) != len(component_outputs):
            raise ValueError("Queries must be provided and match the number of component outputs")
        
        # Convert component outputs to contexts
        contexts = []
        
        for output in component_outputs:
            if "documents" not in output:
                raise ValueError("Component output must contain 'documents' key")
            
            documents = output["documents"]
            if not documents:
                contexts.append([""])  # Empty context
                continue
            
            # Extract text from documents
            context_texts = [doc.content for doc in documents]
            contexts.append(context_texts)
        
        # Run the Haystack evaluator
        result = self.evaluator.run(questions=queries, contexts=contexts)
        
        # Convert result to our metric format
        self.score = result.get("score", 0.0)
        self.success = self.score >= self.threshold
        
        return {
            "score": self.score,
            "success": self.success,
            "individual_scores": result.get("individual_scores", []),
            "results": result.get("results", [])
        }
    
@MetricRegistry.register_component_metric("retriever")
class MAPAtKMetric(BaseMetric):
    """
    Mean Average Precision at K (MAP@K) metric.
    
    This metric evaluates the performance of a retrieval system by measuring
    the mean of the average precision scores for each query, taking into account
    the ranking of retrieved documents. Unlike simple precision, MAP@K considers
    the position of relevant documents in the result list.
    """
    
    def __init__(
        self, 
        threshold: float = 0.5, 
        k: int = 5,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_params: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        raise_on_failure: bool = False,
        progress_bar: bool = False
    ):
        """
        Initialize the MAP@K metric.
        
        Args:
            threshold: Minimum score for the evaluation to be considered successful
            k: The maximum number of retrieved documents to consider
            api_key: OpenAI API key (will use env var OPENAI_API_KEY if None)
            model: Model to use for relevance evaluation
            api_params: Additional parameters for the API call
            examples: Few-shot examples to improve evaluation quality
            raise_on_failure: Whether to raise an exception on API call failure
            progress_bar: Whether to show a progress bar during evaluation
        """
        super().__init__(threshold=threshold)
        self.k = k
        
        # Initialize the async Haystack evaluator for document relevance judgments
        self.evaluator = AsyncContextRelevanceEvaluator(
            examples=examples,
            raise_on_failure=raise_on_failure,
            progress_bar=progress_bar
        )
        
        logger.info(f"Initialized MAP@{k} metric")
    
    def _calculate_average_precision(self, relevance_judgments: List[int]) -> float:
        """
        Calculate the Average Precision for a single query.
        
        Args:
            relevance_judgments: A list of binary relevance judgments (0 or 1)
                                 for each retrieved document, in order of retrieval.
                                 
        Returns:
            Average Precision score
        """
        if not relevance_judgments or sum(relevance_judgments) == 0:
            return 0.0
        
        # Limit to top-k documents
        relevance_judgments = relevance_judgments[:self.k]
        
        # Calculate precision at each position of a relevant document
        precisions = []
        num_relevant_so_far = 0
        
        for i, is_relevant in enumerate(relevance_judgments):
            if is_relevant:
                num_relevant_so_far += 1
                # Precision@i+1 = number of relevant docs up to position i+1 / (i+1)
                precision_at_i = num_relevant_so_far / (i + 1)
                precisions.append(precision_at_i)
        
        # Average precision is the mean of precisions at each relevant document
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(precisions)
    
    def run(
        self, 
        component_outputs: List[Dict[str, Any]], 
        queries: List[str] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the MAP@K evaluation.
        
        Args:
            component_outputs: Outputs from the retriever component
            queries: List of queries corresponding to the outputs
            **kwargs: Additional arguments
            
        Returns:
            Evaluation results with MAP@K score
        """
        if not queries or len(queries) != len(component_outputs):
            raise ValueError("Queries must be provided and match the number of component outputs")
        
        # Convert component outputs to contexts for evaluation
        all_contexts = []
        
        for output in component_outputs:
            if "documents" not in output:
                raise ValueError("Component output must contain 'documents' key")
            
            documents = output["documents"]
            if not documents:
                all_contexts.append([""])  # Empty context
                continue
            
            # Extract text from documents
            context_texts = [doc.content for doc in documents]
            all_contexts.append(context_texts)
        
        # Run the Haystack evaluator to get document relevance
        haystack_result = self.evaluator.run(questions=queries, contexts=all_contexts)

        # Process results for MAP@K calculation
        average_precisions = []
        detailed_results = []
        
        for i, (query, contexts, result) in enumerate(zip(queries, all_contexts, haystack_result.get("results", []))):
            # Skip if result is None or no relevant statements
            if result is None or "relevant_statements" not in result:
                logger.error("Failed to get relevance judgments from LLM")
                raise ValueError("Could not evaluate document relevance - LLM evaluation failed")
            
            # Extract relevance judgments for each document
            relevance_judgments = []
            
            for j, context in enumerate(contexts[:self.k]):
                # A document is relevant if it has at least one relevant statement
                has_relevant_statements = self._document_has_relevant_statements(context, result["relevant_statements"])
                relevance_judgments.append(1 if has_relevant_statements else 0)
            
            # Calculate Average Precision for this query
            ap_score = self._calculate_average_precision(relevance_judgments)
            average_precisions.append(ap_score)
            
            # Record detailed results
            detailed_results.append({
                "query": query,
                "relevance_judgments": relevance_judgments[:self.k],
                "ap_score": ap_score
            })
            
            logger.info(f"Query: '{query[:50]}...', AP@{self.k}: {ap_score:.2f}")
        
        # Calculate Mean Average Precision (MAP)
        self.score = np.mean(average_precisions) if average_precisions else 0.0
        self.success = self.score >= self.threshold
        
        return {
            "score": self.score,
            "success": self.success,
            "individual_scores": average_precisions,
            "detailed_results": detailed_results,
            "metric": f"MAP@{self.k}"
        }

    def _split_string_into_tuples(self, string: str) -> List[str]:
        """
        Split a string into tuples of 2 elements.
        """
        return [string[i:i+2] for i in range(0, len(string), 2)]

    def _calc_jaccard_similarity(self, string1: str, string2: str) -> float:
        """
        Calculate the Jaccard similarity between two strings.
        """
        set1 = set(self._split_string_into_tuples(string1))
        set2 = set(self._split_string_into_tuples(string2))
        return len(set1 & set2) / len(set1 | set2)
    
    def _statements_are_equal(self, context: str, document: str) -> bool:
        """
        Check if the context equals the document.
        """
        return self._calc_jaccard_similarity(context, document) >= 0.98

    def _document_has_relevant_statements(self, context: str, relevant_statements: List[str]) -> bool:
        """
        Check if the document is in the relevant statements.
        """
        return any(
            self._statements_are_equal(context_sentence, statement_sentence) 
            for statement in relevant_statements
            for statement_sentence in statement.split(".")
            for context_sentence in context.split(".")
            if len(statement_sentence) > 20 and len(context_sentence) > 20
        )


if __name__ == "__main__":
    import os
    from haystack import Document
    
    print("=== MAP@K Manual Verification Test with Real Documents ===")
    print("\nThis test demonstrates how Average Precision (AP) at K is calculated")
    print("and how it forms Mean Average Precision (MAP) at K using real documents.\n")
    
    # Test query about renewable energy
    query = "What are the benefits of solar energy?"
    print(f"Query: '{query}'\n")
    
    # Create 6 real small documents with 1 sentence each
    documents = [
        Document(
            content="Solar energy reduces electricity bills and has a lower carbon footprint.",
            meta={"id": "doc1", "is_relevant": True}  # Highly relevant to query
        ),
        Document(
            content="Solar panels can be installed on rooftops or in large solar farms.",
            meta={"id": "doc2", "is_relevant": True}  # Relevant to query
        ),
        Document(
            content="Wind energy is another renewable energy source that uses turbines.",
            meta={"id": "doc3", "is_relevant": False}  # Not directly relevant to solar
        ),
        Document(
            content="The sun provides enough energy in one hour to power the world for a year.",
            meta={"id": "doc4", "is_relevant": True}  # Relevant fact about solar
        ),
        Document(
            content="Football is a popular sport played by many people.",
            meta={"id": "doc5", "is_relevant": False}  # Not relevant to query
        ),
        Document(
            content="Basketball is a popular sport played by many people.",
            meta={"id": "doc6", "is_relevant": False}  # Not directly relevant to solar
        )
    ]
    
    # Create a MAP@K calculator (k=6 to include all documents)
    k = 6
    map_calculator = MAPAtKMetric(k=k)
    
    from random import shuffle

    for _ in range(3):
        print("--------------------------------")
        shuffle(documents)
        for doc in documents:
            print(doc.content)
        result = map_calculator.run([{"documents": documents}], [query])
        print(result)
