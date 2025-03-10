from typing import Dict, Any, List, Union, Callable, Optional
from abc import ABC, abstractmethod
from functools import lru_cache
from dotenv import load_dotenv
import logging
import re
from haystack import Document
from haystack.components.generators import OpenAIGenerator, HuggingFaceAPIGenerator
from haystack.components.evaluators import ContextRelevanceEvaluator
from haystack.utils import Secret
from ragnroll.metrics.base import BaseMetric, MetricRegistry

load_dotenv()
# Import RAGAS components
logger = logging.getLogger(__name__)

class BaseModelProvider(ABC):
    """Abstract base class for model providers."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text based on the provided prompt."""
        pass

class OpenAIModelProvider(BaseModelProvider):
    """Uses OpenAI models for text generation."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18", api_key: Optional[str] = None):
        self.generator = OpenAIGenerator(
            model=model,
            generation_kwargs={"temperature": 0.0}
        )
    
    def generate(self, prompt: str) -> str:
        return self.generator.run(prompt=prompt)["replies"][0]

class HuggingFaceModelProvider(BaseModelProvider):
    """Uses Hugging Face models for text generation."""
    
    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.2", token: Optional[str] = None):
        self.generator = HuggingFaceAPIGenerator(
            api_type="serverless_inference_api",
            api_params={"model": model},
            token=Secret.from_token(token) if token else None,
            generation_kwargs={"temperature": 0.0}
        )
    
    def generate(self, prompt: str) -> str:
        return self.generator.run(prompt=prompt)["replies"][0]

class ModelProviderFactory:
    """Factory for creating model providers."""
    
    @staticmethod
    def get_provider(provider_type: str, model: str, **kwargs) -> BaseModelProvider:
        """
        Get a model provider.
        
        Args:
            provider_type: Type of provider ('openai' or 'huggingface')
            model: Model name or ID
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            A model provider instance
        """
        if provider_type.lower() == 'openai':
            return OpenAIModelProvider(model=model, **kwargs)
        elif provider_type.lower() in ['huggingface', 'hf']:
            return HuggingFaceModelProvider(model=model, **kwargs)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

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
        
        # Initialize the Haystack evaluator
        self.evaluator = ContextRelevanceEvaluator(
            api_key=api_key,
            api_params=api_params or ({} if model is None else {"model": model}),
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
class LLMContextPrecisionMetric(BaseMetric):
    """
    Custom Context Precision Metric using LLM as a judge.
    
    This metric evaluates how relevant each retrieved context is to the query
    and expected response using a language model to make the judgment.
    """
    
    def __init__(
        self, 
        threshold: float = 0.5,
        provider_type: str = "openai",
        model: str = "gpt-4o-mini-2024-07-18",
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LLMContextPrecisionMetric.
        
        Args:
            threshold: Minimum score for the evaluation to be considered successful
            provider_type: Type of model provider ('openai' or 'huggingface')
            model: Model to use for evaluation 
            api_key: API key for OpenAI (if using OpenAI provider)
            token: API token for HuggingFace (if using HuggingFace provider)
            **kwargs: Additional parameters
        """
        super().__init__(threshold=threshold)
            
        self.model_provider = ModelProviderFactory.get_provider(
            provider_type=provider_type,
            model=model,
        )
        
        logger.info(f"Initialized LLMContextPrecisionMetric with {provider_type} model: {model}")
        
    def _is_context_relevant(self, query: str, context: str, expected_output: str) -> bool:
        """
        Use LLM to judge if a context is relevant to the query and expected output.
        
        Args:
            query: The user query
            context: The retrieved context
            expected_output: The expected answer
            
        Returns:
            Boolean indicating whether the context is relevant
        """
        prompt = f"""
You are an expert judge evaluating retrieved context for Retrieval Augmented Generation (RAG) systems.

I will give you a QUERY, a piece of CONTEXT that was retrieved for this query, and the EXPECTED ANSWER.

Your task is to determine if the CONTEXT is RELEVANT to both the QUERY and the EXPECTED ANSWER.
A relevant context contains information that directly helps to answer the query correctly or contains facts used in the expected answer.

QUERY: {query}

CONTEXT: {context}

EXPECTED ANSWER: {expected_output}

Is this context relevant for answering the query? 
Think step by step and analyze whether the context provides information helpful to directly answer the query.
After your analysis, answer with either "RELEVANT" or "NOT RELEVANT" only.
"""
        
        try:
            response = self.model_provider.generate(prompt).strip()
            # Extract just the judgment from potentially longer responses
            if "RELEVANT" in response and "NOT RELEVANT" not in response:
                return True
            elif "NOT RELEVANT" in response:
                return False
            else:
                # Extract judgment using regex as fallback
                result = re.search(r"(RELEVANT|NOT RELEVANT)", response)
                if result:
                    return result.group(0) == "RELEVANT"
                # Default to not relevant if we can't determine
                logger.warning(f"Could not determine relevance from LLM response: {response}")
                return False
        except Exception as e:
            logger.error(f"Error when judging context relevance: {e}")
            return False
    
    def run(
        self, 
        component_outputs: List[Dict[str, Any]], 
        queries: List[str] = None, 
        expected_outputs: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the evaluation.
        
        Args:
            component_outputs: Outputs from the retriever component
            queries: List of queries corresponding to the outputs
            expected_outputs: List of expected outputs (ground truth answers)
            **kwargs: Additional arguments
            
        Returns:
            Evaluation results
        """
        if not queries or len(queries) != len(component_outputs):
            raise ValueError("Queries must be provided and match the number of component outputs")
            
        if not expected_outputs or len(expected_outputs) != len(component_outputs):
            raise ValueError("Expected outputs must be provided and match the number of component outputs")
        
        individual_scores = []
        detailed_results = []
        
        # Process each query and its corresponding output
        for query, actual_output, expected_output in zip(queries, component_outputs, expected_outputs):
            if "documents" not in actual_output:
                raise ValueError("Component output must contain 'documents' key")
            
            documents = actual_output["documents"]
            if not documents:
                individual_scores.append(0.0)
                detailed_results.append({
                    "query": query,
                    "context_judgments": [],
                    "precision_score": 0.0
                })
                continue
            
            # Evaluate relevance of each context
            context_judgments = []
            for doc in documents:
                is_relevant = self._is_context_relevant(query, doc.content, expected_output)
                context_judgments.append({
                    "content": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
                    "is_relevant": is_relevant
                })
            
            # Calculate precision score (relevant contexts / total contexts)
            relevant_count = sum(1 for judgment in context_judgments if judgment["is_relevant"])
            precision_score = relevant_count / len(context_judgments) if context_judgments else 0.0
            
            individual_scores.append(precision_score)
            detailed_results.append({
                "query": query,
                "context_judgments": context_judgments,
                "precision_score": precision_score
            })
            
            logger.info(f"Query: '{query[:50]}...', Precision: {precision_score:.2f} ({relevant_count}/{len(context_judgments)} relevant)")
        
        # Calculate average score
        avg_score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
        
        self.score = avg_score
        self.success = self.score >= self.threshold
        
        return {
            "score": self.score,
            "success": self.success,
            "individual_scores": individual_scores,
            "detailed_results": detailed_results
        }
