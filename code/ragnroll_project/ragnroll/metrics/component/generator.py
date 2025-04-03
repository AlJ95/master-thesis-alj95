from typing import Dict, Any, List, Union, Optional
import re
import logging

try:
    from ragnroll.metrics.base import BaseMetric, MetricRegistry
    from ragnroll.utils.llm_judge import LLMAsAJudge
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base import BaseMetric, MetricRegistry
    # Handle utils import for development environment
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.llm_judge import LLMAsAJudge

logger = logging.getLogger(__name__)

@MetricRegistry.register_component_metric("generator")
class FormatValidatorMetric(BaseMetric):
    """
    Evaluates if the generator's output follows the expected format for classification tasks.
    
    This metric checks if the output contains the required answer format string,
    such as 'The answer is "True"' or similar pattern.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        positive_pattern: str = r'The answer is ["\'](True|Yes|Valid|1)["\']',
        negative_pattern: str = r'The answer is ["\'](False|No|Invalid|0)["\']',
        case_sensitive: bool = False
    ):
        """
        Initialize the format validator metric.
        
        Args:
            threshold: Minimum score for the evaluation to be considered successful
            positive_pattern: Regex pattern for positive answers
            negative_pattern: Regex pattern for negative answers
            case_sensitive: Whether the pattern matching should be case-sensitive
        """
        super().__init__(threshold=threshold)
        
        self.positive_pattern = positive_pattern
        self.negative_pattern = negative_pattern
        self.case_sensitive = case_sensitive
        
        # Compile regex patterns with appropriate flags
        flags = 0 if case_sensitive else re.IGNORECASE
        self.positive_regex = re.compile(positive_pattern, flags)
        self.negative_regex = re.compile(negative_pattern, flags)
    
    def _check_format(self, output: str) -> bool:
        """
        Check if the output follows the expected format.
        
        Args:
            output: Generated output to check
            
        Returns:
            True if the output follows the expected format, False otherwise
            Returns False if BOTH patterns match (inconsistent output)
        """
        positive_match = bool(self.positive_regex.search(output))
        negative_match = bool(self.negative_regex.search(output))
        
        # Return False if both patterns match (inconsistent output)
        if positive_match and negative_match:
            logger.warning(f"Output contains both positive and negative patterns: {output[:100]}")
            return False
            
        # Return True if exactly one pattern matches
        if positive_match or negative_match:
            return True
            
        logger.warning(f"Output does not follow the expected format: {output[:100]}")
        return False
    
    def run(
        self,
        component_outputs: List[Dict[str, Any]],
        expected_outputs: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the format validation.
        
        Args:
            component_outputs: Outputs from the generator component
            expected_outputs: Expected outputs (not used for format validation)
            **kwargs: Additional arguments
            
        Returns:
            Evaluation results
        """
        if not component_outputs:
            raise ValueError("Component outputs cannot be empty")
        
        # Extract actual outputs from component outputs
        actual_outputs = []
        for output in component_outputs:
            if "replies" in output:
                # Handle LLM component output
                actual_outputs.append(output["replies"][0])
            elif "answer" in output:
                # Handle AnswerBuilder component output
                actual_outputs.append(output["answer"])
            else:
                raise ValueError(f"Unknown generator output format: {output}")
        
        # Validate format of each output
        format_valid = []
        detailed_results = []
        
        for i, output in enumerate(actual_outputs):
            is_valid = self._check_format(output)
            format_valid.append(is_valid)
            
            detailed_results.append({
                "output": output[:100] + "..." if len(output) > 100 else output,
                "is_valid_format": is_valid
            })
        
        # Calculate score as percentage of valid formats
        self.score = sum(format_valid) / len(format_valid) if format_valid else 0.0
        self.success = self.score >= self.threshold
        
        return {
            "score": self.score,
            "success": self.success,
            "individual_scores": format_valid,
            "detailed_results": detailed_results
        }


class JudgeBasedMetric(BaseMetric):
    """
    Base class for metrics that use an LLM as a judge.
    
    This class provides common functionality for creating and using
    an LLM judge for evaluation.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs
    ):
        """
        Initialize the judge-based metric.
        
        Args:
            threshold: Minimum score for the evaluation to be considered successful
            api_key: API key for the LLM provider
            api_url: Base URL for the API (if None, uses OpenAI's default URL)
            model: Model name to use
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for the LLM API
        """
        super().__init__(threshold=threshold)
        
        # Create LLM judge with simplified parameters
        # If api_url is None, it will use OpenAI's default URL
        self.judge = LLMAsAJudge(
            api_key=api_key,
            api_url=api_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def _extract_predicted_answers(self, component_outputs: List[Dict[str, Any]]) -> List[str]:
        """
        Extract answers from component outputs.
        
        Args:
            component_outputs: Outputs from the generator component
            
        Returns:
            List of extracted answers
        """
        predicted_answers = []
        for output in component_outputs:
            if "replies" in output:
                # Handle LLM component output
                predicted_answers.append(output["replies"][0])
            elif "answer" in output:
                # Handle AnswerBuilder component output
                predicted_answers.append(output["answer"])
            else:
                raise ValueError(f"Unknown generator output format: {output}")
        return predicted_answers


@MetricRegistry.register_component_metric("generator")
class ContextUtilizationMetric(JudgeBasedMetric):
    """
    Evaluates how well the generator utilizes the provided context.
    
    This metric assesses whether the generated answer can be derived from
    the given context, using an LLM as a judge.
    """
    
    def run(
        self,
        component_outputs: List[Dict[str, Any]],
        queries: List[str] = None,
        contexts: List[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the context utilization evaluation.
        
        Args:
            component_outputs: Outputs from the generator component
            queries: List of queries corresponding to the outputs
            contexts: List of contexts for each query (required)
            **kwargs: Additional arguments
            
        Returns:
            Evaluation results
        """
        if not queries or len(queries) != len(component_outputs):
            raise ValueError("Queries must be provided and match the number of component outputs")
        
        if not contexts:
            raise ValueError("Contexts must be provided for context utilization evaluation")
            
        if len(contexts) != len(component_outputs):
            raise ValueError(f"Number of contexts ({len(contexts)}) must match number of component outputs ({len(component_outputs)})")
        
        # Check for empty contexts
        empty_contexts = [i for i, ctx_list in enumerate(contexts) if not ctx_list]
        if empty_contexts:
            raise ValueError(f"Empty contexts found at indices: {empty_contexts}. All queries must have associated contexts for evaluation.")
        
        # Extract generated answers
        predicted_answers = self._extract_predicted_answers(component_outputs)
        
        # Evaluate each answer
        individual_scores = []
        detailed_results = []
        
        for query, answer, context_list in zip(queries, predicted_answers, contexts):
            # Combine contexts into a single string with separators
            context_text = "\n\n".join([context.content for context in context_list])
            
            # Create evaluation prompt
            prompt = f"""Evaluate how well the following answer utilizes the provided context.
            
Question: {query}

Context:
{context_text}

Answer: {answer}

Evaluate the answer's context utilization on a scale from 0 to 1, where:
0 = No context utilization, answer is completely disconnected from the provided context
0.5 = Moderate context utilization, answer uses some information from the context but misses key points
1 = Excellent context utilization, answer effectively uses all and only relevant information from the context

Return just a single number between 0 and 1."""
            
            try:
                # Get rating from judge
                score = self.judge.judge_rating(prompt, expected_rating_format="0-1 float")
                individual_scores.append(score)
                
                detailed_results.append({
                    "query": query,
                    "answer": answer,
                    "context": context_text[:500] + "..." if len(context_text) > 500 else context_text,
                    "score": score
                })
            except Exception as e:
                logger.error(f"Error evaluating context utilization: {e}")
                individual_scores.append(0.0)
                detailed_results.append({
                    "query": query,
                    "answer": answer,
                    "context": context_text[:100] + "..." if len(context_text) > 100 else context_text,
                    "score": 0.0,
                    "error": str(e)
                })
        
        # Calculate overall score
        self.score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
        self.success = self.score >= self.threshold
        
        return {
            "score": self.score,
            "success": self.success,
            "individual_scores": individual_scores,
            "detailed_results": detailed_results
        }


@MetricRegistry.register_component_metric("generator")
class AnswerRelevancyMetric(JudgeBasedMetric):
    """
    Evaluates how relevant the generated answer is to the query.
    
    This metric uses an LLM-as-a-Judge approach to determine if the generated
    answer is relevant to the question, independently of the context.
    """
    
    def run(
        self,
        component_outputs: List[Dict[str, Any]],
        queries: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the answer relevancy evaluation.
        
        Args:
            component_outputs: Outputs from the generator component
            queries: List of queries corresponding to the outputs
            **kwargs: Additional arguments
            
        Returns:
            Evaluation results
        """
        if not queries or len(queries) != len(component_outputs):
            raise ValueError("Queries must be provided and match the number of component outputs")
        
        # Extract generated answers
        predicted_answers = self._extract_predicted_answers(component_outputs)
        
        # Evaluate each answer
        individual_scores = []
        detailed_results = []
        
        for query, answer in zip(queries, predicted_answers):
            # Create evaluation prompt
            prompt = f"""Evaluate the relevance of the answer to the question on a scale from 0 to 1.
            
Question: {query}
Answer: {answer}

Your evaluation should be based solely on whether the answer addresses what the question is asking for,
regardless of factual accuracy or context. Consider:

1. Does the answer address the specific question asked?
2. Is the answer on-topic?
3. Does the answer provide information relevant to what the user wants to know?

Provide your evaluation as a single number between 0 and 1, where:
0 = Completely irrelevant
0.5 = Somewhat relevant but missing key points
1 = Highly relevant

Your evaluation (single number between 0 and 1):"""
            
            try:
                # Get rating from judge
                score = self.judge.judge_rating(prompt, expected_rating_format="0-1 float")
                individual_scores.append(score)
                
                detailed_results.append({
                    "query": query,
                    "answer": answer,
                    "score": score
                })
            except Exception as e:
                logger.error(f"Error evaluating answer relevancy: {e}")
                individual_scores.append(0.0)
                detailed_results.append({
                    "query": query,
                    "answer": answer,
                    "score": 0.0,
                    "error": str(e)
                })
        
        # Calculate overall score
        self.score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
        self.success = self.score >= self.threshold
        
        return {
            "score": self.score,
            "success": self.success,
            "individual_scores": individual_scores,
            "detailed_results": detailed_results
        } 