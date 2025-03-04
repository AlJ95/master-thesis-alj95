from typing import Dict, Any, List, Union, Callable, Optional
from abc import ABC, abstractmethod
from functools import lru_cache
import logging
from haystack import Document
from haystack.components.generators import OpenAIGenerator
from ragnroll.metrics.base import BaseMetric, MetricRegistry

logger = logging.getLogger(__name__)

class DocumentRelevanceEvaluator(ABC):
    """
    Base class for evaluating document relevance.
    
    This class provides methods to determine if retrieved documents are relevant
    to the ground truth documents or query.
    """
    
    @abstractmethod
    def is_relevant(self, retrieved_doc: Document, ground_truth_docs: List[Document], query: str) -> bool:
        """
        Determine if a retrieved document is relevant.
        
        Args:
            retrieved_doc: The document to evaluate
            ground_truth_docs: List of ground truth documents
            query: The original query
            
        Returns:
            bool: True if the document is relevant, False otherwise
        """
        pass
    
    def classify_documents(
        self, 
        retrieved_docs: List[Document], 
        ground_truth_docs: List[Document], 
        query: str
    ) -> Dict[str, List[Document]]:
        """
        Classify retrieved documents as relevant or irrelevant.
        
        Args:
            retrieved_docs: List of retrieved documents
            ground_truth_docs: List of ground truth documents
            query: The original query
            
        Returns:
            Dict with 'relevant' and 'irrelevant' lists of documents
        """
        result = {
            "relevant": [],
            "irrelevant": []
        }
        
        for doc in retrieved_docs:
            if self.is_relevant(doc, ground_truth_docs, query):
                result["relevant"].append(doc)
            else:
                result["irrelevant"].append(doc)
                
        return result


class LLMRelevanceEvaluator(DocumentRelevanceEvaluator):
    """Uses an LLM to determine document relevance."""
    
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """
        Initialize with OpenAI API key and model.
        
        Args:
            model: Model to use for relevance determination
        """
        self.generator = OpenAIGenerator(model=model)
        
    @lru_cache(maxsize=1000)
    def _generate(self, prompt: str):
        return self.generator.run(prompt=prompt)

    def is_relevant(self, retrieved_doc: Document, ground_truth_docs: List[Document], query: str) -> bool:
        """
        Use LLM to determine if retrieved document is relevant to the query and ground truth.
        
        Args:
            retrieved_doc: The document to evaluate
            ground_truth_docs: List of ground truth documents
            query: The original query
            
        Returns:
            bool: True if the document is relevant, False otherwise
        """
        if not retrieved_doc.content:
            return False
            
        # Combine ground truth contents
        gt_contents = "\n".join([doc.content for doc in ground_truth_docs if doc.content])
        
        prompt = f"""
        Query: {query}
        
        Ground Truth Information:
        {gt_contents}
        
        Retrieved Document:
        {retrieved_doc.content}
        
        Is the retrieved document relevant to answering the query based on the ground truth information?
        Answer only 'Yes' or 'No'.
        """
        
        try:
            result = self._generate(prompt=prompt)
            answer = result["replies"][0].lower()
            return "yes" in answer
        except Exception as e:
            logger.error(f"Error using LLM for relevance evaluation: {e}")


@MetricRegistry.register_component_metric("retriever")
class RetrievalPrecisionMetric(BaseMetric):
    """
    Measures the precision of the retrieval component.
    
    Precision = Relevant Retrieved / Total Retrieved
    """
    
    def __init__(self, threshold: float = 0.5, relevance_evaluator: Optional[DocumentRelevanceEvaluator] = None):
        """
        Initialize the precision metric.
        
        Args:
            threshold: Threshold for success
            relevance_evaluator: Custom evaluator for document relevance
        """
        super().__init__(threshold=threshold)
        self.relevance_evaluator = relevance_evaluator or LLMRelevanceEvaluator()
    
    def run(self, component_output: Dict[str, Any], ground_truth_docs: List[Document] = None, 
            query: str = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate precision of retrieved documents.
        
        Args:
            component_output: Output from the retriever component
            ground_truth_docs: List of ground truth documents
            query: The original query
            
        Returns:
            Dict[str, Any]: Results with score and details
        """
        try:
            # Get documents from the retriever output
            retrieved_docs = component_output.get("documents", [])
            
            if not retrieved_docs:
                print("\n\nNo documents retrieved")
                print(component_output)
                print("\n\n")
                return {
                    "score": 0.0,
                    "success": False,
                    "details": {"error": "No documents retrieved"}
                }
            
            # If no ground truth is provided, try to use expected_output to create a synthetic ground truth
            if not ground_truth_docs and "expected_output" in kwargs:
                ground_truth_docs = [Document(content=kwargs["expected_output"])]
                
            # If no query is provided, try to use input_text
            if not query and "input_text" in kwargs:
                query = kwargs["input_text"]
                
            if not ground_truth_docs or not query:
                return {
                    "score": 0.0,
                    "success": False,
                    "details": {"error": "Missing ground truth documents or query"}
                }
            
            # Classify documents as relevant or irrelevant
            classification = self.relevance_evaluator.classify_documents(
                retrieved_docs, ground_truth_docs, query
            )
            
            # Calculate precision
            relevant_count = len(classification["relevant"])
            total_count = len(retrieved_docs)
            
            precision = relevant_count / total_count if total_count > 0 else 0.0
            
            self.score = precision
            self.success = precision >= self.threshold
            self.details = {
                "relevant_retrieved": relevant_count,
                "total_retrieved": total_count,
                "evaluator": self.relevance_evaluator.__class__.__name__
            }
            
            return {
                "score": self.score,
                "success": self.success,
                "details": self.details
            }
        except Exception as e:
            logger.error(f"Error calculating precision: {e}")
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }


@MetricRegistry.register_component_metric("retriever")
class RetrievalRecallMetric(BaseMetric):
    """
    Measures recall of the retrieval component.
    
    Recall = Relevant Retrieved / Total Relevant
    """
    
    def __init__(self, threshold: float = 0.5, relevance_evaluator: Optional[DocumentRelevanceEvaluator] = None):
        """
        Initialize the recall metric.
        
        Args:
            threshold: Threshold for success
            relevance_evaluator: Custom evaluator for document relevance
        """
        super().__init__(threshold=threshold)
        self.relevance_evaluator = relevance_evaluator or LLMRelevanceEvaluator()
    
    def run(self, component_output: Dict[str, Any], ground_truth_docs: List[Document] = None, 
            query: str = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate recall of retrieved documents.
        
        Args:
            component_output: Output from the retriever component
            ground_truth_docs: List of ground truth documents
            query: The original query
            
        Returns:
            Dict[str, Any]: Results with score and details
        """
        try:
            # Get documents from the retriever output
            retrieved_docs = component_output.get("documents", [])
            
            if not retrieved_docs:
                return {
                    "score": 0.0,
                    "success": False,
                    "details": {"error": "No documents retrieved"}
                }
            
            # If no ground truth is provided, try to use expected_output to create a synthetic ground truth
            if not ground_truth_docs and "expected_output" in kwargs:
                ground_truth_docs = [Document(content=kwargs["expected_output"])]
                
            # If no query is provided, try to use input_text
            if not query and "input_text" in kwargs:
                query = kwargs["input_text"]
                
            if not ground_truth_docs or not query:
                return {
                    "score": 0.0,
                    "success": False,
                    "details": {"error": "Missing ground truth documents or query"}
                }
            
            # Classify documents as relevant or irrelevant
            classification = self.relevance_evaluator.classify_documents(
                retrieved_docs, ground_truth_docs, query
            )
            
            # For recall, we need to know how many ground truth documents were covered
            # This is an approximation since we don't have a 1:1 mapping
            relevant_count = len(classification["relevant"])
            total_relevant = len(ground_truth_docs)
            
            recall = relevant_count / total_relevant if total_relevant > 0 else 0.0
            
            self.score = recall
            self.success = recall >= self.threshold
            self.details = {
                "relevant_retrieved": relevant_count,
                "total_relevant": total_relevant,
                "evaluator": self.relevance_evaluator.__class__.__name__
            }
            
            return {
                "score": self.score,
                "success": self.success,
                "details": self.details
            }
        except Exception as e:
            logger.error(f"Error calculating recall: {e}")
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }


@MetricRegistry.register_component_metric("retriever")
class RetrievalF1Metric(BaseMetric):
    """
    Measures F1 score of the retrieval component.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    
    def __init__(self, threshold: float = 0.5, relevance_evaluator: Optional[DocumentRelevanceEvaluator] = None):
        """
        Initialize the F1 metric.
        
        Args:
            threshold: Threshold for success
            relevance_evaluator: Custom evaluator for document relevance
        """
        super().__init__(threshold=threshold)
        self.relevance_evaluator = relevance_evaluator or LLMRelevanceEvaluator()
        self.precision_metric = RetrievalPrecisionMetric(relevance_evaluator=self.relevance_evaluator)
        self.recall_metric = RetrievalRecallMetric(relevance_evaluator=self.relevance_evaluator)
    
    def run(self, component_output: Dict[str, Any], ground_truth_docs: List[Document] = None, 
            query: str = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate F1 score of retrieved documents.
        
        Args:
            component_output: Output from the retriever component
            ground_truth_docs: List of ground truth documents
            query: The original query
            
        Returns:
            Dict[str, Any]: Results with score and details
        """
        try:
            # Calculate precision and recall
            precision_result = self.precision_metric.run(
                component_output=component_output,
                ground_truth_docs=ground_truth_docs,
                query=query,
                **kwargs
            )
            
            recall_result = self.recall_metric.run(
                component_output=component_output,
                ground_truth_docs=ground_truth_docs,
                query=query,
                **kwargs
            )
            
            precision = precision_result["score"]
            recall = recall_result["score"]
            
            # Calculate F1
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            self.score = f1
            self.success = f1 >= self.threshold
            self.details = {
                "precision": precision,
                "recall": recall,
                "precision_details": precision_result["details"],
                "recall_details": recall_result["details"],
                "evaluator": self.relevance_evaluator.__class__.__name__
            }
            
            return {
                "score": self.score,
                "success": self.success,
                "details": self.details
            }
        except Exception as e:
            logger.error(f"Error calculating F1: {e}")
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }


@MetricRegistry.register_component_metric("retriever")
class RetrievalMAPMetric(BaseMetric):
    """
    Measures Mean Average Precision (MAP) of the retrieval component.
    
    MAP evaluates the ranking of relevant documents in the retrieval results.
    """
    
    def __init__(self, threshold: float = 0.5, relevance_evaluator: Optional[DocumentRelevanceEvaluator] = None):
        """
        Initialize the MAP metric.
        
        Args:
            threshold: Threshold for success
            relevance_evaluator: Custom evaluator for document relevance
        """
        super().__init__(threshold=threshold)
        self.relevance_evaluator = relevance_evaluator or LLMRelevanceEvaluator()
    
    def run(self, component_output: Dict[str, Any], ground_truth_docs: List[Document] = None, 
            query: str = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate MAP of retrieved documents.
        
        Args:
            component_output: Output from the retriever component
            ground_truth_docs: List of ground truth documents
            query: The original query
            
        Returns:
            Dict[str, Any]: Results with score and details
        """
        try:
            # Get documents from the retriever output
            retrieved_docs = component_output.get("documents", [])
            
            if not retrieved_docs:
                return {
                    "score": 0.0,
                    "success": False,
                    "details": {"error": "No documents retrieved"}
                }
            
            # If no ground truth is provided, try to use expected_output to create a synthetic ground truth
            if not ground_truth_docs and "expected_output" in kwargs:
                ground_truth_docs = [Document(content=kwargs["expected_output"])]
                
            # If no query is provided, try to use input_text
            if not query and "input_text" in kwargs:
                query = kwargs["input_text"]
                
            if not ground_truth_docs or not query:
                return {
                    "score": 0.0,
                    "success": False,
                    "details": {"error": "Missing ground truth documents or query"}
                }
            
            # Calculate Average Precision
            average_precision = 0.0
            relevant_count = 0
            
            for rank, doc in enumerate(retrieved_docs):
                if self.relevance_evaluator.is_relevant(doc, ground_truth_docs, query):
                    relevant_count += 1
                    # Precision at current rank
                    precision_at_k = relevant_count / (rank + 1)
                    average_precision += precision_at_k
            
            # Normalize by the number of relevant documents
            if relevant_count > 0:
                average_precision /= relevant_count
            else:
                average_precision = 0.0
            
            self.score = average_precision
            self.success = average_precision >= self.threshold
            self.details = {
                "relevant_retrieved": relevant_count,
                "total_retrieved": len(retrieved_docs),
                "evaluator": self.relevance_evaluator.__class__.__name__
            }
            
            return {
                "score": self.score,
                "success": self.success,
                "details": self.details
            }
        except Exception as e:
            logger.error(f"Error calculating MAP: {e}")
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }
