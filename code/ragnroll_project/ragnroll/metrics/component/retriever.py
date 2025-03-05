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
    
    def run(self, component_outputs: List[Dict[str, Any]], expected_outputs: List[str] = None,
            queries: List[str] = None, expected_retrievals: List[List[Document]] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate precision of retrieved documents across multiple queries.
        
        Args:
            component_outputs: List of outputs from the retriever component
            expected_outputs: List of expected outputs (used as fallback for ground truth)
            queries: List of original queries
            expected_retrievals: List of expected retrieval document lists
            
        Returns:
            Dict[str, Any]: Results with score and details
        """
        try:
            if not component_outputs:
                return {
                    "score": 0.0,
                    "success": False,
                    "details": {"error": "No component outputs provided"}
                }
                
            # Track results for each query
            precision_scores = []
            total_relevant_retrieved = 0
            total_retrieved = 0
            
            # Process each query
            for i, output in enumerate(component_outputs):
                # Get documents from the retriever output
                retrieved_docs = output.get("documents", [])
                
                if not retrieved_docs:
                    continue
                
                # Get ground truth documents and query
                ground_truth_docs = None
                if expected_retrievals and i < len(expected_retrievals):
                    ground_truth_docs = expected_retrievals[i]
                elif expected_outputs and i < len(expected_outputs):
                    ground_truth_docs = [Document(content=expected_outputs[i])]
                
                query = queries[i] if queries and i < len(queries) else None
                
                if not ground_truth_docs or not query:
                    continue
                
                # Classify documents as relevant or irrelevant
                classification = self.relevance_evaluator.classify_documents(
                    retrieved_docs, ground_truth_docs, query
                )
                
                # Calculate precision for this query
                relevant_count = len(classification["relevant"])
                query_total = len(retrieved_docs)
                
                if query_total > 0:
                    precision_scores.append(relevant_count / query_total)
                    total_relevant_retrieved += relevant_count
                    total_retrieved += query_total
            
            # Calculate overall precision
            if precision_scores:
                avg_precision = sum(precision_scores) / len(precision_scores)
                global_precision = total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
                
                # Use average precision as the score
                self.score = avg_precision
                self.success = self.score >= self.threshold
                self.details = {
                    "avg_precision": avg_precision,
                    "global_precision": global_precision,
                    "relevant_retrieved": total_relevant_retrieved,
                    "total_retrieved": total_retrieved,
                    "num_queries": len(precision_scores),
                    "evaluator": self.relevance_evaluator.__class__.__name__
                }
            else:
                self.score = 0.0
                self.success = False
                self.details = {
                    "error": "No valid retrieval results to evaluate",
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
    
    def run(self, component_outputs: List[Dict[str, Any]], expected_outputs: List[str] = None,
            queries: List[str] = None, expected_retrievals: List[List[Document]] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate recall of retrieved documents across multiple queries.
        
        Args:
            component_outputs: List of outputs from the retriever component
            expected_outputs: List of expected outputs (used as fallback for ground truth)
            queries: List of original queries
            expected_retrievals: List of expected retrieval document lists
            
        Returns:
            Dict[str, Any]: Results with score and details
        """
        try:
            if not component_outputs:
                return {
                    "score": 0.0,
                    "success": False,
                    "details": {"error": "No component outputs provided"}
                }
                
            # Track results for each query
            recall_scores = []
            total_relevant_retrieved = 0
            total_relevant = 0
            
            # Process each query
            for i, output in enumerate(component_outputs):
                # Get documents from the retriever output
                retrieved_docs = output.get("documents", [])
                
                # Get ground truth documents and query
                ground_truth_docs = None
                if expected_retrievals and i < len(expected_retrievals):
                    ground_truth_docs = expected_retrievals[i]
                elif expected_outputs and i < len(expected_outputs):
                    ground_truth_docs = [Document(content=expected_outputs[i])]
                
                query = queries[i] if queries and i < len(queries) else None
                
                if not ground_truth_docs or not query or not retrieved_docs:
                    continue
                
                # Classify documents as relevant or irrelevant
                classification = self.relevance_evaluator.classify_documents(
                    retrieved_docs, ground_truth_docs, query
                )
                
                # Calculate recall for this query
                # Assume all ground truth documents are relevant
                relevant_count = len(classification["relevant"])
                total_ground_truth = len(ground_truth_docs)
                
                if total_ground_truth > 0:
                    recall_scores.append(relevant_count / total_ground_truth)
                    total_relevant_retrieved += relevant_count
                    total_relevant += total_ground_truth
            
            # Calculate overall recall
            if recall_scores:
                avg_recall = sum(recall_scores) / len(recall_scores)
                global_recall = total_relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
                
                # Use average recall as the score
                self.score = avg_recall
                self.success = self.score >= self.threshold
                self.details = {
                    "avg_recall": avg_recall,
                    "global_recall": global_recall,
                    "relevant_retrieved": total_relevant_retrieved,
                    "total_relevant": total_relevant,
                    "num_queries": len(recall_scores),
                    "evaluator": self.relevance_evaluator.__class__.__name__
                }
            else:
                self.score = 0.0
                self.success = False
                self.details = {
                    "error": "No valid retrieval results to evaluate",
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
    
    def run(self, component_outputs: List[Dict[str, Any]], expected_outputs: List[str] = None,
            queries: List[str] = None, expected_retrievals: List[List[Document]] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate F1 score of retrieved documents across multiple queries.
        
        Args:
            component_outputs: List of outputs from the retriever component
            expected_outputs: List of expected outputs (used as fallback for ground truth)
            queries: List of original queries
            expected_retrievals: List of expected retrieval document lists
            
        Returns:
            Dict[str, Any]: Results with score and details
        """
        try:
            # Calculate precision and recall
            precision_result = self.precision_metric.run(
                component_outputs=component_outputs,
                expected_outputs=expected_outputs,
                queries=queries,
                expected_retrievals=expected_retrievals,
                **kwargs
            )
            
            recall_result = self.recall_metric.run(
                component_outputs=component_outputs,
                expected_outputs=expected_outputs,
                queries=queries,
                expected_retrievals=expected_retrievals,
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
    
    def run(self, component_outputs: List[Dict[str, Any]], expected_outputs: List[str] = None,
            queries: List[str] = None, expected_retrievals: List[List[Document]] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate MAP of retrieved documents across multiple queries.
        
        Args:
            component_outputs: List of outputs from the retriever component
            expected_outputs: List of expected outputs (used as fallback for ground truth)
            queries: List of original queries
            expected_retrievals: List of expected retrieval document lists
            
        Returns:
            Dict[str, Any]: Results with score and details
        """
        try:
            if not component_outputs:
                return {
                    "score": 0.0,
                    "success": False,
                    "details": {"error": "No component outputs provided"}
                }
                
            # Track results for each query
            average_precisions = []
            total_relevant_retrieved = 0
            total_retrieved = 0
            
            # Process each query
            for i, output in enumerate(component_outputs):
                # Get documents from the retriever output
                retrieved_docs = output.get("documents", [])
                
                if not retrieved_docs:
                    continue
                
                # Get ground truth documents and query
                ground_truth_docs = None
                if expected_retrievals and i < len(expected_retrievals):
                    ground_truth_docs = expected_retrievals[i]
                elif expected_outputs and i < len(expected_outputs):
                    ground_truth_docs = [Document(content=expected_outputs[i])]
                
                query = queries[i] if queries and i < len(queries) else None
                
                if not ground_truth_docs or not query:
                    continue
                
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
                
                # Track results for this query
                average_precisions.append(average_precision)
                total_relevant_retrieved += relevant_count
                total_retrieved += len(retrieved_docs)
            
            # Calculate overall MAP
            if average_precisions:
                avg_map = sum(average_precisions) / len(average_precisions)
                global_map = total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
                
                # Use average MAP as the score
                self.score = avg_map
                self.success = self.score >= self.threshold
                self.details = {
                    "avg_map": avg_map,
                    "global_map": global_map,
                    "relevant_retrieved": total_relevant_retrieved,
                    "total_retrieved": total_retrieved,
                    "num_queries": len(average_precisions),
                    "evaluator": self.relevance_evaluator.__class__.__name__
                }
            else:
                self.score = 0.0
                self.success = False
                self.details = {
                    "error": "No valid retrieval results to evaluate",
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
