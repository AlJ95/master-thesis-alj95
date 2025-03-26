from typing import Dict, Any, List, Tuple, Optional
from ragnroll.metrics.base import BaseMetric, MetricRegistry
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, auc

SENT_WARNING_ONCE = False

class ClassificationBaseMetric(BaseMetric):
    """Base class for binary classification metrics that work with batches of predictions."""
    
    def __init__(self, threshold: float = 0.5, positive_label: str = "1", 
                 negative_label: str = "0", case_sensitive: bool = False):
        """
        Initialize the classification metric.
        
        Args:
            threshold: Threshold for determining success
            positive_label: String representing the positive class
            negative_label: String representing the negative class
            case_sensitive: Whether to treat labels as case-sensitive
        """
        super().__init__(threshold=threshold)
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.case_sensitive = case_sensitive
    
    def _normalize_label(self, label: str) -> str:
        """Normalize the label based on case sensitivity."""
        return label if self.case_sensitive else label.lower()
    
    def _convert_to_binary_class(self, label: str) -> int:
        """Convert string label to binary (0/1) format."""
        normalized = self._normalize_label(label)
        if normalized == self._normalize_label(self.positive_label):
            return 1
        elif normalized == self._normalize_label(self.negative_label):
            return 0
        else:
            try:
                # Try to convert to int/float and check if it matches positive/negative
                numeric_val = float(normalized)
                if numeric_val > 0:
                    return 1
                else:
                    return 0
            except ValueError:
                # Return 1 if the label contains the positive label
                # Otherwise return 0
                return 1 if self._normalize_label(self.positive_label) in normalized else 0
    
    def _process_predictions(self, expected_outputs: List[str], actual_outputs: List[str]) -> Tuple[List[int], List[int]]:
        """
        Process a batch of predictions and return binary class labels.
        
        Args:
            expected_outputs: List of ground truth labels
            actual_outputs: List of predicted labels
            
        Returns:
            Tuple containing lists of true and predicted binary classes
        """
        y_true = []
        y_pred = []

        for expected, actual in zip(expected_outputs, actual_outputs):
            expected_class = self._convert_to_binary_class(expected)
            actual_class = self._convert_to_binary_class(actual)
            y_true.append(expected_class)
            y_pred.append(actual_class)
        
        return y_true, y_pred
    
    def get_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
        """
        Calculate confusion matrix values from true and predicted labels.
        
        Args:
            y_true: List of true binary labels
            y_pred: List of predicted binary labels
            
        Returns:
            Dict with TP, FP, TN, FN counts
        """
        tp = sum((a == 1 and b == 1) for a, b in zip(y_true, y_pred))
        tn = sum((a == 0 and b == 0) for a, b in zip(y_true, y_pred))
        fp = sum((a == 0 and b == 1) for a, b in zip(y_true, y_pred))
        fn = sum((a == 1 and b == 0) for a, b in zip(y_true, y_pred))
        
        return {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }


@MetricRegistry.register_end_to_end
class AccuracyMetric(ClassificationBaseMetric):
    """Accuracy metric for binary classification tasks."""
    
    def run(self, expected_outputs: List[str], actual_outputs: List[str]) -> Dict[str, Any]:
        """
        Calculate accuracy across all test samples.
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        Args:
            expected_outputs: List of ground truth labels
            actual_outputs: List of predicted labels
            
        Returns:
            Dict with accuracy score and success flag
        """
        try:
            y_true, y_pred = self._process_predictions(expected_outputs, actual_outputs)
            self.score = accuracy_score(y_true, y_pred)
            self.success = self.score >= self.threshold
            
            # Calculate confusion matrix for details
            cm = self.get_confusion_matrix(y_true, y_pred)
            exact_matches = sum(e == a for e, a in zip(expected_outputs, actual_outputs))
            
            return {
                "score": self.score,
                "success": self.success,
                "details": {
                    "confusion_matrix": cm,
                    "num_examples": len(expected_outputs),
                    "exact_matches": exact_matches
                }
            }
        except Exception as e:
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }


@MetricRegistry.register_end_to_end
class PrecisionMetric(ClassificationBaseMetric):
    """Precision metric for binary classification tasks."""
    
    def run(self, expected_outputs: List[str], actual_outputs: List[str]) -> Dict[str, Any]:
        """
        Calculate precision on predictions.
        
        Precision = TP / (TP + FP)
        
        Args:
            expected_outputs: List of ground truth labels
            actual_outputs: List of predicted labels
            
        Returns:
            Dict with precision score and success flag
        """
        try:
            y_true, y_pred = self._process_predictions(expected_outputs, actual_outputs)
            self.score = precision_score(y_true, y_pred, zero_division=0)
            self.success = self.score >= self.threshold
            
            # Calculate confusion matrix components for details
            cm = self.get_confusion_matrix(y_true, y_pred)
            
            return {
                "score": self.score,
                "success": self.success,
                "details": {
                    "true_positives": cm["TP"],
                    "false_positives": cm["FP"],
                    "num_examples": len(expected_outputs)
                }
            }
        except Exception as e:
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }


@MetricRegistry.register_end_to_end
class RecallMetric(ClassificationBaseMetric):
    """Recall metric for binary classification tasks."""
    
    def run(self, expected_outputs: List[str], actual_outputs: List[str]) -> Dict[str, Any]:
        """
        Calculate recall on predictions.
        
        Recall = TP / (TP + FN)
        
        Args:
            expected_outputs: List of ground truth labels
            actual_outputs: List of predicted labels
            
        Returns:
            Dict with recall score and success flag
        """
        try:
            y_true, y_pred = self._process_predictions(expected_outputs, actual_outputs)
            self.score = recall_score(y_true, y_pred, zero_division=0)
            self.success = self.score >= self.threshold
            
            # Calculate confusion matrix components for details
            cm = self.get_confusion_matrix(y_true, y_pred)
            
            return {
                "score": self.score,
                "success": self.success,
                "details": {
                    "true_positives": cm["TP"],
                    "false_negatives": cm["FN"],
                    "num_examples": len(expected_outputs)
                }
            }
        except Exception as e:
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }


@MetricRegistry.register_end_to_end
class F1ScoreMetric(ClassificationBaseMetric):
    """F1 score metric for binary classification tasks."""
    
    def run(self, expected_outputs: List[str], actual_outputs: List[str]) -> Dict[str, Any]:
        """
        Calculate F1 score on predictions.
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            expected_outputs: List of ground truth labels
            actual_outputs: List of predicted labels
            
        Returns:
            Dict with F1 score and success flag
        """
        try:
            y_true, y_pred = self._process_predictions(expected_outputs, actual_outputs)
            self.score = f1_score(y_true, y_pred, zero_division=0)
            self.success = self.score >= self.threshold
            
            # Calculate precision and recall for details
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            return {
                "score": self.score,
                "success": self.success,
                "details": {
                    "precision": precision,
                    "recall": recall,
                    "num_examples": len(expected_outputs)
                }
            }
        except Exception as e:
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }


@MetricRegistry.register_end_to_end
class MatthewsCorrCoefMetric(ClassificationBaseMetric):
    """Matthews Correlation Coefficient metric for binary classification tasks."""
    
    def run(self, expected_outputs: List[str], actual_outputs: List[str]) -> Dict[str, Any]:
        """
        Calculate Matthews Correlation Coefficient on predictions.
        
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        
        Args:
            expected_outputs: List of ground truth labels
            actual_outputs: List of predicted labels
            
        Returns:
            Dict with MCC score and success flag
        """
        try:
            y_true, y_pred = self._process_predictions(expected_outputs, actual_outputs)
            
            # Handle edge case: if all predictions are the same class
            if len(set(y_pred)) <= 1:
                if set(y_pred) == set(y_true):  # All correct
                    self.score = 1.0
                else:  # All incorrect
                    self.score = 0.0
            else:
                self.score = matthews_corrcoef(y_true, y_pred)
            
            self.success = self.score >= self.threshold
            
            # Calculate confusion matrix for details
            cm = self.get_confusion_matrix(y_true, y_pred)
            
            return {
                "score": self.score,
                "success": self.success,
                "details": {
                    "confusion_matrix": cm,
                    "num_examples": len(expected_outputs)
                }
            }
        except Exception as e:
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }


@MetricRegistry.register_end_to_end
class FalsePositiveRateMetric(ClassificationBaseMetric):
    """False Positive Rate metric for binary classification tasks."""
    
    def run(self, expected_outputs: List[str], actual_outputs: List[str]) -> Dict[str, Any]:
        """
        Calculate False Positive Rate on predictions.
        
        FPR = FP / (FP + TN)
        
        Args:
            expected_outputs: List of ground truth labels
            actual_outputs: List of predicted labels
            
        Returns:
            Dict with FPR score and success flag
        """
        try:
            y_true, y_pred = self._process_predictions(expected_outputs, actual_outputs)
            
            # Calculate confusion matrix for FPR
            cm = self.get_confusion_matrix(y_true, y_pred)
            fp = cm["FP"]
            tn = cm["TN"]
            
            if fp + tn > 0:
                self.score = fp / (fp + tn)
            else:
                self.score = 0.0
            
            # For FPR, lower is better, so we invert for the success check
            self.success = (1 - self.score) >= self.threshold
            
            return {
                "score": self.score,
                "success": self.success,
                "details": {
                    "false_positives": fp,
                    "true_negatives": tn,
                    "num_examples": len(expected_outputs)
                }
            }
        except Exception as e:
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }


@MetricRegistry.register_end_to_end
class FalseNegativeRateMetric(ClassificationBaseMetric):
    """False Negative Rate metric for binary classification tasks."""
    
    def run(self, expected_outputs: List[str], actual_outputs: List[str]) -> Dict[str, Any]:
        """
        Calculate False Negative Rate on predictions.
        
        FNR = FN / (FN + TP)
        
        Args:
            expected_outputs: List of ground truth labels
            actual_outputs: List of predicted labels
            
        Returns:
            Dict with FNR score and success flag
        """
        try:
            y_true, y_pred = self._process_predictions(expected_outputs, actual_outputs)
            
            # Calculate confusion matrix for FNR
            cm = self.get_confusion_matrix(y_true, y_pred)
            fn = cm["FN"]
            tp = cm["TP"]
            
            if fn + tp > 0:
                self.score = fn / (fn + tp)
            else:
                self.score = 0.0
            
            # For FNR, lower is better, so we invert for the success check
            self.success = (1 - self.score) >= self.threshold
            
            return {
                "score": self.score,
                "success": self.success,
                "details": {
                    "false_negatives": fn,
                    "true_positives": tp, 
                    "num_examples": len(expected_outputs)
                }
            }
        except Exception as e:
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }


@MetricRegistry.register_end_to_end
class ROCAUCMetric(ClassificationBaseMetric):
    """Area Under ROC Curve metric for binary classification tasks."""
    
    def run(self, expected_outputs: List[str], actual_outputs: List[str], 
            probabilities: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Calculate ROC AUC on predictions.
        
        Args:
            expected_outputs: List of ground truth labels
            actual_outputs: List of predicted labels
            probabilities: Optional list of probabilities for the positive class
            
        Returns:
            Dict with AUC score and success flag
        """
        try:
            y_true, y_pred = self._process_predictions(expected_outputs, actual_outputs)
            
            # Use probabilities if provided, otherwise use binary predictions
            if probabilities is not None:
                y_probs = probabilities
            else:
                y_probs = [float(pred) for pred in y_pred]
                # Add a warning to the results when using binary predictions
                if not SENT_WARNING_ONCE:
                    print("Warning: ROC AUC calculated with binary predictions instead of probabilities. Results may be less informative.")
                    SENT_WARNING_ONCE = True
            
            # Calculate AUC if we have enough data points and at least two classes
            if len(y_true) >= 2 and len(set(y_true)) > 1:
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_probs)
                self.score = auc(fpr, tpr)
                self.success = self.score >= self.threshold
                
                return {
                    "score": self.score,
                    "success": self.success,
                    "details": {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "num_examples": len(expected_outputs)
                    }
                }
            else:
                # Not enough data or classes for ROC AUC
                self.score = 0.0
                self.success = False
                return {
                    "score": self.score,
                    "success": False,
                    "details": {
                        "error": "Not enough data points or classes for ROC AUC calculation",
                        "num_examples": len(expected_outputs),
                        "unique_classes": len(set(y_true))
                    }
                }
        except Exception as e:
            self.error = e
            self.success = False
            return {
                "score": 0.0,
                "success": False,
                "details": {"error": str(e)}
            }

