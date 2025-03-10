"""Top-level package for RAGnRoll."""
# rptodo/__init__.py

__app_name__ = "RAGnRoll"
__version__ = "0.1.0"

(
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    DB_READ_ERROR,
    DB_WRITE_ERROR,
    JSON_ERROR,
    ID_ERROR,
) = range(7)

ERRORS = {
    DIR_ERROR: "config directory error",
    FILE_ERROR: "config file error",
    DB_READ_ERROR: "database read error",
    DB_WRITE_ERROR: "database write error",
    ID_ERROR: "to-do id error",
}

# Import main modules for easier access
from ragnroll.metrics import (
    BaseMetric, 
    MetricRegistry, 
    HaystackContextRelevanceMetric,
    LLMContextPrecisionMetric
)

from ragnroll.evaluation import evaluate, print_scores, Evaluator