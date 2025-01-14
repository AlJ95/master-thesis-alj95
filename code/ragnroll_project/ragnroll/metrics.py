from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class ExactMatchMetric(BaseMetric):
    def __init__(self, threshold: float = 1):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        self.score = int(test_case.actual_output == test_case.expected_output)
        self.success = self.score >= self.threshold
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success

    @property
    def __name__(self):
        return "ExactMatchMetric"