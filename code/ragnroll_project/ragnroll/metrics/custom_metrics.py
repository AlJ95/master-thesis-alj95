from ragnroll.metrics.base import BaseMetric, MetricRegistry
from typing import List, Dict, Any
@MetricRegistry.register_end_to_end
class MyCustomEndToEndMetric(BaseMetric):
    def run(self, expected_outputs: List[str], actual_outputs: List[str], **kwargs) -> Dict[str, Any]:
        # Ihre Metrik-Logik hier
        score = 1
        return {
            "score": score,
            "success": score >= self.threshold,
            "details": {
                # Zusätzliche Details
            }
        }