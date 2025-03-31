# Langfuse Metrics Integration

This document explains how to use the Langfuse metrics integration in RagnRoll for visualizing and tracking evaluation metrics.

## Overview

RagnRoll now supports reporting evaluation metrics to Langfuse as "scores". This allows you to:

- View metrics in the Langfuse UI alongside your traces
- Track metrics over time
- Compare different runs and pipeline configurations
- Set up alerts and notifications based on metric thresholds

This integration runs in parallel with the existing pandas DataFrame-based reporting, so you can continue to use both approaches.

## Setup

### Prerequisites

1. You need a Langfuse account. Sign up at [Langfuse](https://langfuse.com/) if you don't have one.
2. Set the following environment variables:
   ```bash
   export LANGFUSE_HOST="http://localhost:3000"  # Or your Langfuse instance URL
   export LANGFUSE_SECRET_KEY="your-secret-key"
   export LANGFUSE_PUBLIC_KEY="your-public-key"
   ```

You can find your API keys in the Langfuse dashboard under Settings > API Keys.

### Enabling Metrics Reporting

The metrics reporting is automatically enabled in the `run_evaluations` CLI command. When you run evaluations using this command, your metrics will be reported to Langfuse automatically.

```bash
ragnroll run-evaluations config.yaml eval_data.json corpus_dir output.csv
```

## How It Works

1. When you run an evaluation, the `LangfuseConnector` component is added to your pipeline.
2. Each request processed by your pipeline is traced and gets a unique trace ID in Langfuse.
3. After evaluation completes, all metrics are calculated and attached to the corresponding traces as "scores".
4. You can view these scores in the Langfuse UI by navigating to your traces.

## Customizing Metric Reporting

### Adding Your Own Metrics

To add your own metrics to be reported to Langfuse:

1. Create a custom metric class that extends `BaseMetric`
2. Register it with the `MetricRegistry`
3. Your metric will automatically be included in the Langfuse reporting

Example:

```python
from ragnroll.metrics.base import BaseMetric, MetricRegistry

@MetricRegistry.register_end_to_end
class MyCustomMetric(BaseMetric):
    def run(self, expected_outputs, actual_outputs):
        # Your metric logic here
        score = compute_score(expected_outputs, actual_outputs)
        return {
            "score": score,
            "success": score >= self.threshold,
            "details": {
                # Additional details
            }
        }
```

### Manual Reporting

You can also manually report metrics to Langfuse using the `report_metrics_to_langfuse` function:

```python
from ragnroll.evaluation.tracing import report_metrics_to_langfuse

# Report a single metric
report_metrics_to_langfuse(
    trace_id="trace-id-from-langfuse",
    metrics={
        "my_metric": {
            "score": 0.85,
            "success": True,
            "details": {"additional_info": "value"}
        }
    },
    metric_type="custom",
    component_name="my_component"
)
```

## Viewing Metrics in Langfuse

1. Open the Langfuse UI
2. Navigate to the "Traces" section
3. Find your trace by its ID or filter by trace name
4. Click on a trace to see its details
5. Scroll down to the "Scores" section to see all reported metrics

You can also:
- View scores across multiple traces in the "Scores" dashboard
- Create custom dashboards to visualize metrics over time
- Set up alerts based on score thresholds

## Example Script

See `examples/langfuse_metrics_integration.py` for a complete example of how to use the Langfuse metrics integration.

## Best Practices

1. **Consistent Naming**: Use consistent naming for metrics to make comparisons easier
2. **Grouping**: Group related metrics by using consistent `metric_type` and `component_name` values
3. **Thresholds**: Set appropriate thresholds for `success` flags to make it clear when metrics are satisfactory
4. **Metadata**: Include useful metadata in the `details` field to help with debugging and analysis
5. **Batch Processing**: For large evaluations, consider using `report_batch_metrics_to_langfuse` for better performance 