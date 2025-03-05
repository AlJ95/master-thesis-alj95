# Framework ToDo

This is a list of tasks that I presented in my master thesis, but are still not implemented. 

## Design

- [ ] Do I address all mentioned drawbacks of RAGs in section drawbacks?

### End-to-End Evaluation

#### Metrics
**Format-Checker**
```python
def output_verifier(output, classes)
    """classes = ["valid", "invalid"]"""
    return output in classes
```

#### Baselines
Baselines must be used per default, but only for each unique configuration change that affects the baselines as well (e. g. chunking in naive)

### Component Evaluation

#### Rewrite
flowaicom/Flow-Judge-v0.1

#### Retrieve

#### Rerank

#### Read


### Tracing

https://langfuse.com/self-hosting/local
https://docs.haystack.deepset.ai/docs/langfuseconnector
