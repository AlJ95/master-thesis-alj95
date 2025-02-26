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
Develop new method: Almost fully derived from retrieval -> Retrieval metrics but using rewriting as agnostic component

#### Retrieve

#### Rerank

#### Read