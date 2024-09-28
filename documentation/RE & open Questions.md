# Requirements Engineering

## Data

1. static or dynamic data?
-> **static**
User wants to benchmark a RAG system with a static dataset.


## Experiment

# Isolation
The system needs to evaluate isolated from physical and environmental states. It needs therefore a docker-compose which 
initialises a VectorDB and Runs the full Process of Indexing & Evaluation of Retrieval+Generation

# Repeatability
Tests must be repeated and averaged