# Requirements Engineering

## Data

- User wants to benchmark a RAG system with a **static** dataset.


## Experiment

IS IT POSSIBLE THAT I CAN TRACK THE MISSING NODES / DOCUMENTS IN EACH STEP? 
-> LOOK AT WHERE THE MISTAKE CAN HAPPEN -> IS IT POSSIBLE TO TRACK THAT?

### Isolation
- The system needs to evaluate isolated from physical and environmental states. It needs therefore a docker-compose which 
initialises a VectorDB and Runs the full Process of Indexing & Evaluation of Retrieval+Generation

### Repeatability
- Tests must be repeated and averaged

## User Interface
- There is no WebGUI planned, the user will use a rich commented configuration file with 


## Architecture & Design

### Modular Architecture
- ...