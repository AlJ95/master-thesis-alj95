# Ragnroll Project

## Motivation
RAG evaluation is hard! With so many different architectures, optimizing end-to-end metrics while dealing with component bottlenecks and production concerns like latency is challenging. Worse yet, endless reconfigurations often just lead to overfitting on test datasets rather than real improvements.

Ragnroll tackles this problem head-on by providing a flexible evaluation framework for modularized RAG systems focused on classification tasks. Our goal is simple: help you build better RAG systems that perform well in the real world, not just on benchmark datasets.

## Overview
Ragnroll is a command-line based framework that provides comprehensive tools for evaluating and optimizing RAG systems. The framework requires Docker Compose to run its supporting services:

- **MLflow**: Used for visualization of evaluation metrics and results, making it easier to compare different RAG configurations and track improvements over time.
- **Langfuse**: Provides detailed tracing capabilities for monitoring the execution flow of your RAG pipelines. We integrated Langfuse because, at the time of development, MLflow's tracing functionality was not compatible with Haystack pipelines.

This architecture allows you to thoroughly analyze your RAG system's performance, identify bottlenecks, and make data-driven optimization decisions.

The baselines `llm-standalone` and `naive-rag` serve as essential benchmarks for evaluating RAG systems. They provide a straightforward comparison, allowing users to quickly assess the performance of their configurations against these foundational models. By analyzing results from these baselines, developers can identify key areas for improvement and ensure their RAG systems are optimized for real-world applications. For evaluations that focus solely on custom configurations, the `--no-baselines` option can be used to exclude these benchmarks from the results.

## Usage
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AlJ95/ragnroll
   cd ragnroll
   ```

2. Configure environment variables:
   - Copy `.env.local` to `.env`
   - Update the values in `.env` with your own settings
   - Required environment variables:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `LLM_AS_A_JUDGE_MODEL`: The model to use for LLM-as-a-Judge evaluation (default: "gpt-4o-mini")
     - `OPENAI_BASE_URL`: Base URL for OpenAI API (e.g., "https://openrouter.ai/api/v1" for OpenRouter)
     - Langfuse configuration variables (see .env.local for details)

3. Update security settings in `docker-compose.yml`:
   - Find all instances marked with `#CHANGEME` and replace them with secure values

4. Start the supporting services:
   ```bash
   docker-compose up -d
   ```

## Data Preparation

### Evaluation Data
Place your evaluation data in `data/processed/<your-eval-data>.json` or `.csv`. The data must follow this format as shown in `synthesize_config.json`:
- Input text
- Expected output
- Reason (optional)
- Expected retrieval (optional)

### Corpus
Store your document corpus in `data/processed/corpus/`:
- You can include an `urls.csv` file for scraping documents before indexing
- Supported file types include PDF, TXT, DOCX, MD, HTML, JSON, and CSV formats, as defined in `preprocesser.py`, which are used for converting various document types into a standardized format for processing.

## Creating RAG Pipelines

You can create Haystack RAG pipelines in three different ways:

1. **Pipeline Module**:
   - Create a Python module in `pipelines/<your-module>.py`
   - See `pipelines/sample.py` for an example

2. **YAML Configuration**:
   - Define your pipeline in `configs/<your-configuration>.yaml`
   - See `configs/predefined_4r.yaml` for an example

3. **Matrix Configuration**:
   - Create a matrix configuration in `configs/<your-matrix-configuration>.yaml`
   - This generates multiple configurations based on all possible combinations
   - See `configs/matrix_examples.yaml` for an example

- Ensure you have the following components defined in your configuration:
  - **Answer Builder**: Atleast one `answer_builder` for the finals answer in the pipeline
  - **Generators**: Must have the type `*.generators.*`
  - **Retrievers**: Must have the type `*.retrievers.*`
  
If you use custom generators or retrievers, you must inherit from the `Generator` or `Retriever` classes respectively.


## Running Evaluations

Start an evaluation with the following command:

```bash
python -m ragnroll run-evaluations ./configs/matrix_example.yaml ./data/processed/synthetic_rag_evaluation.json ./data/processed/corpus ./output.csv --no-baselines
```
