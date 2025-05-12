# Ragnroll Project

## Motivation
Large Language Models (LLMs) have limitations like knowledge cutoffs and difficulty with private or real-time data. Retrieval-Augmented Generation (RAG) systems address this by adding a retrieval step, but evaluating these complex, sensitive systems is challenging. Tuning RAG involves assessing components and end-to-end performance, risking overfitting.

Ragnroll provides a systematic framework for benchmarking modular RAG systems (initially focused on classification) to facilitate reproducible experiments and build robust systems.

## Overview
Ragnroll is a CLI framework for evaluating and optimizing RAG systems. It uses Docker Compose for supporting services:

- **MLflow**: Logs parameters and visualizes metrics for comparison and tracking.
- **Langfuse**: Provides detailed tracing for failure analysis and understanding pipeline flow.

This setup enables thorough performance analysis and data-driven optimization.

The framework includes two optional baselines (`llm-standalone` and `naive-rag`) for comparison. The `--no-baselines` flag skips them.

## Usage

### Installation

1.  Clone the repository (if not already done):
    ```bash
    git clone https://github.com/AlJ95/master-thesis-alj95
    cd master-thesis-alj95/code/ragnroll_project
    ```

2.  Configure environment variables:
    - Copy `.env.local` to `.env` and update values.
    - Required: `OPENAI_API_KEY`, `LLM_AS_A_JUDGE_MODEL`, `OPENAI_BASE_URL`, Langfuse variables.
    - **Note:** Use `OPENAI_API_KEY` for the primary LLM API key.

3.  Update security settings (marked `#CHANGEME`) in `docker-compose.yml`.

4.  Start services (and wait few minutes):
    ```bash
    docker-compose up -d
    ```

### Data Preparation

-   **Evaluation Data**: Place `.json` or `.csv` files in `data/processed/`. Required columns depend on the task and its metrics (e.g., input, expected output for classification, or reason for Answer-Relevance).
-   **Corpus**: Store documents for retrieval in `data/processed/corpus/`. Supports file types: `.pdf`, `.txt`, `.docx`, `.md`, `.html`, `.htm`, `.json`, `.csv`. Optionally add `urls.csv` for web scraping.

### Creating RAG Pipelines

Define Haystack pipelines via:

1.  **Python Modules**: In `pipelines/<your-module>.py` (see `pipelines/sample.py`).
2.  **YAML Configuration**: Declaratively in `configs/<your-configuration>.yaml` (see `configs/predefined_4r.yaml`). Enhances reconfigurability.
3.  **Matrix Configuration (YAML)**: Define multiple variations using lists in `configs/<your-matrix-configuration>.yaml` for automated combinatorial testing (see `configs/matrix_examples.yaml`).

Ensure pipelines include necessary components like `answer_builder`, `generators`, and `retrievers`. Register custom components in `pipelines/components/__init__.py`.

### Configuring Document Chunking

Set chunking parameters (`split`, `chunk_size`, `chunk_overlap`, `chunk_separator`) in the `metadata.chunking` section of your YAML configuration. This affects how documents are split during ingestion.

```yaml
metadata:
  chunking:
    split: true
    chunk_size: 500
    chunk_overlap: 150
    chunk_separator: "\n\n"
```

### Running Evaluations

We highly recommend using virtual environments to run the framework.

1. Create a virtual environment:
    ```bash
    python -m venv .venv
    ```

2. Activate the virtual environment:  
    for Ubuntu / Mac
    ```bash
    source .venv/bin/activate
    ```

    for Windows
    ```bash
    .venv/Scripts/activate
    ```

3. Install dependencies:
    ```bash
    python -m pip install -r requirements.txt
    ```

4. Run the framework:

   ```bash
   python -m ragnroll run-evaluations <config_path_or_dir> <eval_data_path> <corpus_dir> [--no-baselines] [--test-size 20]
   ```

   or

   preparing and running the evaluation using the run-eval.sh script:

   ```bash
   ./run-eval.sh
   ```

Example:

```bash
python -m ragnroll run-evaluations ./configs/examples/predefined.yaml ./data/processed/dev_data/synthetic_rag_evaluation.json ./data/dev_data/processed/corpus --test-size 20
```

