export config_path=./configs/examples/predefined.yaml
export evaluation_data_path=./data/processed/dev_data/synthetic_rag_evaluation.json
export test_size=20
export experiment_name=test-run
export corpus_path=./data/processed/dev_data/corpus_filtered_cleaned

## Comment out the following lines if you already have a virtual environment

# create virtual environment
python -m venv .venv        # Comment out if you already have a virtual environment

# activate virtual environment
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt  # Comment out if you already have the dependencies installed

## Double Check if Docker containers are running
# $ docker ps

# run evaluation
python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --test-size=$test_size --experiment-name=$experiment_name --no-baselines --positive-label=valid --negative-label=invalid