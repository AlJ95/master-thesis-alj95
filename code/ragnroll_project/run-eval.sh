export config_path=./pipelines/cp0_rag_dense_4r_vllm_embedder.py
export evaluation_data_path=./data/processed/config_val/evaluation_data.json
export test_size=30
export experiment_name=config-validation

export corpus_path=./data/processed/config_val/corpus_filtered_cleaned
export output_path=./dense1234f13.csv
/srv/master-thesis-alj95/code/ragnroll_project/.venv2/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --test-size=$test_size --experiment-name=$experiment_name --no-baselines --positive-label=valid --negative-label=invalid




export experiment_name=config-validation-ciri
export output_path=./dense1234f133.csv
export evaluation_data_path=./data/processed/config_val_ciri/evaluation_data_ciri.json
export config_path=./pipelines/cp0_rag_dense_4r_vllm_embedder_ciri.py
/srv/master-thesis-alj95/code/ragnroll_project/.venv2/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --test-size=$test_size --experiment-name=$experiment_name --no-baselines --positive-label=valid --negative-label=invalid


