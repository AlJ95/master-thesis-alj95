export config_path=./configs/configuration_validation/configuration_phase/rag_dense_retriever.yaml
export evaluation_data_path=./data/processed/config_val/evaluation_data.json
export corpus_path=./data/processed/config_val/corpus
export output_path=./dense.csv
export test_size=30
export experiment_name=config-validation

screen -L -Logfile "${experiment_name}.log" -dmS $experiment_name /srv/master-thesis-alj95/code/ragnroll_project/.venv/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --no-baselines --test-size=$test_size --experiment-name=$experiment_name