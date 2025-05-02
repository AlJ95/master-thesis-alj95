export config_path=./configs/configuration_validation/reconfiguration_phase_1/rag_dense_4r_iterative_max5.yaml
export evaluation_data_path=./data/processed/config_val/evaluation_data.json
export corpus_path=./data/processed/config_val/corpus_filtered_cleaned
export output_path=./dense123.csv
export test_size=30
export experiment_name=config-validation-small

screen -L -Logfile "${experiment_name}.log" -dmS $experiment_name /srv/master-thesis-alj95/code/ragnroll_project/.venv2/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --test-size=$test_size --experiment-name=$experiment_name
