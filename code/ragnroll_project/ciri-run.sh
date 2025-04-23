export config_path=./pipelines/ciri.py
export evaluation_data_path=./data/processed/config_val_ciri/evaluation_data.json
export corpus_path=./data/processed/config_val_ciri/corpus
export output_path=./ciri.csv
export test_size=30
export experiment_name=ciri-baseline

screen -L -Logfile "${experiment_name}.log" -dmS $experiment_name /srv/master-thesis-alj95/code/ragnroll_project/.venv2/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --no-baselines --test-size=$test_size --experiment-name=$experiment_name
