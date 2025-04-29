export config_path=./configs/configuration_validation/reconfiguration_phase_3/ciri_gpt_4o_mini_versionized.yaml
export evaluation_data_path=./data/processed/config_val_ciri/evaluation_data_ciri.json
export corpus_path=./data/processed/config_val_ciri/corpus_filtered_cleaned
export output_path=./ciri_gpt_4o_mini_versionized.csv
export test_size=30
export experiment_name=ciri-model-comparison

screen -L -Logfile "${experiment_name}.log" -dmS $experiment_name /srv/master-thesis-alj95/code/ragnroll_project/.venv2/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --no-baselines --test-size=$test_size --experiment-name=$experiment_name

export config_path=./configs/configuration_validation/reconfiguration_phase_3/ciri_deepseek.yaml
export output_path=./ciri_deepseek.csv

screen -L -Logfile "${experiment_name}.log" -dmS $experiment_name /srv/master-thesis-alj95/code/ragnroll_project/.venv2/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --no-baselines --test-size=$test_size --experiment-name=$experiment_name

export config_path=./configs/configuration_validation/reconfiguration_phase_3/ciri_gemma_25_flash.yaml
export output_path=./ciri_gemma_25_flash.csv

screen -L -Logfile "${experiment_name}.log" -dmS $experiment_name /srv/master-thesis-alj95/code/ragnroll_project/.venv2/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --no-baselines --test-size=$test_size --experiment-name=$experiment_name

export config_path=./configs/configuration_validation/reconfiguration_phase_3/ciri_gpt_4o_most_recent.yaml
export output_path=./ciri_gpt_4o_most_recent.csv

screen -L -Logfile "${experiment_name}.log" -dmS $experiment_name /srv/master-thesis-alj95/code/ragnroll_project/.venv2/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --no-baselines --test-size=$test_size --experiment-name=$experiment_name

export config_path=./configs/configuration_validation/reconfiguration_phase_3/ciri_gpt_41.yaml
export output_path=./ciri_gpt_41.csv

screen -L -Logfile "${experiment_name}.log" -dmS $experiment_name /srv/master-thesis-alj95/code/ragnroll_project/.venv2/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --no-baselines --test-size=$test_size --experiment-name=$experiment_name

export config_path=./configs/configuration_validation/reconfiguration_phase_3/ciri_qwen_32.yaml
export output_path=./ciri_qwen_32.csv

screen -L -Logfile "${experiment_name}.log" -dmS $experiment_name /srv/master-thesis-alj95/code/ragnroll_project/.venv2/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --no-baselines --test-size=$test_size --experiment-name=$experiment_name

export config_path=./configs/configuration_validation/reconfiguration_phase_3/ciri_qwen_235.yaml
export output_path=./ciri_qwen_235.csv

screen -L -Logfile "${experiment_name}.log" -dmS $experiment_name /srv/master-thesis-alj95/code/ragnroll_project/.venv2/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --no-baselines --test-size=$test_size --experiment-name=$experiment_name
