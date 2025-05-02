export output_path=./dense123.csv
export test_size=30
export experiment_name=config-validation-ciri

#export corpus_path=./data/processed/config_val/corpus
#export evaluation_data_path=./data/processed/config_val/evaluation_data.json
#export config_path=./configs/configuration_validation/configuration_phase/rag_hybrid_retriever.yaml
#/srv/master-thesis-alj95/code/ragnroll_project/.venv/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --test-size=$test_size --experiment-name=$experiment_name --no-baselines


export corpus_path=./data/processed/config_val/corpus_filtered_cleaned
export evaluation_data_path=./data/processed/config_val_ciri/evaluation_data_ciri.json
export config_path=./configs/configuration_validation/reconfiguration_phase_2/rag_hybrid_retriever_ciri.yaml
/srv/master-thesis-alj95/code/ragnroll_project/.venv/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --test-size=$test_size --experiment-name=$experiment_name --no-baselines --positive-label=true --negative-label=false


export config_path=./configs/configuration_validation/reconfiguration_phase_2/rag_sparse_retriever_ciri.yaml
/srv/master-thesis-alj95/code/ragnroll_project/.venv/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --test-size=$test_size --experiment-name=$experiment_name --no-baselines --positive-label=true --negative-label=false 


export config_path=./configs/configuration_validation/reconfiguration_phase_2/rag_dense_retriever_ciri.yaml
/srv/master-thesis-alj95/code/ragnroll_project/.venv/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --test-size=$test_size --experiment-name=$experiment_name --no-baselines --positive-label=true --negative-label=false 