python -m ragnroll run-evaluations ./configs/configuration_validation/configuration_phase/rag_dense_retriever.yaml ./data/processed/config_val/evaluation_data.json ./data/processed/config_val/corpus ./rag_dense_retriever.csv --no-baselines --experiment-name=config-validation &
pid1=$!
wait $pid1
python -m ragnroll run-evaluations ./configs/configuration_validation/configuration_phase/rag_sparse_retriever.yaml ./data/processed/config_val/evaluation_data.json ./data/processed/config_val/corpus ./rag_sparse_retriever.csv --no-baselines --experiment-name=config-validation
