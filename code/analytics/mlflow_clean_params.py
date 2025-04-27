import mlflow

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://91.99.14.73:8080")
    mlflow.set_experiment(experiment_name="config-validation")

    runs = mlflow.search_runs()

    new_run_params = {}
    existing_types = []
    for run_id in runs["run_id"]:
        new_run_params[run_id] = {}
        print("-" * 100)
        run = runs[runs["run_id"] == run_id].iloc[0]
        for key, value in run.items():
            if key.endswith(".type") and value and value != "LOCAL":
                new_value = "includes." + value.split(".")[-1]
                count_key = "Nr.of." + value.split(".")[-1]
                print(value, "=", True)
                new_run_params[run_id][new_value] = True

                if count_key not in new_run_params[run_id]:
                    new_run_params[run_id][count_key] = 1
                else:
                    new_run_params[run_id][count_key] += 1

                if new_value not in existing_types:
                    existing_types.append(new_value)

    for run_id, params in new_run_params.items():
        for existing_type in existing_types:
            if existing_type not in params:
                params[existing_type] = False
                params[f"Nr.of.{existing_type.split('.')[-1]}"] = 0

    for run_id, params in new_run_params.items():
        for key, value in params.items():
            print(run_id, key, value)
        print("-" * 100)

    # replace nan and na with empty string
    runs = runs.fillna("")

    for run_id, params in new_run_params.items():
        run = runs[runs["run_id"] == run_id]

        run_metrics = run.loc[:, run.columns.str.startswith("metrics.")]

        # replace empty strings with 0
        run_metrics = run_metrics.replace("", 0)
        run_metrics = run_metrics.astype(float, errors="ignore")
        print(run_metrics.iloc[0].to_dict())

        run_params = run.loc[:, ~run.columns.str.startswith("metrics.")]

        mlflow.set_experiment(experiment_name="config-validation-cleaned-2")
        mlflow.start_run(run_name=run_id)
        mlflow.log_params(run_params.iloc[0].to_dict())
        mlflow.log_params(params)
        mlflow.log_metrics(run_metrics.iloc[0].to_dict())
        mlflow.end_run()

    print("Done")
