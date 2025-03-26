import yaml

def extract_run_params(yaml_file: str) -> dict:
    with open(yaml_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    # recursively convert all keys to strings
    params = {}
    flatten_dict = _extract_nested_items(yaml_config)
    for key, value in flatten_dict:
        if "api_key" in key:
            continue

        params[key] = value

    params["Nr.of.Components"] = len(yaml_config["components"])
    params["Nr.of.Connections"] = len(yaml_config["connections"])
    params["Nr.of.Retrievers"] = len([c for c, v in yaml_config["components"].items() if "retriever" in v["type"]])
    params["Nr.of.Generators"] = len([c for c, v in yaml_config["components"].items() if "generator" in v["type"]])

    if _has_dense_retriever(yaml_config):
        params["Retriever"] = "DenseRetriever"
        
        if _has_sparse_retriever(yaml_config):
            params["Retriever"] = "HybridRetriever"

    elif _has_sparse_retriever(yaml_config):
        params["Retriever"] = "SparseRetriever"
    else:
        params["Retriever"] = "NoRetriever"

    return params

def _extract_nested_items(yaml_config: dict, key_prefix: str = "") -> dict:
    for key, value in yaml_config.items():
        if isinstance(value, dict):
            yield from _extract_nested_items(value, key_prefix + key + ".")
        else:
            yield key_prefix + key, value

def _has_dense_retriever(yaml_config: dict) -> bool:
    return any("embedding_retriever" in v["type"] for v in yaml_config["components"].values())

def _has_sparse_retriever(yaml_config: dict) -> bool:
    return any("bm25_retriever" in v["type"] for v in yaml_config["components"].values())

def get_components_from_config_by_class(configuration: dict, component_class: str):
    """
    Get a component from the configuration by class name e. g. "haystack.components.retrievers.in_memory.bm25_retriever.InMemoryBM25Retriever"
    """
    components = {}
    for name, values in configuration["components"].items():
        if component_class in values["type"]:
            components[name] = values
    return components


if __name__ == "__main__":
    print(extract_run_params("configs/baselines/llm_config.yaml"))
