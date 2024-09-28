import os
import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Property, DataType, VectorDistances, Configure

WEAVIATE_CLIENT = None

def get_weaviate_client(distance_metric, host=None):


    if host is None:
        host = os.getenv("WEAVIATE_HOST", "weaviate")
    http_port = os.getenv("WEAVIATE_HTTP_PORT", 8080)
    grpc_port = os.getenv("WEAVIATE_GRPC_PORT", 50051)

    print(f"Connecting to Weaviate at {host}:{http_port} (HTTP) and {host}:{grpc_port} (gRPC)")

    global WEAVIATE_CLIENT
    
    WEAVIATE_CLIENT = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host=host,
            http_port=http_port,
            http_secure=False,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=False,
        )
    )

    if not WEAVIATE_CLIENT.is_ready():
        WEAVIATE_CLIENT.connect()

    create_collection(WEAVIATE_CLIENT, "Documents", distance_metric)

    return WEAVIATE_CLIENT


def create_collection(weaviate_client, collection_name, distance_metric):
    if distance_metric == VectorDistances.COSINE.value:
        distance = VectorDistances.COSINE
    elif distance_metric == VectorDistances.DOT.value:
        distance = VectorDistances.DOT
    elif distance_metric == VectorDistances.L2_SQUARED.value:
        distance = VectorDistances.L2_SQUARED
    elif distance_metric == VectorDistances.HAMMING.value:
        distance = VectorDistances.HAMMING
    elif distance_metric == VectorDistances.MANHATTAN.value:
        distance = VectorDistances.MANHATTAN
    else:
        raise ValueError("Unknown distance metric")
    
    try:
        weaviate_client.collections.create(
            collection_name,
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=distance
            ),
            properties=[
                Property(name="url", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="uuid", data_type=DataType.UUID)
            ]
        )
    except weaviate.exceptions.UnexpectedStatusCodeError:
        print(f"Collection {collection_name} already exists.")
        return


def get_collection(weaviate_client, collection_name):
    try:
        return weaviate_client.collections.get(collection_name)
    except weaviate.exceptions.UnexpectedStatusCodeError:
        print(f"Collection {collection_name} does not exist.")
        return None