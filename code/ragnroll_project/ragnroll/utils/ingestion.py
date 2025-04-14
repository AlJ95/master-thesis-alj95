import os
import time
import warnings
from typing import Dict
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    OpenAIDocumentEmbedder,
    HuggingFaceAPIDocumentEmbedder,
    AzureOpenAIDocumentEmbedder
    )
import json
import logging
import yaml

from ragnroll.utils.preprocesser import get_all_documents
from ragnroll.utils.config import get_components_from_config_by_classes


logger = logging.getLogger(__name__)

BM25Retriever = ["InMemoryBM25Retriever", "QdrantSparseEmbeddingRetriever"]
EmbeddingRetriever = ["InMemoryEmbeddingRetriever", "ChromaTextRetriever", "ChromaEmbeddingRetriever",  "QdrantEmbeddingRetriever"]
SentenceWindowRetriever = ["SentenceWindowRetriever"]
HybridRetriever = ["QdrantHybridRetriever"]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_SEPARATOR = ["\n\n", "\n", " ", ""]

# 1. Load your JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 2. Convert JSON data to Haystack Documents
def convert_to_documents(json_data):
    documents = []
    
    # Adjust this logic based on your JSON structure
    for item in json_data:
        # Create Document objects from your JSON data
        # Assumes each JSON item has 'content' and optional 'meta' fields
        doc = Document(
            content=item.get("content", ""),
            meta=item.get("meta", {})
        )
        documents.append(doc)
        
    return documents

def _extract_chunking_params(pipeline: Pipeline):
    """
    Extract chunking parameters from the configuration.
    """
    chunking_params = {}

    if "chunking" in pipeline["metadata"]:
        _params = pipeline["metadata"]["chunking"]
        chunking_params["split"] = _params["split"] if "split" in _params else True
        chunking_params["chunk_size"] = _params["chunk_size"] if "chunk_size" in _params else CHUNK_SIZE
        chunking_params["chunk_overlap"] = _params["chunk_overlap"] if "chunk_overlap" in _params else CHUNK_OVERLAP
        chunking_params["chunk_separator"] = _params["chunk_separator"] if "chunk_separator" in _params else CHUNK_SEPARATOR
    else:
        chunking_params = {
            "split": True,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunk_separator": CHUNK_SEPARATOR
        }

    return chunking_params


# 3. Create a document store and index the documents
def index_documents(corpus_dir: str, pipeline: Pipeline):
    """Index documents in the document store."""

    start_time = time.time()

    configuration = pipeline.to_dict()

    embedding_retriever=get_components_from_config_by_classes(configuration, EmbeddingRetriever)
    bm25_retriever=get_components_from_config_by_classes(configuration, BM25Retriever)
    sentence_window_retriever=get_components_from_config_by_classes(configuration, SentenceWindowRetriever)
    hybrid_retriever=get_components_from_config_by_classes(configuration, HybridRetriever)

    
    if not embedding_retriever and not bm25_retriever and not sentence_window_retriever and not hybrid_retriever:
        print("No retriever found in configuration. Skipping indexing.")
        return pipeline, 0

    chunking_params = _extract_chunking_params(configuration)
    documents = get_all_documents(
        corpus_dir=corpus_dir,
        split=chunking_params["split"],
        chunk_size=chunking_params["chunk_size"],
        chunk_overlap=chunking_params["chunk_overlap"],
        chunk_separator=chunking_params["chunk_separator"]
    )

    document_store_config = _extract_document_stores(embedding_retriever + bm25_retriever + sentence_window_retriever + hybrid_retriever)

    document_store = get_document_store_from_type(document_store_config)

    if embedding_retriever or hybrid_retriever:
        
        # Get text embedder parameters dictionary from configuration
        text_embedder = get_components_from_config_by_classes(configuration, [".embedders."])

        if len(text_embedder) > 1:
            warnings.warn("Multiple text embedders found in configuration. Using the first one to extract the embedding model.")
        else:
            text_embedder = text_embedder[0]

        if text_embedder:
            doc_embedder = _get_document_embedder_from_text_embedder(text_embedder)
            documents = doc_embedder.run(documents)["documents"]
        else:
            raise ValueError("No text embedder found in configuration.")

    # Write documents to the document store
    try:
        document_store.write_documents(documents)
    except Exception as e:
        document_store._collection_name = document_store._collection_name + "_" + str(time.time())
        document_store.write_documents(documents)
    
    print(f"Indexed {len(documents)} documents in the document store")
    
    for component_name, _ in configuration["components"].items():
        if "document_store" in configuration["components"][component_name]["init_parameters"]:
            pipeline.get_component(component_name).document_store = document_store
    end_time = time.time()

    return pipeline, end_time - start_time

def _get_document_embedder_from_text_embedder(text_embedder: Dict):
    """
    Get a document embedder from a text embedder.
    """
    del text_embedder["init_parameters"]["api_key"]
    if "OpenAITextEmbedder" in text_embedder["type"]:
        return OpenAIDocumentEmbedder(**text_embedder["init_parameters"])
    elif "HuggingFaceAPITextEmbedder" in text_embedder["type"]:
        return HuggingFaceAPIDocumentEmbedder(**text_embedder["init_parameters"])
    elif "AzureOpenAITextEmbedder" in text_embedder["type"]:
        return AzureOpenAIDocumentEmbedder(**text_embedder["init_parameters"])
    elif "SentenceTransformersTextEmbedder" in text_embedder["type"]:
        doc_embedder = SentenceTransformersDocumentEmbedder(**text_embedder["init_parameters"])
        doc_embedder.warm_up()
        return doc_embedder
    else:
        raise ValueError(f"Unsupported text embedder: {text_embedder['type']}")

def _extract_document_stores(retrievers: list):
    """
    Get a document store from a type string.
    """
    
    document_stores = [
        retriever["init_parameters"]["document_store"]
        for retriever in retrievers
        if retriever
    ]

    document_store_types = [
        document_store["type"]
        for document_store in document_stores
    ]

    if len(set(document_store_types)) > 1:
        raise ValueError("All retrievers must use the same document store.")

    return document_stores[0]

# 4. Main function to tie everything together
def index_json_data(json_file_path, configuration: dict):

    # Load JSON data
    json_data = load_json_data(json_file_path)
    
    # Convert to documents
    documents = convert_to_documents(json_data)
    
    # Index the documents
    document_store = index_documents(documents, configuration)
    
    return document_store

def get_document_store_from_type(document_store_config: dict):
    """
    Get a document store from a type string.
    """
    document_store_type = document_store_config["type"]
    if document_store_type == "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore":
        return InMemoryDocumentStore.from_dict(document_store_config)
    elif document_store_type == "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore":
        from haystack_integrations.document_stores.chroma import ChromaDocumentStore
        return ChromaDocumentStore.from_dict(document_store_config)
    elif document_store_type == "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore":
        from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
        return QdrantDocumentStore.from_dict(document_store_config)
    else:
        raise NotImplementedError(f"Unsupported document store type: {document_store_type}")

# Usage example
if __name__ == "__main__":
    with open("../configs/predefined_bm25.yaml", "r") as f:
        configuration = yaml.safe_load(f)

    document_store = index_json_data("../data/synthetic_rag_corpus.json", configuration)
    
    # Optional: Simple verification query
    results = document_store.filter_documents()
    print(f"Retrieved {len(results)} documents")