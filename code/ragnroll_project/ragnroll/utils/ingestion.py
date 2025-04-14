import os
import time
from typing import Dict
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import JSONConverter
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder,
    OpenAIDocumentEmbedder, OpenAITextEmbedder,
    HuggingFaceAPIDocumentEmbedder, HuggingFaceAPITextEmbedder,
    AzureOpenAIDocumentEmbedder, AzureOpenAITextEmbedder
    )
import json
import logging
import yaml

from ragnroll.utils.preprocesser import get_all_documents
from ragnroll.utils.config import get_components_from_config_by_class


logger = logging.getLogger(__name__)

BM25Retriever = "InMemoryBM25Retriever"
EmbeddingRetriever = "InMemoryEmbeddingRetriever"
SentenceWindowRetriever = "SentenceWindowRetriever"

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

    embedding_retriever=get_components_from_config_by_class(configuration, EmbeddingRetriever)
    bm25_retriever=get_components_from_config_by_class(configuration, BM25Retriever)
    sentence_window_retriever=get_components_from_config_by_class(configuration, SentenceWindowRetriever)
    
    if not embedding_retriever and not bm25_retriever and not sentence_window_retriever:
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

    document_store_types = [
        retriever["init_parameters"]["document_store"]["type"]
        for retriever in [bm25_retriever, embedding_retriever, sentence_window_retriever]
        if retriever
    ]

    if len(set(document_store_types)) > 1:
        raise ValueError("All retrievers must use the same document store.")

    document_store_type = document_store_types[0]

    document_store = get_document_store_from_type(document_store_type, configuration["components"]["embedding_retriever"]["init_parameters"]["document_store"])

    if embedding_retriever:
        
        # Get text embedder parameters dictionary from configuration
        text_embedder = next(iter(get_components_from_config_by_class(configuration, ".embedders.").values()))

        if text_embedder:
            doc_embedder = _get_document_embedder_from_text_embedder(text_embedder)
            documents = doc_embedder.run(documents)["documents"]
        else:
            raise ValueError("No text embedder found in configuration.")

    # Write documents to the document store
    document_store.write_documents(documents)
    
    print(f"Indexed {len(documents)} documents in the document store")
    
    for component_name, _ in configuration["components"].items():
        if any(retriever in configuration["components"][component_name]["type"] 
               for retriever in [EmbeddingRetriever, BM25Retriever, SentenceWindowRetriever]
               ):
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

# 4. Main function to tie everything together
def index_json_data(json_file_path, configuration: dict):

    # Load JSON data
    json_data = load_json_data(json_file_path)
    
    # Convert to documents
    documents = convert_to_documents(json_data)
    
    # Index the documents
    document_store = index_documents(documents, configuration)
    
    return document_store

def get_document_store_from_type(document_store_type: str, config: dict):
    """
    Get a document store from a type string.
    """
    if document_store_type == "'haystack.document_stores.in_memory.document_store.InMemoryDocumentStore'":
        return InMemoryDocumentStore.from_dict(config)
    elif document_store_type == "haystack_integrations.document_stores.chroma.ChromaDocumentStore":
        from haystack_integrations.document_stores.chroma import ChromaDocumentStore
        return ChromaDocumentStore.from_dict(config)
    elif document_store_type == "haystack_integrations.document_stores.pinecone.PineconeDocumentStore":
        from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
        return PineconeDocumentStore.from_dict(config)
    elif document_store_type == "haystack_integrations.document_stores.qdrant.QdrantDocumentStore":
        from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
        return QdrantDocumentStore.from_dict(config)
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