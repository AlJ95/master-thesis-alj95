import os
import time
import warnings
from typing import Dict, List, Set
from haystack import Document, AsyncPipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    OpenAIDocumentEmbedder,
    HuggingFaceAPIDocumentEmbedder,
    AzureOpenAIDocumentEmbedder
    )
from haystack.utils.device import ComponentDevice
from haystack.utils import Secret
import json
import logging
import yaml
from datetime import datetime

from ragnroll.utils.preprocesser import get_all_documents
from ragnroll.utils.config import get_components_from_config_by_classes
from ragnroll.utils.ingestion_tracker import IngestionTracker, IngestionRecord, tracker
from ragnroll.utils.index_existence_checker import check_index_exists


logger = logging.getLogger(__name__)

BM25Retriever = ["InMemoryBM25Retriever", "QdrantSparseEmbeddingRetriever"]
EmbeddingRetriever = ["InMemoryEmbeddingRetriever", "ChromaTextRetriever", "ChromaEmbeddingRetriever",  "QdrantEmbeddingRetriever", "<YOUR_CUSTOM_RETRIEVER>"]
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

def _extract_chunking_params(pipeline):
    """
    Extract chunking parameters from the configuration.
    """
    chunking_params = {}

    if hasattr(pipeline, 'to_dict'):
        configuration = pipeline.to_dict()
    else:
        configuration = pipeline

    if "chunking" in configuration.get("metadata", {}):
        _params = configuration["metadata"]["chunking"]
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
def index_documents(corpus_dir: str, pipeline):
    """Index documents in the document store with deduplication and tracking."""

    start_time = time.time()

    if hasattr(pipeline, 'to_dict'):
        configuration = pipeline.to_dict()
    else:
        configuration = pipeline

    embedding_retriever=get_components_from_config_by_classes(configuration, ".".join(EmbeddingRetriever))
    bm25_retriever=get_components_from_config_by_classes(configuration, ".".join(BM25Retriever))
    sentence_window_retriever=get_components_from_config_by_classes(configuration, ".".join(SentenceWindowRetriever))
    hybrid_retriever=get_components_from_config_by_classes(configuration, ".".join(HybridRetriever))

    if not embedding_retriever and not bm25_retriever and not sentence_window_retriever and not hybrid_retriever:
        logger.info("No retriever found in configuration. Skipping indexing.")
        return pipeline, 0

    # Extract processing configuration for deterministic ID generation
    chunking_params = _extract_chunking_params(configuration)
    processing_config = {
        "chunking": chunking_params,
        "embedders": get_components_from_config_by_classes(configuration, ".embedders."),
        "retrievers": {
            "embedding": embedding_retriever,
            "bm25": bm25_retriever,
            "sentence_window": sentence_window_retriever,
            "hybrid": hybrid_retriever
        }
    }

    # Generate deterministic index ID
    index_id = tracker.generate_index_id(corpus_dir, processing_config)

    # Check if index already exists
    existing_record = tracker.get_existing_record(index_id)
    if existing_record:
        logger.info(f"Index {index_id} already exists with status: {existing_record.status}")
        if existing_record.status == "completed":
            logger.info("Skipping ingestion - index is already complete")
            # Still need to connect document store to pipeline
            document_store_config = _extract_document_stores(embedding_retriever + bm25_retriever + sentence_window_retriever + hybrid_retriever)
            document_store = get_document_store_from_type(document_store_config)
            if hasattr(pipeline, 'get_component'):
                for component_name, _ in configuration["components"].items():
                    if "document_store" in configuration["components"][component_name]["init_parameters"]:
                        component = pipeline.get_component(component_name)
                        if hasattr(component, 'document_store'):
                            component.document_store = document_store
            return pipeline, 0

    # Check if index exists in document store
    document_store_config = _extract_document_stores(embedding_retriever + bm25_retriever + sentence_window_retriever + hybrid_retriever)
    index_exists_in_store = check_index_exists(document_store_config, index_id)

    if index_exists_in_store and existing_record and existing_record.status == "completed":
        logger.info("Index exists in document store and is marked complete. Skipping ingestion.")
        document_store = get_document_store_from_type(document_store_config)
        if hasattr(pipeline, 'get_component'):
            for component_name, _ in configuration["components"].items():
                if "document_store" in configuration["components"][component_name]["init_parameters"]:
                    component = pipeline.get_component(component_name)
                    if hasattr(component, 'document_store'):
                        component.document_store = document_store
        return pipeline, 0

    # Load and process documents
    documents = get_all_documents(
        corpus_dir=corpus_dir,
        split=chunking_params["split"],
        chunk_size=chunking_params["chunk_size"],
        chunk_overlap=chunking_params["chunk_overlap"],
        chunk_separator=chunking_params["chunk_separator"]
    )

    # Generate document IDs for tracking
    document_ids = []
    for doc in documents:
        doc_content = doc.content or ""  # Handle None content
        doc_id = tracker.generate_document_id(doc_content, processing_config)
        doc.id = doc_id  # Set document ID
        document_ids.append(doc_id)

    # Check for missing documents if index partially exists
    if existing_record and existing_record.status == "partial":
        required_doc_ids = set(document_ids)
        existing_doc_ids = set(existing_record.document_ids)
        missing_doc_ids = required_doc_ids - existing_doc_ids

        if not missing_doc_ids:
            logger.info("All documents already exist. Marking index as completed.")
            tracker.update_record_status(index_id, "completed")
            document_store = get_document_store_from_type(document_store_config)
            if hasattr(pipeline, 'get_component'):
                for component_name, _ in configuration["components"].items():
                    if "document_store" in configuration["components"][component_name]["init_parameters"]:
                        component = pipeline.get_component(component_name)
                        if hasattr(component, 'document_store'):
                            component.document_store = document_store
            return pipeline, 0

        # Filter to only missing documents
        documents = [doc for doc in documents if doc.id in missing_doc_ids]
        logger.info(f"Ingesting {len(documents)} missing documents")

    document_store = get_document_store_from_type(document_store_config)

    # Generate embeddings if needed
    if embedding_retriever or hybrid_retriever:
        # Get text embedder parameters dictionary from configuration
        text_embedders = get_components_from_config_by_classes(configuration, ".embedders.")

        if not text_embedders:
            raise ValueError("No text embedder found in configuration.")

        # Use the first text embedder found
        text_embedder = text_embedders[0] if isinstance(text_embedders, list) else text_embedders

        if isinstance(text_embedder, list):
            text_embedder = text_embedder[0]
            warnings.warn("Multiple text embedders found in configuration. Using the first one to extract the embedding model.")

        try:
            doc_embedder = _get_document_embedder_from_text_embedder(text_embedder)
            documents = doc_embedder.run(documents)["documents"]
        except ValueError as e:
            logger.warning(f"Skipping document embedding due to unsupported embedder: {e}")
            # Continue without embedding if embedder is not supported

    # Write documents to the document store
    try:
        document_store.write_documents(documents)
        ingestion_status = "completed"
        logger.info(f"Successfully indexed {len(documents)} documents")
    except Exception as e:
        logger.warning(f"Failed to write documents: {e}")
        ingestion_status = "failed"
        # Try to continue without failing the entire process

    # Record the ingestion
    processing_config_hash = tracker.generate_index_id(corpus_dir, processing_config)  # Reuse for hash
    record = IngestionRecord(
        corpus_path=corpus_dir,
        processing_config_hash=processing_config_hash,
        index_id=index_id,
        document_ids=document_ids,
        timestamp=datetime.now().isoformat(),
        status=ingestion_status
    )
    tracker.record_ingestion(record)

    logger.info(f"Indexed {len(documents)} documents in the document store")

    # Connect document store to pipeline components
    if hasattr(pipeline, 'get_component'):
        for component_name, _ in configuration["components"].items():
            if "document_store" in configuration["components"][component_name]["init_parameters"]:
                component = pipeline.get_component(component_name)
                if hasattr(component, 'document_store'):
                    component.document_store = document_store

    end_time = time.time()
    return pipeline, end_time - start_time

def _get_document_embedder_from_text_embedder(text_embedder: Dict):
    """
    Get a document embedder from a text embedder.
    """
    try:
        del text_embedder["init_parameters"]["api_key"]
    except:
        pass
    if "OpenAITextEmbedder" in text_embedder["type"]:
        return OpenAIDocumentEmbedder(**text_embedder["init_parameters"])
    elif "HuggingFaceAPITextEmbedder" in text_embedder["type"]:
        return HuggingFaceAPIDocumentEmbedder(**text_embedder["init_parameters"])
    elif "AzureOpenAITextEmbedder" in text_embedder["type"]:
        return AzureOpenAIDocumentEmbedder(**text_embedder["init_parameters"])
    elif "SentenceTransformersTextEmbedder" in text_embedder["type"]:
        if "device" in text_embedder["init_parameters"]:
            text_embedder["init_parameters"]["device"] = ComponentDevice.from_dict(text_embedder["init_parameters"]["device"])
        
        if "token" in text_embedder["init_parameters"] and text_embedder["init_parameters"]["token"]["type"] == "env_var":
            text_embedder["init_parameters"]["token"] = Secret.from_env_var(text_embedder["init_parameters"]["token"]["env_vars"])
        
        doc_embedder = SentenceTransformersDocumentEmbedder(**text_embedder["init_parameters"])
        doc_embedder.warm_up()
        return doc_embedder
    else:
        raise ValueError(f"Unsupported text embedder: {text_embedder['type']}")

def _extract_document_stores(retrievers: list):
    """
    Get a document store from a type string.
    """

    document_stores = []
    for retriever in retrievers:
        if retriever and "init_parameters" in retriever and "document_store" in retriever["init_parameters"]:
            document_stores.append(retriever["init_parameters"]["document_store"])

    if not document_stores:
        # Create a default in-memory document store if none found
        return {
            "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
            "init_parameters": {}
        }

    document_store_types = [
        document_store["type"]
        for document_store in document_stores
    ]

    if len(set(document_store_types)) > 1:
        raise ValueError("All retrievers must use the same document store.")

    return document_stores[0]

# 4. Main function to tie everything together
def index_json_data(json_file_path, configuration):

    # Load JSON data
    json_data = load_json_data(json_file_path)

    # Convert to documents
    documents = convert_to_documents(json_data)

    # Index the documents - note: this expects a corpus_dir, not documents
    # For JSON data, we'd need to create a temporary corpus or modify the approach
    # For now, return None as this function needs refactoring
    logger.warning("index_json_data needs to be updated for the new ingestion system")
    return None

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
    logger.info("Ingestion module updated with tracking and deduplication functionality.")
    logger.info("Use the index_documents function with corpus directories instead of this example.")
