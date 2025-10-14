"""
Index existence checking utilities for different document store types.

This module provides functions to check if an index exists in various
vector database implementations used in RAGnRoll.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def check_index_exists(document_store_config: Dict[str, Any], index_id: str) -> bool:
    """
    Check if an index exists in the document store.

    Args:
        document_store_config: Configuration dictionary for the document store
        index_id: The index ID to check for

    Returns:
        True if index exists, False otherwise
    """
    try:
        document_store_type = document_store_config.get("type", "")

        if "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore" in document_store_type:
            return _check_in_memory_index_exists(document_store_config, index_id)
        elif "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore" in document_store_type:
            return _check_chroma_index_exists(document_store_config, index_id)
        elif "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore" in document_store_type:
            return _check_qdrant_index_exists(document_store_config, index_id)
        else:
            logger.warning(f"Unknown document store type: {document_store_type}. Assuming index doesn't exist.")
            return False

    except Exception as e:
        logger.warning(f"Error checking index existence: {e}")
        return False


def _check_in_memory_index_exists(document_store_config: Dict[str, Any], index_id: str) -> bool:
    """
    Check if index exists in InMemoryDocumentStore.

    For in-memory stores, we can't really check persistence, so we assume
    the index exists if the store is configured (backward compatibility).
    """
    # In-memory stores don't persist indexes, so we rely on our tracking system
    # Return False to force re-ingestion (safe default)
    return False


def _check_chroma_index_exists(document_store_config: Dict[str, Any], index_id: str) -> bool:
    """
    Check if index exists in ChromaDocumentStore.
    """
    try:
        from chromadb import Client
        from chromadb.config import Settings

        # Extract connection parameters
        init_params = document_store_config.get("init_parameters", {})
        collection_name = init_params.get("collection_name", "default_collection")

        # Try to connect to Chroma
        settings = Settings()
        if "host" in init_params:
            settings.chroma_server_host = init_params["host"]
        if "port" in init_params:
            settings.chroma_server_http_port = init_params["port"]

        client = Client(settings)

        # Check if collection exists
        try:
            collection = client.get_collection(collection_name)
            # If collection exists and has documents, consider index existing
            count = collection.count()
            return count is not None and count > 0
        except Exception:
            # Collection doesn't exist
            return False

    except ImportError:
        logger.warning("Chroma not available, cannot check index existence")
        return False
    except Exception as e:
        logger.warning(f"Error checking Chroma index: {e}")
        return False


def _check_qdrant_index_exists(document_store_config: Dict[str, Any], index_id: str) -> bool:
    """
    Check if index exists in QdrantDocumentStore.
    """
    try:
        from qdrant_client import QdrantClient

        # Extract connection parameters
        init_params = document_store_config.get("init_parameters", {})
        collection_name = init_params.get("collection_name", "default_collection")

        # Connect to Qdrant
        if "url" in init_params:
            client = QdrantClient(url=init_params["url"])
        elif "host" in init_params and "port" in init_params:
            client = QdrantClient(host=init_params["host"], port=init_params["port"])
        else:
            # Local instance
            client = QdrantClient(":memory:")

        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name not in collection_names:
            return False

        # Check if collection has vectors
        collection_info = client.get_collection(collection_name)
        return collection_info.vectors_count > 0

    except ImportError:
        logger.warning("Qdrant client not available, cannot check index existence")
        return False
    except Exception as e:
        logger.warning(f"Error checking Qdrant index: {e}")
        return False


def get_document_count(document_store_config: Dict[str, Any], index_id: str) -> Optional[int]:
    """
    Get the number of documents in an existing index.

    Args:
        document_store_config: Configuration dictionary for the document store
        index_id: The index ID

    Returns:
        Number of documents if index exists, None otherwise
    """
    try:
        document_store_type = document_store_config.get("type", "")

        if "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore" in document_store_type:
            return None  # In-memory doesn't persist
        elif "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore" in document_store_type:
            return _get_chroma_document_count(document_store_config, index_id)
        elif "haystack_integrations.document_stores.qdrant.document_store.QdrantDocumentStore" in document_store_type:
            return _get_qdrant_document_count(document_store_config, index_id)
        else:
            return None

    except Exception as e:
        logger.warning(f"Error getting document count: {e}")
        return None


def _get_chroma_document_count(document_store_config: Dict[str, Any], index_id: str) -> Optional[int]:
    """Get document count from Chroma collection."""
    try:
        from chromadb import Client
        from chromadb.config import Settings

        init_params = document_store_config.get("init_parameters", {})
        collection_name = init_params.get("collection_name", "default_collection")

        settings = Settings()
        if "host" in init_params:
            settings.chroma_server_host = init_params["host"]
        if "port" in init_params:
            settings.chroma_server_http_port = init_params["port"]

        client = Client(settings)
        collection = client.get_collection(collection_name)
        return collection.count()

    except Exception:
        return None


def _get_qdrant_document_count(document_store_config: Dict[str, Any], index_id: str) -> Optional[int]:
    """Get document count from Qdrant collection."""
    try:
        from qdrant_client import QdrantClient

        init_params = document_store_config.get("init_parameters", {})
        collection_name = init_params.get("collection_name", "default_collection")

        if "url" in init_params:
            client = QdrantClient(url=init_params["url"])
        elif "host" in init_params and "port" in init_params:
            client = QdrantClient(host=init_params["host"], port=init_params["port"])
        else:
            client = QdrantClient(":memory:")

        collection_info = client.get_collection(collection_name)
        return collection_info.vectors_count

    except Exception:
        return None
