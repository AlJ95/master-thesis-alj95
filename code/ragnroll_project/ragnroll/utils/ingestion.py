from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import JSONConverter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
import json
import logging
import yaml

from ragnroll.utils.preprocesser import get_all_documents


logger = logging.getLogger(__name__)

BM25Retriever = "haystack.components.retrievers.in_memory.bm25_retriever.InMemoryBM25Retriever"
EmbeddingRetriever = "haystack.components.retrievers.in_memory.embedding_retriever.InMemoryEmbeddingRetriever"
SentenceWindowRetriever = "haystack.components.retrievers.sentence_window_retriever.SentenceWindowRetriever"

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

# 3. Create a document store and index the documents
def index_documents(corpus_dir: str, configuration: dict):
    # Initialize document store
    document_store = InMemoryDocumentStore()

    if (
        "ingestion" not in configuration or 
        "chunk_size" not in configuration["ingestion"] or 
        "chunk_overlap" not in configuration["ingestion"]
        ):
        chunk_size = 1000
        chunk_overlap = 200
    else:
        chunk_size = configuration["ingestion"]["chunk_size"]
        chunk_overlap = configuration["ingestion"]["chunk_overlap"]

    documents = get_all_documents(
        corpus_dir=corpus_dir,
        clean=True,
        split=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if embedding_retriever:=get_component_from_config_by_class(configuration, EmbeddingRetriever):

        if "init_parameters" not in embedding_retriever or "model" not in embedding_retriever["init_parameters"]:
            raise ValueError("model not found in init_parameters of embedding_retriever")

        doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_retriever["init_parameters"]["model"])

        doc_embedder.warm_up()
        documents = doc_embedder.run(documents)["documents"]

    # Write documents to the document store
    document_store.write_documents(documents)
    
    print(f"Indexed {len(documents)} documents in the document store")
    return document_store

def get_component_from_config_by_class(configuration: dict, component_name: str):
    """
    Get a component from the configuration by class name e. g. "haystack.components.retrievers.in_memory.bm25_retriever.InMemoryBM25Retriever"
    """
    for _, values in configuration["components"].items():
        if values["type"] == component_name:
            return values
    return None

# 4. Main function to tie everything together
def index_json_data(json_file_path, configuration: dict):

    # Load JSON data
    json_data = load_json_data(json_file_path)
    
    # Convert to documents
    documents = convert_to_documents(json_data)
    
    # Index the documents
    document_store = index_documents(documents, configuration)
    
    return document_store


# Usage example
if __name__ == "__main__":
    with open("../configs/predefined_bm25.yaml", "r") as f:
        configuration = yaml.safe_load(f)

    document_store = index_json_data("../data/synthetic_rag_corpus.json", configuration)
    
    # Optional: Simple verification query
    results = document_store.filter_documents()
    print(f"Retrieved {len(results)} documents")