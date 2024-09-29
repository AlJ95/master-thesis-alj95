import json

from RAGBench.pipelines.indexing import IngestionPipeline


# Function to load configuration from JSON file
def load_config(config_file):
    """
    Loads configuration from a JSON file.

    Args:
        config_file (str): Path to the JSON file containing the configuration.

    Returns:
        dict: The loaded configuration.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

# Base class for pipeline components
class PipelineComponent:
    def execute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

# Retrieval Pipeline
class RetrievalPipeline(PipelineComponent):
    def __init__(self, index, top_k, embedding_model):
        self.index = index
        self.top_k = top_k
        self.embedding_model = embedding_model

    def execute(self, query: str):
        raise NotImplementedError("Subclasses should implement this method.")
    
# Generation Pipeline
class GenerationPipeline(PipelineComponent):
    def __init__(self, api_base_url, api_key, max_tokens, temperature, top_p):
        pass

    def execute(self, retrieved_docs):
        raise NotImplementedError("Subclasses should implement this method.")
    

# Main RAG Pipeline
class RAGPipeline:
    def __init__(self, config):
        self.ingestion_pipeline = None
        self.retrieval_pipeline = None
        self.generation_pipeline = None


# Sample usage
if __name__ == "__main__":
    # # Load configuration from file
    # config = load_config("config.json")
    
    # # Initialize RAG Pipeline with the configuration
    # rag_pipeline = RAGPipeline(config)

    # # Add documents from directory (to be indexed)
    # rag_pipeline.add_documents("path_to_your_documents")

    # # Perform a query and generate a response
    # query = "Explain the concept of RAG."
    # response = rag_pipeline.query(query)
    # print(response)

    print("Hello World")

    indexing_pipeline = IngestionPipeline()
    indexing_pipeline.run()