import json
import os
import weaviate
from llama_index import ServiceContext, LLMPredictor, GPTSimpleVectorIndex, WeaviateIndex, SimpleDirectoryReader
from langchain import OpenAI

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

# Ingestion & Indexing Pipeline
class IngestionIndexingPipeline(PipelineComponent):
    def __init__(self, client, embedding_model):
        self.client = client
        self.embedding_model = embedding_model
        self.index = None

    def execute(self, dir_path: str):
        documents = SimpleDirectoryReader(dir_path).load_data()
        # Create a Weaviate index
        weaviate_index = WeaviateIndex(self.client, "DocumentIndex")
        self.index = GPTSimpleVectorIndex.from_documents(
            documents, 
            vector_store=weaviate_index, 
            embedding_model=self.embedding_model
        )

    def get_index(self):
        return self.index

# Retrieval Pipeline
class RetrievalPipeline(PipelineComponent):
    def __init__(self, index, top_k):
        self.index = index
        self.top_k = top_k

    def execute(self, query: str):
        if not self.index:
            return "No index found."
        return self.index.query(query, top_k=self.top_k)

# Generation Pipeline
class GenerationPipeline(PipelineComponent):
    def __init__(self, api_base_url, api_key, max_tokens, temperature, top_p):
        self.llm_predictor = LLMPredictor(OpenAI(
            temperature=temperature, 
            max_tokens=max_tokens, 
            top_p=top_p,
            api_base=api_base_url, 
            api_key=api_key
        ))

    def execute(self, retrieved_docs):
        if not retrieved_docs:
            return "No relevant documents found."
        return self.llm_predictor.predict(str(retrieved_docs))

# Main RAG Pipeline
class RAGPipeline:
    def __init__(self, config):
        self.client = weaviate.Client(f"http://{config['VectorDB']['weaviate_IP']}:{config['VectorDB']['weaviate_port']}")
        self.ingestion_pipeline = IngestionIndexingPipeline(self.client, config['Indexing_Retrieval']['Embedding_Model'])
        self.retrieval_pipeline = None
        self.generation_pipeline = GenerationPipeline(
            config['Generation']['API_Base_URL'], 
            config['Generation']['API_Key'],
            config['Generation']['Max_Tokens'],
            config['Generation']['Temperature'],
            config['Generation']['Top_P']
        )
        self.top_k = config['Retrieval']['Top_K']

    def add_documents(self, dir_path: str):
        self.ingestion_pipeline.execute(dir_path)
        self.retrieval_pipeline = RetrievalPipeline(self.ingestion_pipeline.get_index(), self.top_k)

    def query(self, query: str):
        retrieved_docs = self.retrieval_pipeline.execute(query)
        response = self.generation_pipeline.execute(retrieved_docs)
        return response


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