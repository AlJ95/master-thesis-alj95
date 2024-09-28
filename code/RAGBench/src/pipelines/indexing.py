import json
from pathlib import Path
from typing import Generator
from itertools import chain
from weaviate.util import generate_uuid5
import warnings as warning

from ..config import Configuration
from ..vector_db.weaviate import get_weaviate_client

from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter


class IndexingPipeline:
    DATA_DIR = Path() / "data"
    LAWS_FILE = DATA_DIR / "laws.json"
    EMBEDDING_MODEL_DEFAULT = "intfloat/multilingual-e5-large-instruct"
        

    def __init__(self, config: Configuration):
        self.config = config

        if self.config.embedding_model:
            self.embedding_model = self.config.embedding_model
        else:
            self.embedding_model = self.EMBEDDING_MODEL_DEFAULT

        self.distance_metric = "cosine"


    def load_data(self, file_path: Path):
        try:
            with open(file_path) as f:
                data = json.load(f)
                if data is None:
                    raise ValueError(f"File {file_path} is empty.")
                if "data" not in data:
                    raise ValueError(f"File {file_path} does not contain 'data' key.")
                data = data["data"]
                if data is None:
                    raise ValueError(f"File {file_path} contains null data.")
                return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON in {file_path}: {e}")

    def documents_from_json(self, json_data: list, doc_type: str):
        for doc in json_data:
            metadata={"url": doc["url"], "title": doc["title"], "doc_type": doc_type}

            doc_uuid = generate_uuid5({"url": doc["url"], "title": doc["title"], "text": doc["text"]})
            metadata["uuid"] = doc_uuid

            # Only judgements have a date
            if doc_type == self.JUDGEMENT_TYPE:
                metadata["date"] = doc["date"]
            else:
                metadata["date"] = None

            yield Document(
                text=doc["text"],
                metadata=metadata,
                metadata_seperator="::",
                metadata_template="{key}=>{value}",
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            )
    
    def get_transformations(self):
        transformations = []

        if self.config.splitting_strategy == "sentence":
            transformations.append(SentenceSplitter(chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap))
        
        elif self.config.splitting_strategy == "paragraph":
            transformations.append(TokenTextSplitter(chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap, separator="\n\n"))
            transformations.append(SentenceSplitter(chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap))

        else:
            raise ValueError(f"Unknown splitting strategy: {self.config.splitting_strategy}. Using default sentence splitting")

        transformations.append(self.get_embedding_model(embedding_model_name=self.config.embedding_model))

        return transformations

    def index_documents(self, documents: Generator[Document, None, None], index_name: str):
        
        print("Connecting to Weaviate")
        weaviate_client = get_weaviate_client(distance_metric=self.distance_metric)

        print("Creating vector store")
        vector_store = WeaviateVectorStore(
            weaviate_client=weaviate_client, index_name=index_name
        )

        transformations = self.get_transformations()

        print("Creating ingestion pipeline")
        pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=vector_store,
        )

        print("Checking for existing documents in VectorDB")

        collection = weaviate_client.collections.get("Documents")
        new_documents = []

        for doc in documents:
            doc_uuid = doc.metadata["uuid"]
            if collection.data.exists(doc_uuid):
                print(f"Documents with UUID {doc_uuid} already existing.")
            else:
                new_documents.append(doc)

        if new_documents:
            print("Running ingestion pipeline for new documents")
            pipeline.run(
                documents=new_documents,
                show_progress=True,
            )
        else:
            print("No new documents.")

    def run(self):
        laws = self.load_data(self.LAWS_FILE)
        documents = self.documents_from_json(laws)

        self.index_documents(documents=documents, index_name="Documents")

        return True # Return something meaningful here

if __name__ == "__main__":

    config = Configuration(
        chunk_size=512,
        chunk_overlap=128,
        embedding_model="intfloat/multilingual-e5-large-instruct",
    )

    indexing_pipeline = IndexingPipeline(config)
    indexing_pipeline.run()

    print(indexing_pipeline.DATA_DIR)
    print(indexing_pipeline.LAWS_FILE.exists())

