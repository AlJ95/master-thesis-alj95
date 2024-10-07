import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAGBench.pipelines.indexing import IndexingPipeline, Configuration


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

    print("Starting indexing")

    config = Configuration(
        chunk_size=512,
        chunk_overlap=128,
        embedding_model="intfloat/multilingual-e5-large-instruct",
        splitting_strategy="sentence",
    )

    indexing_pipeline = IndexingPipeline(config)
    indexing_pipeline.run()

    print(indexing_pipeline.DATA_DIR)
    print(indexing_pipeline.LAWS_FILE.exists())