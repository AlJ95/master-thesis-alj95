from haystack import Pipeline, Document
from datasets import load_dataset
from pathlib import Path
from dotenv import load_dotenv

prev = ""
while not load_dotenv(prev + ".env"):
    prev += "../"

config_path = Path(__file__).parent.parent.parent / "configs"
config_name = "se4ai_mietbot"

print("Load from Template ...")
pipeline = Pipeline.load(open(config_path / f"{config_name}.yaml", "r"))

print("Draw pipeline ...")
pipeline.draw(config_path / f"{config_name}.png")

# Load data
print("Load data ...")
dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

print("Embedding documents ...")
docs_with_embeddings = pipeline.get_component("docs_embedder").run(docs)
pipeline.remove_component("docs_embedder")
pipeline.get_component("embedding_retriever").document_store.write_documents(docs_with_embeddings["documents"])


question = "What does Rhodes Statue look like?"
print("Ask a question ...")
response = pipeline.run({"text_embedder": {"text": question}, "bm25_retriever": {"query": question}, "prompt_builder": {"query": question}})

print(response["llm"]["replies"][0])

# print("Dump to YAML ...")
# yaml = pipeline.dump(open(config_path / "predefined.yaml", "w+"))
