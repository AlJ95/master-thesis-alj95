from haystack import Pipeline
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory.embedding_retriever import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders.openai_text_embedder import OpenAITextEmbedder
from haystack.components.rankers.lost_in_the_middle import LostInTheMiddleRanker
from haystack.components.builders.answer_builder import AnswerBuilder

pipeline = Pipeline()
pipeline.add_component("embedder", OpenAITextEmbedder(model="text-embedding-ada-002"))
pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()))
pipeline.add_component("ranker", LostInTheMiddleRanker())
pipeline.add_component("prompt_builder", PromptBuilder(template="""

        Given these documents, answer the question.

        Your answer is used within a pipeline and must follow the following format:
        "The answer is "valid"" or "The answer is "invalid"" with quotation marks. Also allowed are "yes" or "no", "true" or "false", "1" or "0".
        After stating your answer, explain your answer in detail.

        Example:
        Question: London is the capital of France.
        Answer: "false", because the capital of France is Paris, not London.

        Documents:

        {% for doc in documents %}\

        {{ doc.content }}

        {% endfor %}        

        Question: {{query}}


        Answer:

        """))
pipeline.add_component("generator", OpenAIGenerator())
pipeline.add_component("answer_builder", AnswerBuilder(pattern="The answer is \"(.*)\"."))

pipeline.connect("embedder", "retriever.query_embedding")
pipeline.connect("retriever", "ranker")
pipeline.connect("ranker", "prompt_builder")
pipeline.connect("prompt_builder", "generator")
pipeline.connect("generator.replies", "answer_builder.replies")

# pipeline.dump(open("pipeline.yaml", "w"))



query = """Is this Dockerfile valid?:

´´´
FROM python:3.10

RUN pip install --no-cache-dir requests

CMD ["python", "-m", "http.server", "8000"]
´´´
             
Answer with in the following format:
The answer is "valid".
The answer is "invalid".
             
with the reason afterwards
"""

# pipeline.run(data=dict(prompt_builder=dict(query=query), embedder=dict(text=query), answer_builder=dict(query=query)))
