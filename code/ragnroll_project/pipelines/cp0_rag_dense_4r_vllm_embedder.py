from haystack import Pipeline
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory.embedding_retriever import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.rankers.lost_in_the_middle import LostInTheMiddleRanker
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.converters import OutputAdapter
from haystack.utils import Secret
pipeline = Pipeline()

from dotenv import load_dotenv
load_dotenv()

import os
# api_base_url = "http://172.26.92.115"
# api_key = Secret.from_env_var("SE_OPENAI_KEY")


api_base_url = "https://openrouter.ai/api/v1"
api_key = Secret.from_env_var("OPENROUTER_API_KEY")


# Rewriter
pipeline.add_component("rewriter_prompt", PromptBuilder(template="""The following query consists of a configuration. Rewrite this as sort of documentation page for each of this configuration.
Example (docker):
#######################################################
Query:
```
FROM python:3.10

CMD ["python", "-m", "http.server", "8000"]
```

Documentation:
FROM

FROM [--platform=<platform>] <image> [AS <name>]
Or


FROM [--platform=<platform>] <image>[:<tag>] [AS <name>]
Or


FROM [--platform=<platform>] <image>[@<digest>] [AS <name>]
The FROM instruction initializes a new build stage and sets the base image for subsequent instructions. As such, a valid Dockerfile must start with a FROM instruction. The image can be any valid image.
                                                 
--------------------------------
CMD
The CMD instruction sets the command to be executed when running a container from an image.

You can specify CMD instructions using shell or exec forms:

CMD ["executable","param1","param2"] (exec form)
CMD ["param1","param2"] (exec form, as default parameters to ENTRYPOINT)
CMD command param1 param2 (shell form)
There can only be one CMD instruction in a Dockerfile. If you list more than one CMD, only the last one takes effect.

The purpose of a CMD is to provide defaults for an executing container. These defaults can include an executable, or they can omit the executable, in which case you must specify an ENTRYPOINT instruction as well.
```                                                 
#######################################################
                                                 

Only return the documentation, no other text.
Query: {{query}}
""",
        required_variables=["query"]))
pipeline.add_component("rewriter", OpenAIGenerator(api_base_url=api_base_url, api_key=api_key))
pipeline.add_component("output_adapter", OutputAdapter(template="{{ replies[0] }}", output_type=str))
pipeline.add_component("embedder", OpenAITextEmbedder(model="infly/inf-retriever-v1-1.5b", api_base_url="https://w7mxnj4radnct8-8000.proxy.runpod.net/v1"))
pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()))
pipeline.add_component("ranker", LostInTheMiddleRanker())
pipeline.add_component("prompt_builder", PromptBuilder(template="""

        Given these documents, answer the question.
                                                       
        Is the following configuration valid?

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

        Configuration: {{query}}


        Answer: """,
        required_variables=["query"]))
pipeline.add_component("llm", OpenAIGenerator(api_base_url=api_base_url, api_key=api_key))
pipeline.add_component("answer_builder", AnswerBuilder(pattern="The answer is \"(valid|invalid)\""))

pipeline.connect("rewriter_prompt", "rewriter")
pipeline.connect("rewriter", "output_adapter")
pipeline.connect("output_adapter", "embedder.text")
pipeline.connect("embedder", "retriever.query_embedding")
pipeline.connect("retriever", "ranker")
pipeline.connect("ranker", "prompt_builder")
pipeline.connect("prompt_builder", "llm")
pipeline.connect("llm.replies", "answer_builder.replies")

# pipeline.dump(open("pipeline.yaml", "w"))



query = """
´´´
FROM python:3.10

RUN pip install --no-cache-dir requests

CMD ["python", "-m", "http.server", "8000"]
´´´
"""

# result = pipeline.run(data=dict(query=query), include_outputs_from=["rewriter", "output_adapter", "embedder", "retriever", "ranker", "prompt_builder", "llm", "answer_builder"])

# for key, value in result.items():
#     print(key)
#     print(value)
#     print("-"*100)
