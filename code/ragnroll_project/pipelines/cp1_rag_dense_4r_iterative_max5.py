from haystack import Pipeline, component
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory.embedding_retriever import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.rankers.lost_in_the_middle import LostInTheMiddleRanker
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.converters import OutputAdapter
from haystack.utils import Secret
import re
from typing import List
import warnings

pipeline = Pipeline(
    metadata={
        "hypothesis": "Iterative embedding and retrieval for each configuration item can improve the performance of the pipeline.",
        "chunking": {
            "split": True,
            "chunk_size": 500,
            "chunk_overlap": 150,
            "chunk_separator": "\n\n"
        },
        "reasoning": False,
        "reconfiguration_phase": 1
})

from dotenv import load_dotenv
load_dotenv()

import os
# api_base_url = "http://172.26.92.115"
# api_key = Secret.from_env_var("SE_OPENAI_KEY")


api_base_url = "https://openrouter.ai/api/v1"
api_key = Secret.from_env_var("OPENROUTER_API_KEY")

@component
class IterativeEmbedding:
    """
    For every rewritten query, branch into two different subqueries.
    """
    @component.output_types(queries=List[str])
    def run(self, query: str):
        """
        Branch the query into two different subqueries.
        Each content between <start_documentation> and <end_documentation> is a different branch.
        """
        # extract all branches; ignore \n
        documentations = [match[0] for match in re.findall(r"(?<=<start_documentation>)(.*?)(?=<end_documentation>($|\n))", query, re.DOTALL)]

        if len(documentations) > 5:
            documentations = documentations[:5]
            warnings.warn(f"Found {len(documentations)} branches. Only the first 5 are used.")
        elif len(documentations) < 5:
            documentations.extend([""] * (5 - len(documentations)))
        else:
            print(f"Found {len(documentations)} branches.")
        return {"queries": documentations}


# Rewriter
pipeline.add_component("rewriter_prompt", PromptBuilder(template="""The following query consists of a configuration. Rewrite this as sort of documentation page for each of this configuration. Create exactly 5 documentations. Focus on the configuration items that can lead easily to errors / misconfigurations.
Example (docker):
Query:
```
FROM python:3.10

CMD ["python", "-m", "http.server", "8000"]
```

Documentation:
<start_documentation>
FROM

FROM [--platform=<platform>] <image> [AS <name>]
Or


FROM [--platform=<platform>] <image>[:<tag>] [AS <name>]
Or


FROM [--platform=<platform>] <image>[@<digest>] [AS <name>]
The FROM instruction initializes a new build stage and sets the base image for subsequent instructions. As such, a valid Dockerfile must start with a FROM instruction. The image can be any valid image.
<end_documentation>
<start_documentation>
CMD
The CMD instruction sets the command to be executed when running a container from an image.

You can specify CMD instructions using shell or exec forms:

CMD ["executable","param1","param2"] (exec form)
CMD ["param1","param2"] (exec form, as default parameters to ENTRYPOINT)
CMD command param1 param2 (shell form)
There can only be one CMD instruction in a Dockerfile. If you list more than one CMD, only the last one takes effect.

The purpose of a CMD is to provide defaults for an executing container. These defaults can include an executable, or they can omit the executable, in which case you must specify an ENTRYPOINT instruction as well.
<end_documentation>                                                 
#######################################################
                                                 

Only return the documentation, no other text.
Query:
```
{{query}}
```
                                                        
Documentation:
""",
        required_variables=["query"]))
pipeline.add_component("rewriter", OpenAIGenerator(api_base_url=api_base_url, api_key=api_key))
pipeline.connect("rewriter_prompt", "rewriter")

pipeline.add_component("output_adapter", OutputAdapter(template="""{{replies[0]}}""", output_type=str))
pipeline.connect("rewriter", "output_adapter")

pipeline.add_component("iterative_embedding", IterativeEmbedding())
pipeline.connect("output_adapter", "iterative_embedding")

pipeline.add_component("document_joiner", DocumentJoiner())
document_store=InMemoryDocumentStore()
for i in range(5):
    template = "{{{{queries[{i}]}}}}".format(i=i) # double-double-curly-braces to avoid jinja-syntax
    pipeline.add_component(f"output_adapter_{i}", OutputAdapter(template=template, output_type=str))
    pipeline.connect(f"iterative_embedding", f"output_adapter_{i}")

    pipeline.add_component(f"text_embedder_{i}", OpenAITextEmbedder(model="text-embedding-3-small"))
    pipeline.connect(f"output_adapter_{i}", f"text_embedder_{i}")

    pipeline.add_component(f"retriever_{i}", InMemoryEmbeddingRetriever(document_store=document_store, top_k=2))    
    pipeline.connect(f"text_embedder_{i}.embedding", f"retriever_{i}.query_embedding")

    pipeline.connect(f"retriever_{i}", f"document_joiner")

pipeline.add_component("ranker", LostInTheMiddleRanker())
pipeline.connect("document_joiner", "ranker")
pipeline.add_component("prompt_builder", PromptBuilder(template="""

        Given these documents, answer the question.{{dummy}}
                                                       
        Is the following configuration valid?

        Your answer is used within a pipeline and must follow the following format:
        "The answer is "valid"" or "The answer is "invalid"" with quotation marks. Also allowed are "yes" or "no", "true" or "false", "1" or "0".
        After stating your answer, explain your answer in detail.

        Example:
        Configuration: FROM python:3.10 AS base AS test
        Answer: "false", because multiple AS aliases are not allowed in a single FROM instruction

        Here are helpful documents from the documentation:

        {% for doc in documents %}\

        {{ doc.content }}

        {% endfor %}        

                                                       
        Configuration: {{query}}


        Answer: """,
        required_variables=["query"]))
pipeline.connect("ranker", "prompt_builder")

pipeline.add_component("llm", OpenAIGenerator(api_base_url=api_base_url, api_key=api_key))
pipeline.connect("prompt_builder", "llm")
pipeline.add_component("answer_builder", AnswerBuilder(pattern="The answer is \"(valid|invalid)\""))
pipeline.connect("llm.replies", "answer_builder.replies")


# pipeline.dump(open("pipeline.yaml", "w"))



query = """
´´´
FROM python:3.10

RUN pip install --no-cache-dir requests

CMD ["python", "-m", ["http.server", "8000"]]
´´´
"""

# result = pipeline.run(data=dict(query=query), include_outputs_from=pipeline.to_dict()["components"].keys())

# for key, value in result.items():
#     print(key)
#     print(value)
#     print("-"*100)

