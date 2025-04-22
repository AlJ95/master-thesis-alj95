from haystack import Pipeline, component, Document
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory.embedding_retriever import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders.openai_text_embedder import OpenAITextEmbedder
from haystack.components.rankers.lost_in_the_middle import LostInTheMiddleRanker
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.joiners import BranchJoiner, DocumentJoiner
from haystack.components.routers import ConditionalRouter
from haystack.components.converters import OutputAdapter
from haystack.utils import Secret
import re
import warnings
from typing import List, Dict, Any
pipeline = Pipeline()

from dotenv import load_dotenv
load_dotenv()

import os
# api_base_url = "http://172.26.92.115"
# api_key = Secret.from_env_var("SE_OPENAI_KEY")


api_base_url = "https://openrouter.ai/api/v1"
api_key = Secret.from_env_var("OPENROUTER_API_KEY")

@component
class IterativeBrancher:
    """
    For every rewritten query, branch into two different subqueries.
    """
    @component.output_types(query=str, subquery=str, stop=bool)
    def run(self, query: str):
        """
        Branch the query into two different subqueries.
        Each content between <start_documentation> and <end_documentation> is a different branch.
        """
        # extract all branches; ignore \n
        branches = [match[0] for match in re.findall(r"(?<=<start_documentation>)(.*?)(?=<end_documentation>($|\n))", query, re.DOTALL)]
        print(f"Found {len(branches)} branches. Query starts with: {query[:10]}")
        # create a new query for each branch
        if len(branches) == 0:
            return {"query": query, "subquery": query, "stop": True}
        else:
            # select and remove the first branch from the query
            subquery = branches[0]
            query = re.sub(r"<start_documentation>(.*?)<end_documentation>", "", query)

            return {"query": query, "subquery": subquery, "stop": False}

routes = [
        {
            "condition": "{{stop}}",
            "output": " ", # This is just a dummy output to make the router work
            "output_name": "stop",
            "output_type": str,
        },
        {
            "condition": "{{not stop}}",
            "output": "{{subquery}}",
            "output_name": "continue",
            "output_type": str,
        },
        {
            "condition": "{{not stop}}",
            "output": "{{query}}",
            "output_name": "iterate",
            "output_type": str,
        },

    ]


# Rewriter
pipeline.add_component("rewriter_prompt", PromptBuilder(template="""The following query consists of a configuraiton. Rewrite this as sort of documentation page for each of this configuration.
Example (docker):
#######################################################
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
pipeline.add_component("output_adapter", OutputAdapter(template="""{{replies[0]}}""", output_type=str))
pipeline.add_component("branch_joiner", BranchJoiner(str))
pipeline.add_component("iterative_brancher", IterativeBrancher())
pipeline.add_component("router", ConditionalRouter(routes))
pipeline.add_component("embedder", OpenAITextEmbedder(model="text-embedding-3-small"))
pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()))
pipeline.add_component("document_joiner", DocumentJoiner())
pipeline.add_component("ranker", LostInTheMiddleRanker())
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
pipeline.add_component("llm", OpenAIGenerator(api_base_url=api_base_url, api_key=api_key))
pipeline.add_component("answer_builder", AnswerBuilder(pattern="The answer is \"(valid|invalid)\""))

pipeline.connect("rewriter_prompt", "rewriter")
pipeline.connect("rewriter", "output_adapter")
pipeline.connect("output_adapter", "branch_joiner")
pipeline.connect("branch_joiner", "iterative_brancher")
pipeline.connect("iterative_brancher.query", "router.query")
pipeline.connect("iterative_brancher.subquery", "router.subquery")
pipeline.connect("iterative_brancher.stop", "router.stop")
pipeline.connect("router.continue", "embedder.text")
pipeline.connect("router.iterate", "branch_joiner")
pipeline.connect("router.stop", "prompt_builder.dummy")
pipeline.connect("embedder", "retriever.query_embedding")
pipeline.connect("retriever", "document_joiner")
pipeline.connect("document_joiner", "ranker")
pipeline.connect("ranker", "prompt_builder")
pipeline.connect("prompt_builder", "llm")
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



"""
'<start_documentation>\nFROM\n\nFROM [--platform=<platform>] <image> [AS <name>]\nOr\n\nFROM [--platform=<platform>] <image>[:<tag>] [AS <name>]\nOr\n\nFROM [--platform=<platform>] <image>[@<digest>] [AS <name>]\nThe FROM instruction initializes a new build stage and sets the base image for subsequent instructions. As such, a valid Dockerfile must start with a FROM instruction. The image can be any valid image.\n<end_documentation>\n<start_documentation>\nRUN\n\nRUN <command>\nThe RUN instruction executes a command inside the container at build time. This command can include package installations, file manipulations, and any other system commands. This allows you to customize the image or install dependencies needed for your application.\n\nCommands in RUN are executed in a new layer on top of the current image and are committed to the image as a new layer. If you have multiple commands to run, you can use logic such as chaining commands with `&&`.\n\nIn this example, `RUN pip install --no-cache-dir requests` installs the `requests` library while ensuring no cache is maintained, leading to a smaller image size.\n<end_documentation>\n<start_documentation>\nCMD\nThe CMD instruction sets the command to be executed when running a container from an image.\n\nYou can specify CMD instructions using shell or exec forms:\n\nCMD ["executable","param1","param2"] (exec form)\nCMD ["param1","param2"] (exec form, as default parameters to ENTRYPOINT)\nCMD command param1 param2 (shell form)\nThere can only be one CMD instruction in a Dockerfile. If you list more than one CMD, only the last one takes effect.\n\nThe purpose of a CMD is to provide defaults for an executing container. These defaults can include an executable, or they can omit the executable, in which case you must specify an ENTRYPOINT instruction as well.\n\nNote: The provided CMD has an incorrect syntax due to an extra pair of brackets around the command; it should be corrected to `CMD ["python", "-m", "http.server", "8000"]`.\n<end_documentation>'], 'meta': [{'model': 'openai/gpt-4o-mini', 'index': 0, 'finish_reason': 'stop', 'usage': {'completion_tokens': 447, 'prompt_tokens': 396, 'total_tokens': 843, 'completion_tokens_details': CompletionTokensDetails(accepted_prediction_tokens=None, audio_tokens=None, reasoning_tokens=0, rejected_prediction_tokens=None), 'prompt_tokens_details': PromptTokensDetails(audio_tokens=None, cached_tokens=0)}}]}
----------------------------------------------------------------------------------------------------
output_adapter
{'output': '<start_documentation>\nFROM\n\nFROM [--platform=<platform>] <image> [AS <name>]\nOr\n\nFROM [--platform=<platform>] <image>[:<tag>] [AS <name>]\nOr\n\nFROM [--platform=<platform>] <image>[@<digest>] [AS <name>]\nThe FROM instruction initializes a new build stage and sets the base image for subsequent instructions. As such, a valid Dockerfile must start with a FROM instruction. The image can be any valid image.\n<end_documentation>\n<start_documentation>\nRUN\n\nRUN <command>\nThe RUN instruction executes a command inside the container at build time. This command can include package installations, file manipulations, and any other system commands. This allows you to customize the image or install dependencies needed for your application.\n\nCommands in RUN are executed in a new layer on top of the current image and are committed to the image as a new layer. If you have multiple commands to run, you can use logic such as chaining commands with `&&`.\n\nIn this example, `RUN pip install --no-cache-dir requests` installs the `requests` library while ensuring no cache is maintained, leading to a smaller image size.\n<end_documentation>\n<start_documentation>\nCMD\nThe CMD instruction sets the command to be executed when running a container from an image.\n\nYou can specify CMD instructions using shell or exec forms:\n\nCMD ["executable","param1","param2"] (exec form)\nCMD ["param1","param2"] (exec form, as default parameters to ENTRYPOINT)\nCMD command param1 param2 (shell form)\nThere can only be one CMD instruction in a Dockerfile. If you list more than one CMD, only the last one takes effect.\n\nThe purpose of a CMD is to provide defaults for an executing container. These defaults can include an executable, or they can omit the executable, in which case you must specify an ENTRYPOINT instruction as well.\n\nNote: The provided CMD has an incorrect syntax due to an extra pair of brackets around the command; it should be corrected to `CMD ["python", "-m", "http.server", "8000"]`.\n<end_documentation>'}
----------------------------------------------------------------------------------------------------
branch_joiner
{'value': '<start_documentation>\nFROM\n\nFROM [--platform=<platform>] <image> [AS <name>]\nOr\n\nFROM [--platform=<platform>] <image>[:<tag>] [AS <name>]\nOr\n\nFROM [--platform=<platform>] <image>[@<digest>] [AS <name>]\nThe FROM instruction initializes a new build stage and sets the base image for subsequent instructions. As such, a valid Dockerfile must start with a FROM instruction. The image can be any valid image.\n<end_documentation>\n<start_documentation>\nRUN\n\nRUN <command>\nThe RUN instruction executes a command inside the container at build time. This command can include package installations, file manipulations, and any other system commands. This allows you to customize the image or install dependencies needed for your application.\n\nCommands in RUN are executed in a new layer on top of the current image and are committed to the image as a new layer. If you have multiple commands to run, you can use logic such as chaining commands with `&&`.\n\nIn this example, `RUN pip install --no-cache-dir requests` installs the `requests` library while ensuring no cache is maintained, leading to a smaller image size.\n<end_documentation>\n<start_documentation>\nCMD\nThe CMD instruction sets the command to be executed when running a container from an image.\n\nYou can specify CMD instructions using shell or exec forms:\n\nCMD ["executable","param1","param2"] (exec form)\nCMD ["param1","param2"] (exec form, as default parameters to ENTRYPOINT)\nCMD command param1 param2 (shell form)\nThere can only be one CMD instruction in a Dockerfile. If you list more than one CMD, only the last one takes effect.\n\nThe purpose of a CMD is to provide defaults for an executing container. These defaults can include an executable, or they can omit the executable, in which case you must specify an ENTRYPOINT instruction as well.\n\nNote: The provided CMD has an incorrect syntax due to an extra pair of brackets around the command; it should be corrected to `CMD ["python", "-m", "http.server", "8000"]`.\n<end_documentation>'}
"""