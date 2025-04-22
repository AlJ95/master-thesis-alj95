prompt = """{{query}}"""

# {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": message}
# model=self.model, temperature=0.2, max_tokens=512, stop=["\n```"]


## Shot Selection
# args.system = "alluxio" | "docker" ...
# args.shot_selection = "random"
# args.validconfig_shot_num = 1
# args.misconfig_shot_num = 3
# args.input_file_content = <file_content>

from haystack import Pipeline, component
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.validators import JsonSchemaValidator
from haystack.components.joiners import ListJoiner
from haystack.components.converters import OutputAdapter
from haystack.utils import Secret
pipeline = Pipeline()

from dotenv import load_dotenv
from typing import List
load_dotenv()

import os
# os.environ["OPENAI_API_KEY"] = os.environ["SE_OPENAI_KEY"]
import json
@component
class AnswerExtractor():
    @component.output_types(answer=List[bool])
    def run(self, messages: List[str]):
        try:
            cleaned = messages[-1].replace("```json\n", "").replace("\n```", "")
            answer = json.loads(cleaned)
            if answer["hasError"]:
                return {"answer": [False]}
            else:
                return {"answer": [True]}
        except json.JSONDecodeError as e:
            print(e)
            return {"answer": [False]}
    
@component
class AnswerBuilder():
    @component.output_types(answer=str)
    def run(self, answers: List[bool]):
        """
        There are 3 answers if the configuration is valid or not.
        If the number of "true" answers is greater than 1 (more than 50% of the answers are true), the configuration is valid.
        """
        results_as_bool = [1 if answer else 0 for answer in answers]
        if sum(results_as_bool) > 1:
            return {"answer": "valid"}
        else:
            return {"answer": "invalid"}
        

pipeline.add_component("prompt_builder1", PromptBuilder(template=prompt, required_variables=["query"]))
pipeline.add_component("llm1", OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-3.5-turbo-0125"))
pipeline.add_component("answer_extractor1", AnswerExtractor())

pipeline.connect("prompt_builder1", "llm1")
pipeline.connect("llm1.replies", "answer_extractor1.messages")

pipeline.add_component("prompt_builder2", PromptBuilder(template=prompt, required_variables=["query"]))
pipeline.add_component("llm2", OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-3.5-turbo-0125"))
pipeline.add_component("answer_extractor2", AnswerExtractor())

pipeline.connect("prompt_builder2", "llm2")
pipeline.connect("llm2.replies", "answer_extractor2.messages")

pipeline.add_component("prompt_builder3", PromptBuilder(template=prompt, required_variables=["query"]))
pipeline.add_component("llm3", OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-3.5-turbo-0125"))
pipeline.add_component("answer_extractor3", AnswerExtractor())

pipeline.connect("prompt_builder3", "llm3")
pipeline.connect("llm3.replies", "answer_extractor3.messages")

pipeline.add_component("list_joiner", ListJoiner(List[bool]))

pipeline.connect("answer_extractor1.answer", "list_joiner")
pipeline.connect("answer_extractor2.answer", "list_joiner")
pipeline.connect("answer_extractor3.answer", "list_joiner")

pipeline.add_component("answer_builder", AnswerBuilder())
pipeline.connect("list_joiner.values", "answer_builder.answers")

# import json
# with open("data/processed/config_val_ciri/evaluation_data_ciri.json", "r") as f:
#     eval_data = json.load(f)

# for i in [0, 1, len(eval_data["test_cases"])-1]:
#     print("="*100)
#     print(f"Test case {i}")
#     print("-"*100)
#     test_case = eval_data["test_cases"][i]
#     query = test_case["input"]
#     result = pipeline.run(data=dict(query=query), include_outputs_from=pipeline.to_dict()["components"])

#     for key, value in result.items():
#         print(key)
#         print(value)
#         print("-"*100)
