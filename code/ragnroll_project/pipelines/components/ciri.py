from haystack import component, GeneratedAnswer
from typing import List

import json
@component
class CiriAnswerExtractor():
    @component.output_types(answer=List[bool])
    def run(self, replies: List[str]):
        try:
            cleaned = replies[-1].replace("```json\n", "").replace("\n```", "")
            answer = json.loads(cleaned)
            if answer["hasError"]:
                return {"answer": [False]}
            else:
                return {"answer": [True]}
        except json.JSONDecodeError as e:
            print(e)
            return {"answers": [False]}
    
@component
class CiriAnswerBuilder():
    @component.output_types(answers=List[GeneratedAnswer])
    def run(self, answers: List[bool], query: str):
        """
        There are 3 answers if the configuration is valid or not.
        If the number of "true" answers is greater than 1 (more than 50% of the answers are true), the configuration is valid.
        """
        results_as_bool = [1 if answer else 0 for answer in answers]
        if sum(results_as_bool) > 1:
            return {"answers": [GeneratedAnswer(data="valid", query=query, documents=[], meta={})]}
        else:
            return {"answers": [GeneratedAnswer(data="invalid", query=query, documents=[], meta={})]}
        
