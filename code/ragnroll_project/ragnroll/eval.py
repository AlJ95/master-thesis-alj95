from ragnroll.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset
from haystack import Pipeline


def generate_dataset(pipeline: Pipeline, data):
    """
    Generates the evaluation dataset
    """
    labels = data.data["validation"]["label"]
    contents = data.data["validation"]["content"]
    evaluation_dataset = EvaluationDataset(
        test_cases=[
            LLMTestCase(
                input=str(content),
                expected_output=str(label),
                actual_output=generate_answer(pipeline, content),
                # retrieval_context=test_case["retrieval_context"],
            )
            for label, content in zip(labels, contents)
        ],
    )
    return evaluation_dataset


def generate_answer(pipeline: Pipeline, input_text: str) -> str:
    # Set up weaviate test client as environment variables    
    response = pipeline.run(data=dict(code=input_text))
    return response["llm"]["replies"][0]



def evaluate(data, pipeline):
    """
    Evaluates the model on the test cases
    """
    evaluation_dataset = generate_dataset(pipeline, data)
    metric = ExactMatchMetric()
    evaluation_dataset.evaluate([metric])
    return metric
