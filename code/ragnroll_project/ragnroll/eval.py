from ragnroll.metrics import ExactMatchMetric
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset
from haystack import Pipeline

def generate_dataset(pipeline: Pipeline, evaluation_data: dict):
    """
    Generates the evaluation dataset
    """
    evaluation_dataset = EvaluationDataset(
        test_cases=[
            LLMTestCase(
                input=str(test_case["input"]),
                expected_output=str(test_case["expected_output"]),
                actual_output=generate_answer(pipeline, test_case["input"]),
                retrieval_context=test_case["retrieval_context"]
            )
            for test_case in evaluation_data["test_cases"]
        ],
    )
    return evaluation_dataset


def generate_answer(pipeline: Pipeline, input_text: str) -> str:
    # Set up weaviate test client as environment variables    
    response = pipeline.run(data=dict(query=input_text))
    return response["llm"]["replies"][0]



def evaluate(data, pipeline):
    """
    Evaluates the model on the test cases
    """
    evaluation_dataset = generate_dataset(pipeline, data)
    em_metric = ExactMatchMetric()
    ar_metric = AnswerRelevancyMetric(verbose_mode=False)
    evaluation_dataset.evaluate([em_metric, ar_metric])

    return em_metric, ar_metric
