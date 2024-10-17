from autorag.evaluator import Evaluator
from dotenv import load_dotenv
import os 

if __name__ == "__main__":
    load_dotenv()

    path_to_yaml = "/Users/janalbrecht/projects/master-thesis-alj95/code/Notebooks/autorag/simple_config.yaml"
    path_to_qa = "/Users/janalbrecht/projects/master-thesis-alj95/code/Notebooks/autorag/qa.parquet"
    path_to_corpus = "/Users/janalbrecht/projects/master-thesis-alj95/code/Notebooks/autorag/laws.parquet"

    evaluator = Evaluator(qa_data_path=path_to_qa, corpus_data_path=path_to_corpus)

    evaluator.start_trial(path_to_yaml, skip_validation=True)