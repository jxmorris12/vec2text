import argparse
import logging
import os

import torch

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


datasets_cache_dir = "/home/jxm3/research/retrieval/distractor_exp"


class NoisySentenceBERT(models.SentenceBERT):
    noise_level: float
    def __init__(self, *args, noise_level: float, **kwargs):
        super().__init__(*args, *kwargs)
        self.noise_level = noise_level
    
    def _inject_noise(self, encodings: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(encodings.shape, device=encodings.device)
        return encodings + noise * self.noise_level
    
    def encode_queries(self, *args, **kwargs) -> torch.Tensor:
        encodings = super().encode_queries(*args, **kwargs)
        return self._inject_noise(encodings)
    
    def encode_corpus(self, *args, **kwargs) -> torch.Tensor:
        encodings = super().encode_queries(*args, **kwargs)
        return self._inject_noise(encodings)


def evaluate(model_name: str, noise_level: float, dataset: str):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    #### Download scifact.zip dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(datasets_cache_dir, "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    #### Load the SBERT model and retrieve using cosine-similarity
    model = DRES(NoisySentenceBERT("msmarco-distilbert-base-tas-b", noise_level=noise_level), batch_size=16)
    retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    metrics = {
        "ndcg": ndcg,
        "_map": _map,
        "recall": recall,
        "precision": precision,
    }
    print("*** Metrics ***")
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example argument parser")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="sentence-transformers/gtr-t5-base",
        help="Name of the model (default: sentence-transformers/gtr-t5-base)"
    )

    parser.add_argument(
        "-n",
        "--noise",
        "--noise_level",
        type=float,
        default=0.0,
        help="Noise level (default: 0.0)"
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="scifact",
        help="Name of the dataset (default: scifact)"
    )

    args = parser.parse_args()
    evaluate(args.model, args.noise, args.dataset)