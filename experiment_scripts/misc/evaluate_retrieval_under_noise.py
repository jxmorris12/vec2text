import argparse
import json
import logging
import os

import pandas as pd
import torch
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

results_dir = "/home/jxm3/research/retrieval/inversion/results_defense"
datasets_cache_dir = "/home/jxm3/research/retrieval/distractor_exp"
all_datasets = [
    ####### public datasets #######
    "arguana",
    "climate-fever",
    "cqadupstack",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "trec-covid",
    "webis-touche2020",
    ####### private datasets #######
    "signal1m",
    "trec-news",
    "robust04",
    "bioasq",
]


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


def evaluate(
    model_name: str, noise_level: float, dataset: str, max_seq_length: int = None
):
    model_name_str = model_name.replace("/", "_").replace("-", "_")
    save_path = os.path.join(
        results_dir, f"retrieval_noisy__{model_name_str}__{dataset}__{noise_level}"
    )
    if max_seq_length is not None:
        save_path += f"__{max_seq_length}"
    save_path += ".json"
    if os.path.exists(save_path):
        print(f"found experiment cached at {save_path}.")
        return json.load(open(save_path, "r"))
        exit()
    #### Just some code to print debug information to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    #### /print debug information to stdout

    #### Download scifact.zip dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset
    )
    out_dir = os.path.join(datasets_cache_dir, "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    #### Load the SBERT model and retrieve using cosine-similarity
    model = DRES(NoisySentenceBERT(model_name, noise_level=noise_level), batch_size=512)

    if max_seq_length is not None:
        model.model.q_model.max_seq_length = max_seq_length
        model.model.doc_model.max_seq_length = max_seq_length
    retriever = EvaluateRetrieval(
        model, score_function="cos_sim"
    )  # or "cos_sim" for cosine similarity
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values
    )
    metrics = {
        "ndcg": ndcg,
        "_map": _map,
        "recall": recall,
        "precision": precision,
    }
    print("*** Metrics ***")
    print(metrics)
    json.dump(metrics, open(save_path, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example argument parser")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="sentence-transformers/gtr-t5-base",
        help="Name of the model (default: sentence-transformers/gtr-t5-base)",
    )

    parser.add_argument(
        "-n",
        "--noise",
        "--noise_level",
        type=float,
        default=0.0,
        help="Noise level (default: 0.0)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="scifact",
        help="Name of the dataset (default: scifact)",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=None, help="max sequence length"
    )

    args = parser.parse_args()
    if args.dataset == "all":
        ###########################################################
        model_name = args.model.replace("/", "_").replace("-", "_")
        save_path = os.path.join(
            results_dir, f"retrieval_noisy__{model_name}__{args.noise}.df.parquet"
        )
        if os.path.exists(save_path):
            print(f"found experiment cached at {save_path}. exiting.")
            exit()
        ###########################################################
        all_metrics = []
        for dataset in all_datasets:
            all_metrics.append(
                evaluate(
                    args.model, args.noise, dataset, max_seq_length=args.max_seq_length
                )
            )
        ###########################################################
        df = pd.DataFrame(all_metrics)
        df.to_parquet(save_path)
        ###########################################################
    else:
        evaluate(
            args.model, args.noise, args.dataset, max_seq_length=args.max_seq_length
        )
