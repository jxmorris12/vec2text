import json
import logging
import os
import random
from typing import List, Set

import datasets
import tqdm

DPR_PATH = "/home/jxm3/research/retrieval/DPR/dpr/downloads/data/retriever/{name}.json"
NQ_DEV = "nq-dev"
NQ_TRAIN = "nq-train"


def create_passage__dpr(ctx: dict):
    """(from dpr code)"""
    assert ("psg_id" in ctx.keys()) or (
        "passage_id" in ctx.keys()
    ), f"invalid keys {ctx.keys()}"
    # passage_id = ctx.get("psg_id", ctx.get("passage_id"))
    # return BiEncoderPassage(
    #     normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
    #     ctx["title"],
    #     int(passage_id)
    # )
    #
    # TODO consider using title. DPR concatenates title + [SEP]...
    # TODO also consider normalization -- does DPR not use normalization?
    # (seems disabled in their biencoder_data.py file).
    return ctx["text"]


def load_dpr_corpus_uncached(name: str) -> List[str]:
    path = DPR_PATH.format(name=name)
    assert os.path.exists(path), f"dataset not found: {path}"
    logging.info("Loading DPR dataset from path %s", path)
    items = json.load(open(path, "r", encoding="utf-8"))
    contexts: Set[str] = set()

    ######################################################################
    string_from_dataset = items[0]["positive_ctxs"][0]["text"]
    color = "#" + "".join(hex(ord(x))[2:] for x in string_from_dataset)[:6]
    ######################################################################

    for item in tqdm.tqdm(items, colour=color, desc="Loading dataset", leave=False):
        contexts.update(map(create_passage__dpr, item["positive_ctxs"]))
        contexts.update(map(create_passage__dpr, item["negative_ctxs"]))
        contexts.update(map(create_passage__dpr, item["hard_negative_ctxs"]))

    logging.info("Loaded dataset.")
    return list(contexts)


def load_dpr_corpus(name: str) -> datasets.Dataset:
    cache_path = (
        datasets.config.HF_DATASETS_CACHE
    )  # something like /home/jxm3/.cache/huggingface/datasets
    os.makedirs(os.path.join(cache_path, "emb_inv_dpr"), exist_ok=True)
    dataset_path = os.path.join(cache_path, "emb_inv_dpr", name)

    if os.path.exists(dataset_path):
        logging.info("Loading DPR dataset %s path %s", dataset_path)
        dataset = datasets.load_from_disk(dataset_path)
    else:
        logging.info(
            "Loading DPR dataset %s from JSON (slow) at path %s", name, dataset_path
        )
        corpus = load_dpr_corpus_uncached(name=name)
        dataset = datasets.Dataset.from_list([{"text": t} for t in corpus])
        dataset.save_to_disk(dataset_path)
        logging.info("Saved DPR dataset %s to path %s", name, dataset_path)

    return dataset


def load_msmarco_corpus() -> datasets.Dataset:
    # has columns ["title", "text"]. only one split ("train")
    dataset_dict = datasets.load_dataset("Tevatron/msmarco-passage-corpus")
    return dataset_dict["train"]


def load_luar_reddit() -> datasets.Dataset:
    d = datasets.load_dataset("friendshipkim/reddit_eval_embeddings_luar")
    d = d.rename_column("full_text", "text")
    d = d.rename_column("embedding", "frozen_embeddings")
    return d


def dataset_from_args(data_args) -> datasets.DatasetDict:
    if data_args.dataset_name == "nq":
        raw_datasets = datasets.DatasetDict(
            {
                "train": load_dpr_corpus(NQ_TRAIN),
                "validation": load_dpr_corpus(NQ_DEV),
            }
        )
    elif data_args.dataset_name == "msmarco":
        raw_datasets = load_msmarco_corpus()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]
    elif data_args.dataset_name == "luar_reddit":
        all_luar_datasets = load_luar_reddit()
        raw_datasets = datasets.DatasetDict(
            {
                "train": all_luar_datasets["candidates"],
                "validation": all_luar_datasets["queries"],
            }
        )
    else:
        raise ValueError(f"unsupported dataset {data_args.dataset_name}")
    return raw_datasets


def load_ag_news_test() -> datasets.Dataset:
    return datasets.load_dataset("ag_news")["test"]


def load_xsum_val(col: str) -> datasets.Dataset:
    d = datasets.load_dataset("xsum")["validation"]
    d = d.rename_column(col, "text")
    return d


def load_wikibio_val() -> datasets.Dataset:
    d = datasets.load_dataset("wiki_bio")["val"]
    d = d.rename_column("target_text", "text")
    return d


def load_arxiv_val() -> datasets.Dataset:
    d = datasets.load_dataset("ccdv/arxiv-summarization")["validation"]
    d = d.rename_column("abstract", "text")
    return d


def retain_dataset_columns(
    d: datasets.Dataset, allowed_columns: List[str]
) -> datasets.Dataset:
    column_names_to_remove = [c for c in d.features if c not in allowed_columns]
    return d.remove_columns(column_names_to_remove)


def load_beir_corpus(name: str) -> List[str]:
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader

    #### Download scifact.zip dataset and unzip the dataset
    beir_datasets_cache_dir = "/home/jxm3/research/retrieval/distractor_exp"

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        name
    )
    out_dir = os.path.join(beir_datasets_cache_dir, "datasets")
    data_path = beir_util.download_and_unzip(url, out_dir)

    # Limit each corpus to first 100k documents.
    MAX_N = 100_000

    if name == "cqadupstack":
        full_corpus = []
        for folder in [
            "android",
            "english",
            "gaming",
            "gis",
            "mathematica",
            "physics",
            "programmers",
            "stats",
            "tex",
            "unix",
            "webmasters",
            "wordpress",
        ]:
            corpus, _queries, _qrels = GenericDataLoader(
                data_folder=os.path.join(data_path, folder)
            ).load(split="test")
            full_corpus.extend([k["text"] for k in corpus.values()])
        random.shuffle(full_corpus)
        return full_corpus[:MAX_N]
    else:
        corpus, _queries, _qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )
        corpus = [k["text"] for k in corpus.values()]
        return corpus[:MAX_N]


def load_beir_dataset(name: str) -> datasets.Dataset:
    cache_path = (
        datasets.config.HF_DATASETS_CACHE
    )  # something like /home/jxm3/.cache/huggingface/datasets
    dataset_path = os.path.join(cache_path, "emb_inv_beir", name)
    print(f"loading BEIR dataset: {name}")
    if os.path.exists(dataset_path):
        logging.info("Loading BEIR dataset %s path %s", dataset_path)
        dataset = datasets.load_from_disk(dataset_path)
    else:
        logging.info(
            "Loading BEIR dataset %s from JSON (slow) at path %s", dataset_path
        )
        corpus = load_beir_corpus(name=name)
        dataset = datasets.Dataset.from_list([{"text": t} for t in corpus])
        os.makedirs(os.path.join(cache_path, "emb_inv_beir"), exist_ok=True)
        dataset.save_to_disk(dataset_path)
        logging.info("Saved BEIR dataset as HF path %s", dataset_path)
    return dataset


def load_beir_datasets() -> datasets.DatasetDict:
    all_beir_datasets = [
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
    return datasets.DatasetDict({k: load_beir_dataset(k) for k in all_beir_datasets})


def load_standard_val_datasets() -> datasets.DatasetDict:
    """Loads a pre-defined set of standard val datasets."""
    d = {
        "ag_news": load_ag_news_test(),
        "arxiv": load_arxiv_val(),
        # "xsum_doc": load_xsum_val("document"),
        # "xsum_summ": load_xsum_val("summary"),
        "wikibio": load_wikibio_val(),
    }
    d = {k: retain_dataset_columns(v, ["text"]) for k, v in d.items()}

    return datasets.DatasetDict(d)
