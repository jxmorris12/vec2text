import argparse
import functools
import os
from typing import Dict

import datasets
import torch
import transformers

from vec2text.data_helpers import (
    load_beir_datasets,
    load_standard_val_datasets,
    retain_dataset_columns,
)
from vec2text.models import load_embedder_and_tokenizer
from vec2text.models.model_utils import mean_pool
from vec2text.utils import dataset_map_multi_worker

MAX_LENGTH = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process dataset and embedders")
    parser.add_argument("dataset", type=str, help="Path or name of the dataset")
    parser.add_argument(
        "--embedderA",
        type=str,
        default="gtr_base",
        help="Name or identifier of the first embedder",
    )
    parser.add_argument(
        "--embedderB",
        type=str,
        default="dpr",
        help="Name or identifier of the second embedder",
    )
    return parser.parse_args()


def tokenize(
    example: Dict,
    tokenizerA: transformers.PreTrainedTokenizer,
    tokenizerB: transformers.PreTrainedTokenizer,
) -> Dict:
    tA = tokenizerA(
        example["text"],
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    example["A_input_ids"] = tA.input_ids
    example["A_attention_mask"] = tA.attention_mask

    tB = tokenizerB(
        example["text"],
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    example["B_input_ids"] = tB.input_ids
    example["B_attention_mask"] = tB.attention_mask

    return example


def embed(
    ex: Dict,
    embedderA: transformers.PreTrainedModel,
    embedderB: transformers.PreTrainedModel,
) -> Dict:
    with torch.no_grad():
        A_input_ids = torch.tensor(ex["A_input_ids"]).cuda()
        A_attention_mask = torch.tensor(ex["A_attention_mask"]).cuda()
        outputsA = embedderA(
            input_ids=A_input_ids,
            attention_mask=A_attention_mask,
        ).last_hidden_state

        if hasattr(outputsA, "pooler_output") and (outputsA.pooler_output is not None):
            ex["embeddings_A"] = outputsA.pooler_output
        else:
            ex["embeddings_A"] = mean_pool(outputsA, A_attention_mask)

    with torch.no_grad():
        B_input_ids = torch.tensor(ex["B_input_ids"]).cuda()
        B_attention_mask = torch.tensor(ex["B_attention_mask"]).cuda()
        outputsB = embedderB(
            input_ids=B_input_ids,
            attention_mask=B_attention_mask,
        )
        if hasattr(outputsB, "pooler_output") and (outputsB.pooler_output is not None):
            ex["embeddings_B"] = outputsB.pooler_output
        else:
            ex["embeddings_B"] = mean_pool(outputsB.last_hidden_state, B_attention_mask)

    return ex


def main():
    datasets.disable_caching()
    args = parse_args()

    embedderA, tokenizerA = load_embedder_and_tokenizer(
        args.embedderA, torch_dtype=torch.float32
    )
    embedderB, tokenizerB = load_embedder_and_tokenizer(
        args.embedderB, torch_dtype=torch.float32
    )

    full_name = "__".join((args.dataset, args.embedderA, args.embedderB))

    all_datasets = {
        **load_standard_val_datasets(),
        **load_beir_datasets(),
    }
    print("Available datasets:", all_datasets.keys())
    assert (
        args.dataset in all_datasets
    ), f"unknown dataset {args.dataset}; choices {all_datasets.keys()}"
    dataset = all_datasets[args.dataset]

    print(f"[1/2] tokenizing {args.dataset}")
    dataset = dataset_map_multi_worker(
        dataset,
        batched=True,
        batch_size=1000,
        map_fn=functools.partial(
            tokenize, tokenizerA=tokenizerA, tokenizerB=tokenizerB
        ),
        num_proc=len(os.sched_getaffinity(0)),
    )

    print(f"[2/2] embedding {args.dataset}")
    embedderA.to("cuda")
    embedderB.to("cuda")
    dataset = dataset_map_multi_worker(
        dataset,
        batched=True,
        batch_size=128,
        map_fn=functools.partial(embed, embedderA=embedderA, embedderB=embedderB),
        num_proc=1,
    )
    print(f"pushing to hub with name {full_name}")

    dataset = retain_dataset_columns(dataset, ["text", "embeddings_A", "embeddings_B"])
    dataset.push_to_hub(full_name)
    print("done")


if __name__ == "__main__":
    main()
