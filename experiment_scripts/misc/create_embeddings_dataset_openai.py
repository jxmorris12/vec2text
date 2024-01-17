import argparse
import functools
import math
from typing import Dict

import datasets
import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_fixed

from vec2text.data_helpers import (
    load_beir_datasets,
    load_standard_val_datasets,
    retain_dataset_columns,
)
from vec2text.models import load_embedder_and_tokenizer
from vec2text.models.model_utils import mean_pool
from vec2text.utils import dataset_map_multi_worker

MAX_LENGTH = 128
OPENAI_ADA2_MODEL = "text-embedding-ada-002"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process dataset and embedders")
    parser.add_argument("dataset", type=str, help="Path or name of the dataset")
    return parser.parse_args()


encoding = tiktoken.encoding_for_model("text-embedding-ada-002")


@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def embed_openai_ada2(example: Dict) -> Dict:
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    text_tokens = encoding.encode_batch(example["text"])
    text_tokens = [tok[:MAX_LENGTH] for tok in text_tokens]
    text_list = encoding.decode_batch(text_tokens)

    model = OPENAI_ADA2_MODEL
    client = OpenAI()

    # print(f"running openai on text_list of length {len(text_list)}, first element '{text_list[0]}'")

    batches = math.ceil(len(text_list) / 128)
    outputs = []

    for i in range(len(text_list)):
        if len(text_list[i]) == 0:
            print(f"warning: set element {i} to a random sequence")
            text_list[i] = "random sequence"

    def process_batch(batch):
        text_list_batch = text_list[batch * 128 : (batch + 1) * 128]
        response = client.embeddings.create(
            input=text_list_batch, model=model, encoding_format="float"
        )
        return [e.embedding for e in response.data]

    with ThreadPoolExecutor() as executor:
        batch_indices = range(batches)
        results = executor.map(process_batch, batch_indices)

        for result in results:
            outputs.extend(result)

    example["text"] = text_list
    example["embeddings_A"] = outputs

    return example


def main():
    datasets.disable_caching()
    args = parse_args()

    full_name = "__".join(
        (
            args.dataset,
            "openai_ada2",
        )
    )

    all_datasets = {
        **load_standard_val_datasets(),
        **load_beir_datasets(),
    }
    print("Available datasets:", all_datasets.keys())
    assert (
        args.dataset in all_datasets
    ), f"unknown dataset {args.dataset}; choices {all_datasets.keys()}"
    dataset = all_datasets[args.dataset]

    print(f"[*] embedding {args.dataset}")
    dataset = dataset_map_multi_worker(
        dataset,
        batched=True,
        batch_size=128,
        map_fn=functools.partial(embed_openai_ada2),
        num_proc=1,
    )
    print(f"pushing to hub with name {full_name}")

    dataset = retain_dataset_columns(dataset, ["text", "embeddings_A"])
    dataset.push_to_hub(full_name)
    print("done")


if __name__ == "__main__":
    main()
