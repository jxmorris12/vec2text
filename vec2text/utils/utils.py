import math
import multiprocessing
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import datasets
import numpy as np
import torch
import tqdm
import transformers
from tenacity import retry, stop_after_attempt, wait_fixed

datasets.disable_caching()


def emb(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        emb = model.call_embedding_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
    return emb


def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def get_num_proc() -> int:
    world_size: int = get_world_size()
    try:
        # os.sched_getaffinity respects schedulers, unlike cpu_count(), but it's only available
        # on some Unix platforms, so we support both!
        return len(os.sched_getaffinity(0)) // world_size  # type: ignore[attr-defined]
    except AttributeError:
        return multiprocessing.cpu_count() // world_size


def embed_all_tokens(model: torch.nn.Module, tokenizer: transformers.AutoTokenizer):
    """Generates embeddings for all tokens in tokenizer vocab."""
    i = 0
    model.embedder.eval()
    batch_size = 1024
    all_token_embeddings = []
    V = tokenizer.vocab_size
    #
    # DPR has CLS and SEP.
    # GTR has no CLS or start token at all, and has EOS at the end.
    CLS = tokenizer.cls_token_id
    SEP = (tokenizer.sep_token_id) or (tokenizer.eos_token_id)
    assert SEP is not None
    #
    device = next(model.parameters()).device
    pbar = tqdm.tqdm(
        desc="generating token embeddings", colour="#008080", total=V, leave=False
    )
    while i < V:
        #
        minibatch_size = min(V - i, batch_size)
        inputs = torch.arange(i, min(i + minibatch_size, V))
        #
        if CLS is not None:
            input_ids = torch.stack(
                [
                    torch.tensor([CLS]).repeat(len(inputs)),
                    inputs,
                    torch.tensor([SEP]).repeat(len(inputs)),
                ]
            ).T
        else:
            input_ids = torch.stack([inputs, torch.tensor([SEP]).repeat(len(inputs))]).T
        input_ids = input_ids.to(device)
        #
        attention_mask = torch.ones_like(input_ids, device=device)
        #
        with torch.no_grad():
            token_embeddings = emb(model, input_ids, attention_mask)
        all_token_embeddings.extend(token_embeddings)
        i += batch_size
        pbar.update(batch_size)
    #
    all_token_embeddings_tensor: torch.Tensor = torch.stack(all_token_embeddings)
    assert all_token_embeddings_tensor.shape == (tokenizer.vocab_size, 768)

    all_token_embeddings_tensor /= all_token_embeddings_tensor.norm(
        p=2, dim=1, keepdim=True
    )
    return all_token_embeddings_tensor


def torch_main_worker_finish_first(func: Callable):
    def wrapper(*args, **kwargs):
        # Get local rank (need to support non-DDP).
        try:
            local_rank = torch.distributed.get_rank()
            ddp_enabled = True
        except (RuntimeError, ValueError):
            local_rank = -1
            ddp_enabled = False
        is_main_worker = local_rank <= 0
        # Run on main worker first.
        if is_main_worker:
            result = func(*args, **kwargs)
        # Then everyone waits.
        if ddp_enabled:
            torch.distributed.barrier()
        # Run on other workers now.
        if not is_main_worker:
            result = func(*args, **kwargs)
        # Now everyone waits again.
        if ddp_enabled:
            torch.distributed.barrier()
        return result

    return wrapper


def dataset_map_multi_worker(
    dataset: datasets.Dataset, map_fn: Callable, *args, **kwargs
) -> datasets.Dataset:

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        kwargs["num_proc"] = kwargs.get("num_proc", get_num_proc())
    except (RuntimeError, ValueError):
        # In non-distributed mode, just run regular map()
        kwargs["num_proc"] = kwargs.get("num_proc", get_num_proc())
        return dataset.map(map_fn, *args, **kwargs)
    datasets.disable_caching()

    cache_path = os.environ.get(
        "VEC2TEXT_CACHE", os.path.expanduser("~/.cache/inversion")
    )
    ds_shard_filepaths = [
        os.path.join(cache_path, f"{dataset._fingerprint}_subshard_{w}.cache")
        for w in range(0, world_size)
    ]
    print(f"\tworker {rank} saving sub-shard to {ds_shard_filepaths[rank]}")
    ds_shard = dataset.shard(
        num_shards=world_size,
        index=rank,
        contiguous=True,
    )
    ds_shard = ds_shard.map(map_fn, *args, **kwargs)
    ds_shard.save_to_disk(ds_shard_filepaths[rank])
    print("rank", rank, "saving:", ds_shard_filepaths[rank])
    torch.distributed.barrier()
    full_dataset = datasets.concatenate_datasets(
        [datasets.load_from_disk(p) for p in ds_shard_filepaths]
    )
    torch.distributed.barrier()
    print("rank", rank, "deleting:", ds_shard_filepaths[rank])
    shutil.rmtree(ds_shard_filepaths[rank])
    return full_dataset


manifest_object = None


def get_manifest_global():
    from manifest import Manifest

    global manifest_object
    if manifest_object is None:
        manifest_object = Manifest(
            client_name="openaiembedding",  # defaults to 'text-embedding-ada-002'
            # cache_name="sqlite",
            # cache_connection="/home/jxm3/.manifest/jxm_openai_manifest.sqlite",
        )
        # manifest_object.PARAMS = {
        #     'engine': ('model', 'text-embedding-ada-002'),
        #     'batch_size': ('batch_size', 128),
        # }
    return manifest_object


@retry(wait=wait_fixed(1), stop=stop_after_attempt(15))
def get_embeddings_openai_manifest(
    text_list, model="text-embedding-ada-002"
) -> np.ndarray:
    # embeddings model: https://platform.openai.com/docs/guides/embeddings/use-cases
    #    api ref: https://platform.openai.com/docs/api-reference/embeddings/create
    # TODO: set up a caching system somehow.
    manifest = get_manifest_global()
    # print(
    #     f"running manifest on text_list of length {len(text_list)}, first element '{text_list[0]}'"
    # )
    return np.array(manifest.run(text_list, batch_size=min(len(text_list), 128)))


@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def get_embeddings_openai_vanilla_multithread(
    text_list, model="text-embedding-ada-002"
) -> list:
    from openai import OpenAI

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

    return outputs


@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def get_embeddings_openai_vanilla(text_list, model="text-embedding-ada-002") -> list:
    # embeddings model: https://platform.openai.com/docs/guides/embeddings/use-cases
    #    api ref: https://platform.openai.com/docs/api-reference/embeddings/create
    # TODO: set up a caching system somehow.
    from openai import OpenAI

    client = OpenAI()

    # print(f"running openai on text_list of length {len(text_list)}, first element '{text_list[0]}'")
    batches = math.ceil(len(text_list) / 128)
    outputs = []
    for batch in range(batches):
        text_list_batch = text_list[batch * 128 : (batch + 1) * 128]
        response = client.embeddings.create(
            input=text_list_batch, model=model, encoding_format="float"
        )
        outputs.extend([e.embedding for e in response.data])
    return outputs


@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def embed_api(
    input_ids: torch.Tensor,
    embedder_tokenizer: transformers.PreTrainedTokenizer,
    api_name: str,
) -> torch.Tensor:
    text_list = embedder_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    # get_embeddings_func = get_embeddings_openai_vanilla
    get_embeddings_func = get_embeddings_openai_vanilla_multithread
    # get_embeddings_func = get_embeddings_openai_manifest
    if api_name.startswith("text-embedding-ada"):
        embeddings = get_embeddings_func(
            text_list=text_list,
            model=api_name,
        )
    else:
        raise ValueError(f"unsupported api name {api_name}")

    return torch.tensor(embeddings, device=input_ids.device, dtype=torch.float32)


class MockEmbedder:
    embedder_dim: int

    def __init__(self, embedder_dim: int):
        self.embedder_dim = embedder_dim

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], self.embedder_dim),
            dtype=torch.float32,
            device=input_ids.device,
        )

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], self.embedder_dim),
            dtype=torch.float32,
            device=input_ids.device,
        )
