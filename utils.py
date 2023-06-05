from typing import Callable

import torch
import tqdm
import transformers
from tenacity import retry, stop_after_attempt, wait_fixed


def emb(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        emb = model.call_embedding_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
    return emb


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
    def wrapped_func():
        # Get local rank (need to support non-DDP).
        try:
            local_rank = torch.distributed.get_rank()
            ddp_enabled = True
        except RuntimeError:
            local_rank = -1
            ddp_enabled = False
        # Run on main worker first.
        if local_rank <= 0:
            result = func()
        # Then everyone waits.
        if ddp_enabled:
            torch.distributed.barrier()
        # Run on other workers now.
        if local_rank > 0:
            result = func()
        # Now everyone waits again.
        if ddp_enabled:
            torch.distributed.barrier()
        return result


def get_global_redis():
    import redis

    REDIS_HOST = "rush-compute-01.tech.cornell.edu"
    REDIS_PORT = 9200
    return redis.Redis(host=REDIS_HOST, host=REDIS_PORT)


def redis_cache(func):
    def wrapper(*args):
        key = "_".join(map(str, args))
        r = get_global_redis()
        cached_value = r.get(key)
        if cached_value:
            return cached_value
        else:
            computed_value = func(*args)
            r.set(key, computed_value)
            return computed_value


@redis_cache
@retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
def get_embeddings_openai(text_list, model="text-embedding-ada-002") -> list:
    # embeddings model: https://platform.openai.com/docs/guides/embeddings/use-cases
    #    api ref: https://platform.openai.com/docs/api-reference/embeddings/create
    # TODO: set up a caching system somehow.
    import openai

    response = openai.Embedding.create(input=text_list, model=model)
    return [e["embedding"] for e in response["data"]]


def embed_api(
    input_ids: torch.Tensor,
    embedder_tokenizer: transformers.PreTrainedTokenizer,
    api_name: str,
) -> torch.Tensor:
    text_list = embedder_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    if api_name.startswith("text-embedding-ada"):
        embeddings = get_embeddings_openai(
            text_list=text_list,
            model=api_name,
        )
    else:
        raise ValueError(f"unsupported api name {api_name}")
    return torch.tensor(embeddings, device=input_ids.device)
