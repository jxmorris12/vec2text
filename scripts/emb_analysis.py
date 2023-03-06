import sys
sys.path.append('/home/jxm3/research/retrieval/inversion')

# jxm 3/5/23
# stone st coffee

from typing import Tuple

import collections
import os
import pickle
import random

import datasets
import numpy as np
import torch
import transformers
import tqdm

from data_helpers import load_dpr_corpus, NQ_DEV
from models import load_encoder_decoder, load_embedder_and_tokenizer, InversionModel
from tokenize_data import tokenize_function

num_workers = len(os.sched_getaffinity(0))
max_seq_length = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO impl caching

def emb(
        model: torch.nn.Module, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
    with torch.no_grad():
        emb = model.call_embedding_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
    return emb

def reorder_words_except_padding(input_ids: torch.Tensor, padding_idx: int) -> torch.Tensor:
    # TODO
    pad_begin_idxs = (input_ids == padding_idx).int().argmax(dim=1)
    pad_begin_idxs = torch.where(pad_begin_idxs == 0, input_ids.shape[1], pad_begin_idxs)

    # assume there is a cls token if all first words are the same.
    has_cls = len(set(input_ids[:, 0].tolist())) == 1

    # TODO vectorize
    a = []
    for i in range(len(input_ids)):
        p = pad_begin_idxs[i].item()
        if has_cls:
            text = input_ids[i, 1:p].tolist()
        else:
            text = input_ids[i, 0:p].tolist()
        random.shuffle(text)
        text = torch.tensor(text, device=input_ids.device)
        if has_cls:
            s = torch.cat([input_ids[None, i, 0], text, input_ids[i, p:]])
        else:
            s = torch.cat([text, input_ids[i, p:]])
        a.append(s)

    return torch.stack(a)


def embed_all_tokens(model: torch.nn.Module, tokenizer: transformers.AutoTokenizer):
    """Generates embeddings for all tokens in tokenizer vocab."""
    i = 0
    batch_size = 1024
    all_token_embeddings = []
    V = tokenizer.vocab_size
    #
    CLS = tokenizer.bos_token_id
    SEP = tokenizer.eos_token_id
    #
    pbar = tqdm.tqdm(desc='generating token embeddings', colour='#008080', total=V)
    while i < V:
        # 
        minibatch_size = min(V-i, batch_size)
        inputs = torch.arange(i, min(i+minibatch_size, V))

        if (CLS is not None):
            input_ids = torch.stack([torch.tensor([CLS]).repeat(len(inputs)), inputs, torch.tensor([SEP]).repeat(len(inputs))]).T
        else:
            input_ids = torch.stack([inputs, torch.tensor([SEP]).repeat(len(inputs))]).T
        input_ids = input_ids.to(device)
        # 
        attention_mask = torch.ones_like(input_ids, device=device)
        # 
        token_embeddings = emb(model, input_ids, attention_mask)
        all_token_embeddings.extend(token_embeddings)
        i += batch_size
        pbar.update(batch_size)
    # 
    all_token_embeddings = torch.stack(all_token_embeddings)
    print('all_token_embeddings.shape:', all_token_embeddings.shape)
    assert all_token_embeddings.shape == (tokenizer.vocab_size, 768)
    return all_token_embeddings


def load_model_and_tokenizers(model_name: str) -> Tuple[InversionModel, transformers.AutoTokenizer]:
    embedder, embedder_tokenizer = load_embedder_and_tokenizer(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "t5-small",
        padding=True,
        truncation='max_length',
        max_length=max_seq_length,
    )
    model = InversionModel(
        embedder=embedder,
        encoder_decoder=load_encoder_decoder(model_name="t5-small"),
        num_repeat_tokens=1,
        embedder_no_grad=True,
        freeze_strategy="none",
    )
    return model, embedder_tokenizer, tokenizer


def load_nq_dev(
        tokenizer: transformers.PreTrainedTokenizer,
        embedder_tokenizer: transformers.PreTrainedTokenizer
    ) -> datasets.Dataset:
    raw_datasets = datasets.DatasetDict({
        "validation": load_dpr_corpus(NQ_DEV),
    })
    
    column_names = list(raw_datasets["validation"].features)

    tokenized_datasets = raw_datasets.map(
        tokenize_function(tokenizer, embedder_tokenizer, "text", max_seq_length),
        batched=True,
        num_proc=num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
    return tokenized_datasets["validation"]


def main():
    # model_name = 'dpr'
    model_name = 'gtr_base'
    dataset_name = 'nq_dev'
    batch_size = 128
    i = 0
    n = 10_000

    torch.set_grad_enabled(False)

    csf = torch.nn.CosineSimilarity(dim=1)
    cs = lambda a,b: csf(a, b).cpu().tolist()

    model, embedder_tokenizer, tokenizer = load_model_and_tokenizers(
        model_name=model_name
    )
    model.embedder.to(device)
    dataset = load_nq_dev(tokenizer, embedder_tokenizer)[:n]

    word_embeddings = embed_all_tokens(model, embedder_tokenizer)

    metrics = collections.defaultdict(list)
    pbar = tqdm.tqdm(desc='getting embeddings for dataset', colour='#A020F0', total=n)

    while i < n:
        input_ids = torch.tensor(dataset['embedder_input_ids'][i:i+batch_size], device=device)
        attention_mask = torch.tensor(dataset['embedder_attention_mask'][i:i+batch_size], device=device)
        embeddings = emb(model, input_ids, attention_mask)
        # 
        reordered_input_ids = reorder_words_except_padding(input_ids, embedder_tokenizer.pad_token_id)
        reordered_words_embeddings = emb(model, reordered_input_ids, attention_mask)
        metrics['words_reorder'].extend(cs(embeddings, reordered_words_embeddings))

        # linear/bag-of-words
        linear_word_embeddings = (word_embeddings[input_ids] * attention_mask[..., None])
        linear_word_embeddings = linear_word_embeddings.sum(dim=1)
        metrics['words_sum'].extend(cs(embeddings, linear_word_embeddings))

        # first 16 words & 32 words
        embeddings_8 = emb(model, input_ids[:, :8], attention_mask[:, :8])
        metrics['first_8'].extend(cs(embeddings, embeddings_8))
        embeddings_16 = emb(model, input_ids[:, :16], attention_mask[:, :16])
        metrics['first_16'].extend(cs(embeddings, embeddings_16))
        embeddings_32 = emb(model, input_ids[:, :32], attention_mask[:, :32])
        metrics['first_32'].extend(cs(embeddings, embeddings_32))
        embeddings_64 = emb(model, input_ids[:, :64], attention_mask[:, :64])
        metrics['first_64'].extend(cs(embeddings, embeddings_64))

        # random sim
        ridx = list(range(len(input_ids)))
        random.shuffle(ridx)
        ridx = torch.tensor(ridx, device=device)
        random_inbatch_embeddings = emb(model, input_ids[ridx], attention_mask[ridx])
        metrics['random_language'].extend(cs(embeddings, random_inbatch_embeddings))

        # random words
        first_word_id = max(embedder_tokenizer.all_special_ids) + 1
        random_input_ids = torch.randint(low=0, high=embedder_tokenizer.vocab_size, size=input_ids.shape, device=device)
        random_words_embeddings = emb(model, random_input_ids, torch.ones_like(random_input_ids, device=device))
        metrics['random_words'].extend(cs(embeddings, random_words_embeddings))

        # random vector
        random_gaussian_embeddings = torch.randn(size=embeddings.shape, device=device)
        metrics['random_gaussian'].extend(cs(embeddings, random_gaussian_embeddings))
        # 
        pbar.update(batch_size)
        i += batch_size
    #
    print('[Mean]')
    print({ k: np.mean(v) for k,v in metrics.items()})
    print('[Std]')
    print({ k: np.std(v) for k,v in metrics.items()})
    pickle.dump(metrics, open(f'{model_name}_{dataset_name}_emb_metrics.p', 'wb'))

if __name__ == '__main__': main()