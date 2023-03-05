import sys
sys.path.append('/home/jxm3/research/retrieval/inversion')

# jxm 3/5/23
# stone st coffee

from typing import Tuple

import collections
import os
import pickle

import datasets
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

def reorder_words_except_padding(input_ids: torch.Tensor) -> torch.Tensor:
    # TODO (not sure how to do this - ask chatGPT lol)
    return input_ids


def embed_all_tokens(model: torch.nn.Module, tokenizer: transformers.AutoTokenizer):
    """Generates embeddings for all tokens in tokenizer vocab."""
    i = 0
    batch_size = 16
    all_token_embeddings = []
    V = tokenizer.vocab_size
    CLS = tokenizer.vocab['[CLS]']
    SEP = tokenizer.vocab['[SEP]']
    pbar = tqdm.tqdm(desc='generating token embeddings', colour='#008080', total=V)
    while i < V:
        # 
        minibatch_size = min(V-i, batch_size)
        inputs = torch.arange(i, i+minibatch_size)
        input_ids = torch.stack([torch.tensor([101]).repeat(len(inputs)), inputs, torch.tensor([102]).repeat(len(inputs))]).T
        input_ids = input_ids.to(device)
        # 
        attention_mask = torch.ones_like(input_ids, device=device)
        # 
        with torch.no_grad():
            token_embeddings = model.call_embedding_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        all_token_embeddings.extend(token_embeddings)
        i += batch_size
        pbar.update(batch_size)
    # 
    all_token_embeddings = torch.cat(all_token_embeddings)
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
    model_name = 'dpr'
    dataset_name = 'nq_dev'
    batch_size = 8
    i = 0
    n = 100

    torch.set_grad_enabled(False)

    model, embedder_tokenizer, tokenizer = load_model_and_tokenizers(
        model_name=model_name
    )
    model.embedder.to(device)
    dataset = load_nq_dev(tokenizer, embedder_tokenizer)[:n]

    word_embeddings = embed_all_tokens(model, embedder_tokenizer)

    metrics = collections.defaultdict(list)
    pbar = tqdm.tqdm(desc='generating token embeddings', colour='#A020F0', total=len(dataset))

    while i < len(dataset):
        data_batch = data[i * batch_size : (i+1) * batch_size]
        embeddings = model.embed(data_batch)
        # 
        reordered_words = reorder_words_except_padding(data_batch, tokenizer.padding_idx)
        reordered_words_embeddings = model.embed(reordered_words)

        # linear/bag-of-words
        word_embeddings = torch.gather(word_embeddings, data_batch)
        word_embeddings = word_embeddings.sum(2)

        # first 16 words & 32 words
        embeddings_8 = model(data_batch[:, :8])
        embeddings_16 = model(data_batch[:, :16])
        embeddings_32 = model(data_batch[:, :32])
        embeddings_64 = model(data_batch[:, :64])

        # random sim
        random_inbatch_embeddings = model(random.shuffle(data_batch))

        # random words
        random_words = torch.random.randint(0, vocab_size, shape=data_batch_shape)
        random_words_embeddings = model(random_words)

        # random vector
        random_embeddings = torch.randn(shape=embeddings.shape)
        
        # 
        pbar.update(batch_size)
        i += batch_size
    #
    pickle.dump(metrics, f'{model_name}_{dataset_name}_emb_metrics.p')

if __name__ == '__main__': main()