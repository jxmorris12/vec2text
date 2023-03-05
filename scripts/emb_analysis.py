import sys
sys.path.append('/home/jxm3/research/retrieval/inversion')

# jxm 3/5/23
# stone st coffee

from typing import Tuple

import collections
import datasets
import pickle

import torch
import transformers
import tqdm

from data_helpers import load_dpr_corpus, NQ_DEV
from models import load_encoder_decoder, load_embedder_and_tokenizer, InversionModel
from tokenize_data import tokenize_function


max_seq_length = 128

# TODO impl caching

def reorder_words_except_padding(input_ids: torch.Tensor) -> torch.Tensor:
    # TODO (not sure how to do this - ask chatGPT lol)
    return input_ids


def embed_all_tokens(model: torch.nn.Module, tokenizer: transformers.AutoTokenizer):
    i = 0
    batch_size = 8
    all_token_embeddings = []
    while i < tokenizer.vocab_size:
        # TODO add CLS token
        inputs = torch.arange(i, i+batch_size)
        token_embeddings = model(inputs)
        all_token_embeddings.extend(token_embeddings)
        i += batch_size
    return all_token_embeddings


def load_model_and_tokenizers(model_name: str) -> Tuple[InversionModel, transformers.AutoTokenizer]:
    embedder, embedder_tokenizer = load_embedder_and_tokenizer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
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


def load_nq_dev() -> datasets.Dataset:
    raw_datasets = datasets.DatasetDict({
        "validation": load_dpr_corpus(NQ_DEV),
    })
    
    column_names = list(raw_datasets[train_dataset_key].features)

    tokenized_datasets = raw_datasets.map(
        tokenize_function(tokenizer, embedder_tokenizer, "text", max_seq_length),
        batched=True,
        num_proc=training_args.dataloader_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    return raw_datasets["validation"]


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
    dataset = load_dataset(dataset_name)[:n]

    word_embeddings = embed_all_tokens(tokens, model)

    metrics = collections.defaultdict(list)

    while i < len(data):
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
        
        i += batch_size
    #
    pickle.dump(metrics, f'{model_name}_{dataset_name}_emb_metrics.p')

if __name__ == '__main__': main()