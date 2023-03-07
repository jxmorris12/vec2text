from typing import Tuple

import sys
sys.path.append('/home/jxm3/research/retrieval/inversion')

import argparse
import os
import pickle

import datasets
import torch
import transformers

from data_helpers import load_dpr_corpus, NQ_DEV
from models import (
    InversionModel, 
    MODEL_NAMES, FREEZE_STRATEGIES, 
    load_embedder_and_tokenizer, load_encoder_decoder
)
from tokenize_data import tokenize_function
from utils import emb


num_workers = len(os.sched_getaffinity(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_and_tokenizers(model_name: str, max_seq_length:int) -> Tuple[InversionModel, transformers.AutoTokenizer]:
    embedder, embedder_tokenizer = load_embedder_and_tokenizer(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "t5-small",
        padding=True,
        truncation='max_length',
        max_length=max_seq_length,
    )
    model = InversionModel(
        embedder=embedder,
        embedder_tokenizer=embedder_tokenizer,
        encoder_decoder=load_encoder_decoder(model_name="t5-small"),
        num_repeat_tokens=1,
        embedder_no_grad=True,
        freeze_strategy="none",
    )
    return model, embedder_tokenizer, tokenizer


def load_nq_dev(
        tokenizer: transformers.PreTrainedTokenizer,
        embedder_tokenizer: transformers.PreTrainedTokenizer,
        max_seq_length: int,
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


def parse_args() -> argparse.ArgumentParser:
        # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description='Get embeddings from a pre-trained model')
    parser.add_argument('--model_name',
        type=str,
        required=True,
        help='The name of the pre-trained model to get embeddings from',
        choices=MODEL_NAMES,
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='nq_dev',
        help='The name of the pre-trained model to get embeddings from',
        choices=['nq_dev'],
    )
    parser.add_argument(
        '--n',
        type=int,
        default=1000,
        help='Max number of examples to use',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Inference batch size',
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=128,
        help='Inference batch size',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = f'{args.dataset_name}__{args.model_name}.p'
    out_path = os.path.join(dir_path, os.pardir, 'embeddings', filename)
    out_path = os.path.normpath(out_path)
    print(f'writing embeddings to {out_path}')

    # model
    model, embedder_tokenizer, tokenizer = load_model_and_tokenizers(
        model_name=args.model_name, max_seq_length=args.max_seq_length,
    )
    model.embedder.to(device)

    # dataset
    assert args.dataset_name == "nq_dev"
    dataset = load_nq_dev(tokenizer, embedder_tokenizer, max_seq_length=args.max_seq_length)[:args.n]

    # compute embeddings
    batch_size = args.batch_size
    i = 0
    all_embeddings = []
    pbar = tqdm.tqdm(desc='getting embeddings for dataset', colour='#A020F0', total=n)
    while i < args.n:
        input_ids = torch.tensor(dataset['embedder_input_ids'][i:i+batch_size], device=device)
        attention_mask = torch.tensor(dataset['embedder_attention_mask'][i:i+batch_size], device=device)
        embeddings = emb(model, input_ids, attention_mask)
        all_embeddings.extend(embeddings.cpu())
        i += batch_size
        pbar.update(batch_size)
    
    all_embeddings = torch.stack(all_embeddings)
    pickle.dump(all_embeddings, open(out_path, 'wb'))
    print(f'wrote {len(all_embeddings)} embeddings to {out_path}')
    
if __name__ == '__main__': main()