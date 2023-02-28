from typing import Dict

import os
import pytest
import shlex

def set_hf_dir():
    # use local cache dir if exists
    HF_LOCAL_CACHE_DIR = '/scratch/jxm3/.cache/huggingface'
    if os.path.exists(HF_LOCAL_CACHE_DIR):
        os.environ['HF_DATASETS_CACHE'] = os.path.join(HF_LOCAL_CACHE_DIR, 'datasets')
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(HF_LOCAL_CACHE_DIR, 'transformers')
        # os.environ['HF_HOME'] = HF_LOCAL_CACHE_DIR
        print(f'!Set hf cache_dir {HF_LOCAL_CACHE_DIR}')
    else:
        print('Failed to set local cache dir')
set_hf_dir()

import datasets

from collate import DocumentQueryCollatorWithPadding
from data import load_dataset_train_test, load_multiple_datasets
from models import MODEL_NAMES, load_model
from metrics import compute_metrics_contrastive, compute_metrics_retrieval
from train import (
    HfArgumentParser, ModelArguments, DataTrainingArguments, TrainingArguments
)
from trainer import RetrievalTrainer


DEFAULT_ARGS_STR = '--per_device_train_batch_size 128 --per_device_eval_batch_size 32 --exp_name weighted_embeddings --dataset_histogram_strategy_train ones --dataset_histogram_strategy_eval ones --dataset_name msmarco --model_name laprador --num_train_epochs 100 --seed 44'
DEFAULT_ARGS = shlex.split(DEFAULT_ARGS_STR)

DEFAULT_ARGS += ['--use_wandb', '0']
DEFAULT_ARGS += ['--fp16', '1']

def load_beir(tokenizer) -> Dict[str, datasets.Dataset]:
    beir_dict = load_multiple_datasets(
        dataset_name="beir",
        tokenizer=tokenizer
    )
    for beir_dataset in beir_dict.values():
        beir_dataset.tokenize(query_length=64, document_length=350)
    
    return {
        **{f"BeIR/{k}": v for k,v in beir_dict.items()}
    }


@pytest.fixture
def trainer() -> RetrievalTrainer:
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=DEFAULT_ARGS)

    training_args.num_train_epochs = 1
    model, tokenizer = load_model(
        model_args.model_name,
        exp_name=training_args.exp_name,
        prompt_embed_n_tokens=model_args.prompt_embed_n_tokens,
        dataset_embed_n_tokens=model_args.dataset_embed_n_tokens,
        token_encoder_arch=model_args.token_encoder_arch,
    )

    collator = DocumentQueryCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
        return_tensors='pt'
    )

    train_dataset, _ = load_dataset_train_test(
        dataset_name=data_args.dataset_name,
        tokenizer=tokenizer,
        size=training_args.steps_per_epoch,
    )
    train_dataset.tokenize(
        query_length=64, document_length=350
    )
    train_dataset.size = 3000
    training_args.eval_steps = 128 * 10
    trainer = RetrievalTrainer(
        model=model,
        args=training_args,
        ##################################################################
        train_dataset=train_dataset,
        eval_dataset={},
        ##################################################################
        compute_metrics=compute_metrics_contrastive(training_args),
        ##################################################################
        retrieval_datasets=load_beir(tokenizer),
        compute_metrics_retrieval=compute_metrics_retrieval(training_args),
        ##################################################################
        tokenizer=tokenizer,
        data_collator=collator,
    )
    return trainer

def test_trainer(trainer):
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    print("metrics:", metrics)