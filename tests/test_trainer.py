import pytest
import shlex

import datasets
from transformers import AutoTokenizer, HfArgumentParser, set_seed

from collator import CustomCollator
from data_helpers import load_dpr_corpus, NQ_DEV, NQ_TRAIN
from models import load_encoder_decoder, load_embedder_and_tokenizer, InversionModel
from run_args import ModelArguments, DataTrainingArguments, TrainingArguments
from tokenize_data import tokenize_function
from trainer import InversionTrainer

DEFAULT_ARGS_STR = '--per_device_train_batch_size 32 --max_seq_length 128 --model_name_or_path t5-small --embedding_model_name dpr --num_repeat_tokens 32 --exp_name test-exp-123'
DEFAULT_ARGS = shlex.split(DEFAULT_ARGS_STR)

DEFAULT_ARGS += ['--use_wandb', '0']
DEFAULT_ARGS += ['--fp16', '1']

@pytest.fixture
def trainer() -> InversionTrainer:
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=DEFAULT_ARGS)
    set_seed(training_args.seed)
    
    ###########################################################################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding=True,
        truncation='max_length',
        max_length=model_args.max_seq_length,
    )
    embedder, embedder_tokenizer = load_embedder_and_tokenizer(
        name=model_args.embedding_model_name
    )
    model = InversionModel(
        embedder=embedder,
        encoder_decoder=load_encoder_decoder(
            model_name=model_args.model_name_or_path
        ),
        num_repeat_tokens=model_args.num_repeat_tokens,
        embedder_no_grad=model_args.embedder_no_grad,
        freeze_strategy=model_args.freeze_strategy,
    )
    ###########################################################################
    text_column_name = "text"
    train_dataset_key = "train"
    raw_datasets = datasets.DatasetDict({
        "train": load_dpr_corpus(NQ_TRAIN),
        "validation": load_dpr_corpus(NQ_DEV),
    })
    column_names = list(raw_datasets[train_dataset_key].features)
    tokenized_datasets = raw_datasets.map(
        tokenize_function(tokenizer, embedder_tokenizer, text_column_name, model_args.max_seq_length),
        batched=True,
        num_proc=training_args.dataloader_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    ###########################################################################
    train_dataset = tokenized_datasets[train_dataset_key]
    eval_dataset = tokenized_datasets["validation"]
    # make datasets smaller...
    train_dataset = eval_dataset.select(range(256))
    eval_dataset = eval_dataset.select(range(64))
    ###########################################################################

    training_args.num_train_epochs = 1.0
    training_args.eval_steps = 1

    return InversionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=CustomCollator(tokenizer=tokenizer),
    )

def test_trainer(trainer):
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    print("metrics:", metrics)