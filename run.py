from typing import Optional

import logging
import os
import sys

import datasets
import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint

from collator import CustomCollator
from data_helpers import load_dpr_corpus, NQ_DEV, NQ_TRAIN
from helpers import md5_hash_kwargs
from run_args import ModelArguments, DataTrainingArguments, TrainingArguments
from trainer import InversionTrainer


logger = logging.getLogger(__name__)


def load_embedder_and_tokenizer(name: str):
    # TODO make abstract/argparse for it etc.
    if name == "dpr":
        model = transformers.DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    elif name == "contriever":
        model = transformers.Contriever.from_pretrained("facebook/contriever")
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    return model, tokenizer


def load_model(model_name: str) -> AutoModelForSeq2SeqLM:
    return AutoModelForSeq2SeqLM.from_pretrained(model_name) # for testing


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    exp_name = '__'.join(
        (model_args.model_name_or_path, model_args.embedding_model_name_or_path)
    )

    # Set up output_dir and wandb.
    kwargs_hash = md5_hash_kwargs(**vars(model_args), **vars(data_args), **vars(training_args))
    training_args.output_dir = os.path.join(
        training_args.output_dir, kwargs_hash        
    )

    if training_args.use_wandb:
        import wandb

        wandb.init(
            project="emb-inv-1",
            name=exp_name,
            id=kwargs_hash,
            resume=True,
        )
    else:
        # Disable W&B
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding=True,
        truncation='max_length',
        max_length=model_args.max_seq_length,
    )
    model = load_model(model_name=model_args.model_name_or_path)
    embedder, embedder_tokenizer = load_embedder_and_tokenizer(
        name=model_args.embedding_model_name_or_path
    )

    text_column_name = "text"        
    def tokenize_function(examples):
        output = tokenizer(
            examples[text_column_name],
            padding=True,
            truncation=True,
            max_length=model_args.max_seq_length,
            return_tensors='pt',
        )
        output['labels'] = output['input_ids'] # copy to 'labels' for language modeling loss

        embedder_output = embedder_tokenizer(
            examples[text_column_name],
            padding=True,
            truncation=True,
            max_length=model_args.max_seq_length,
            return_tensors='pt'
        )
        embedder_output = { f'embedder_{k}': v for k,v in embedder_output.items() }

        return {**output, **embedder_output}


    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get and process datasets.
    train_dataset_key = "train"

    logger.info("Loading datasets...")
    raw_datasets = datasets.DatasetDict({
        "train": load_dpr_corpus(NQ_TRAIN),
        "validation": load_dpr_corpus(NQ_DEV),
    })
    column_names = list(raw_datasets[train_dataset_key].features)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=training_args.dataloader_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    train_dataset = tokenized_datasets[train_dataset_key]

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training model from checkpoint `{model_args.model_name_or_path}` - Total size={n_params/2**20:.2f}M params")
    eval_dataset = tokenized_datasets["validation"]
    
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    #############################################################################

    # Initialize our Trainer
    trainer = InversionTrainer(
        embedder=embedder,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=CustomCollator(tokenizer=tokenizer),
    )

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

