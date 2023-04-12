import hashlib
import json
import logging
import os

import datasets
import torch
import transformers
from transformers import AutoTokenizer, set_seed

from collator import CustomCollator
from data_helpers import load_dpr_corpus, load_luar_reddit, NQ_DEV, NQ_TRAIN
from models import load_encoder_decoder, load_embedder_and_tokenizer, InversionModel
from tokenize_data import tokenize_function
from trainer import InversionTrainer


logger = logging.getLogger(__name__)


def md5_hash_kwargs(**kwargs) -> str:
    # We ignore special hf args that start with _ like '__cached__setup_devices'.
    safe_kwargs = {k: str(v) for k,v in kwargs.items() if not k.startswith('_')}
    s = json.dumps(safe_kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


def trainer_from_args(model_args, data_args, training_args) -> InversionTrainer:
    set_seed(training_args.seed)

    name_args = (training_args.exp_group_name, training_args.exp_name, model_args.model_name_or_path, model_args.embedder_model_name)
    name_args = [n for n in name_args if len(n)]
    exp_name = '__'.join(name_args)
    
    # Set up output_dir and wandb.
    kwargs_hash = md5_hash_kwargs(**vars(model_args), **vars(data_args), **vars(training_args))
    training_args.output_dir = os.path.join(
        training_args.output_dir, kwargs_hash        
    )

    if training_args.use_wandb and (training_args.local_rank <= 0) and (int(os.environ.get("LOCAL_RANK", 0)) <= 0):
        import wandb

        wandb.init(
            project="emb-inv-1",
            name=exp_name,
            id=kwargs_hash,
            resume=True,
        )
        wandb.config.update(
            {**vars(model_args), **vars(data_args), **vars(training_args)},
            allow_val_change=True,
        )
    else:
        # Disable W&B
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"

    ###########################################################################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding=True,
        truncation='max_length',
        max_length=model_args.max_seq_length,
    )
    embedder, embedder_tokenizer = load_embedder_and_tokenizer(
        name=model_args.embedder_model_name
    )
    model = InversionModel(
        embedder=embedder,
        embedder_tokenizer=embedder_tokenizer,
        tokenizer=tokenizer,
        encoder_decoder=load_encoder_decoder(
            model_name=model_args.model_name_or_path
        ),
        num_repeat_tokens=model_args.num_repeat_tokens,
        embedder_no_grad=model_args.embedder_no_grad,
        embedder_fake_with_zeros=model_args.embedder_fake_with_zeros,
        use_frozen_embeddings_as_input=model_args.use_frozen_embeddings_as_input,
        use_embedding_batch_norm=model_args.use_embedding_batch_norm,
        encoder_dropout_disabled=model_args.encoder_dropout_disabled,
        decoder_dropout_disabled=model_args.decoder_dropout_disabled,
        freeze_strategy=model_args.freeze_strategy,
        token_decode_alpha=model_args.token_decode_alpha,
    )
    ###########################################################################

    logger.info("Loading datasets...")

    if data_args.dataset_name == "nq":
        raw_datasets = datasets.DatasetDict({
            "train": load_dpr_corpus(NQ_TRAIN),
            "validation": load_dpr_corpus(NQ_DEV),
        })
    elif data_args.dataset_name == "luar_reddit":
        all_luar_datasets = load_luar_reddit()
        raw_datasets = datasets.DatasetDict({
            "train": all_luar_datasets["candidates"],
            "validation": all_luar_datasets["queries"],
        })
    else:
        raise ValueError(f'unsupported dataset {data_args.dataset_name}')

    text_column_name = "text"
    column_names = list(raw_datasets["train"].features)
    ALLOWED_COLUMN_NAMES = { "frozen_embeddings" } # "document_id"}
    column_names = [c for c in column_names if c not in ALLOWED_COLUMN_NAMES]
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function(tokenizer, embedder_tokenizer, text_column_name, model_args.max_seq_length),
        batched=True,
        # num_proc=training_args.dataloader_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training model from checkpoint `{model_args.model_name_or_path}` - Total size={n_params/2**20:.2f}M params")
    
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    return InversionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=CustomCollator(tokenizer=tokenizer),
    )