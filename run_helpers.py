import hashlib
import functools
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
from tokenize_data import tokenize_function, whiten_embedded_dataset, embed_dataset_batch
from trainer import InversionTrainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    ###########################################################################
    model = inversion_model_from_args(model_args)
    train_dataset, eval_dataset = experiment.load_train_and_val_datasets()

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training model from checkpoint `{model_args.model_name_or_path}` - Total size={n_params/2**20:.2f}M params")

    return InversionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=CustomCollator(tokenizer=tokenizer),
    )