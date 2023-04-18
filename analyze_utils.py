from typing import Optional

import functools
import os
import shlex

import datasets
import torch
import transformers
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint

from collator import CustomCollator
from data_helpers import load_dpr_corpus, NQ_DEV, NQ_TRAIN
from experiments import experiment_from_args
from models import load_encoder_decoder, load_embedder_and_tokenizer, InversionModel
from run_args import ModelArguments, DataTrainingArguments, TrainingArguments
from tokenize_data import embed_dataset_batch, tokenize_function, whiten_embedded_dataset
from trainers import InversionTrainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()

#############################################################################

def load_trainer(
        checkpoint_folder: str,
        args_str: str,
        checkpoint: Optional[str] = None
    ) -> InversionTrainer:
    print("Setting --do_eval=1")
    args_str += " --do_eval 1"
    args = shlex.split(args_str)
    if checkpoint is None:
        checkpoint = get_last_checkpoint(checkpoint_folder) # a checkpoint
    print("[0] Loading model from checkpoint:", checkpoint)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args)

    training_args.dataloader_num_workers = 0 # no multiprocessing :)
    training_args = torch.load(os.path.join(checkpoint, 'training_args.bin'))
    training_args.use_wandb = False
    training_args.report_to = []
    
    experiment = experiment_from_args(model_args, data_args, training_args)
    trainer = experiment.get_trainer()
    trainer._load_from_checkpoint(checkpoint)
    return trainer

