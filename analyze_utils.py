import os
import shlex
from typing import Optional

import torch
import transformers
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from experiments import experiment_from_args
from run_args import DataArguments, ModelArguments, TrainingArguments
from trainers import InversionTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()

#############################################################################


def load_trainer(
    checkpoint_folder: str,
    args_str: str,
    checkpoint: Optional[str] = None,
    do_eval: bool = True,
) -> InversionTrainer:
    print("Loading trainer for analysis – setting --do_eval=1")
    if checkpoint is None:
        checkpoint = get_last_checkpoint(checkpoint_folder)  # a checkpoint
    print("[0] Loading model from checkpoint:", checkpoint)
    args = shlex.split(args_str)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args)
    training_args.do_eval = do_eval

    training_args.dataloader_num_workers = 0  # no multiprocessing :)
    training_args = torch.load(os.path.join(checkpoint, "training_args.bin"))
    training_args.use_wandb = False
    training_args.report_to = []

    experiment = experiment_from_args(model_args, data_args, training_args)
    trainer = experiment.load_trainer()
    trainer._load_from_checkpoint(checkpoint)
    return trainer
