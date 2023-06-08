import os
import shlex
from typing import Optional

import torch
import transformers
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

import experiments
from run_args import DataArguments, ModelArguments, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()

#############################################################################


def load_trainer(
    checkpoint_folder: str,
    args_str: Optional[str] = None,
    checkpoint: Optional[str] = None,
    do_eval: bool = True,
    sanity_decode: bool = True,
):  # (can't import due to circluar import) -> trainers.InversionTrainer:
    if checkpoint is None:
        checkpoint = get_last_checkpoint(checkpoint_folder)  # a checkpoint
    if args_str is not None:
        args = shlex.split(args_str)
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            args=args
        )
    else:
        data_args = torch.load(os.path.join(checkpoint, os.pardir, "data_args.bin"))
        model_args = torch.load(os.path.join(checkpoint, os.pardir, "model_args.bin"))

    training_args = torch.load(os.path.join(checkpoint, "training_args.bin"))
    ########################################################################
    from accelerate.state import PartialState

    training_args._n_gpu = 1  # Don't load in DDP
    training_args.local_rank = -1  # Don't load in DDP
    training_args.distributed_state = PartialState()
    ########################################################################
    if do_eval:
        print("Loading trainer for analysis – setting --do_eval=1")
        training_args.do_eval = do_eval
        training_args.dataloader_num_workers = 0  # no multiprocessing :)
    training_args.use_wandb = False
    training_args.report_to = []
    experiment = experiments.experiment_from_args(model_args, data_args, training_args)
    trainer = experiment.load_trainer()
    trainer.model._keys_to_ignore_on_save = []
    trainer._load_from_checkpoint(checkpoint)
    if sanity_decode:
        trainer.sanity_decode()
    return trainer
