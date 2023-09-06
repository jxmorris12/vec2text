import glob
import json
import os
import shlex
from typing import Optional

import pandas as pd
import torch
import transformers
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from vec2text import experiments
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()

#############################################################################


def load_experiment_and_trainer(
    checkpoint_folder: str,
    args_str: Optional[str] = None,
    checkpoint: Optional[str] = None,
    do_eval: bool = True,
    sanity_decode: bool = True,
    max_seq_length: Optional[int] = None,
    use_less_data: Optional[int] = None,
):  # (can't import due to circluar import) -> trainers.InversionTrainer:
    # import previous aliases so that .bin that were saved prior to the
    # existence of the vec2text module will still work.
    import sys

    import vec2text.run_args as run_args

    sys.modules["run_args"] = run_args

    print("run_args:", run_args)

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

    if max_seq_length is not None:
        print(
            f"Overwriting max sequence length from {model_args.max_seq_length} to {max_seq_length}"
        )
        model_args.max_seq_length = max_seq_length

    if use_less_data is not None:
        print(
            f"Overwriting use_less_data from {data_args.use_less_data} to {use_less_data}"
        )
        data_args.use_less_data = use_less_data

    # For batch decoding outputs during evaluation.
    # os.environ["TOKENIZERS_PARALLELISM"] = "True"

    ########################################################################
    print("> checkpoint:", checkpoint)
    if (
        checkpoint
        == "/home/jxm3/research/retrieval/inversion/saves/47d9c149a8e827d0609abbeefdfd89ac/checkpoint-558000"
    ):
        # Special handling for one case of backwards compatibility:
        #   set dataset (which used to be empty) to nq
        data_args.dataset_name = "nq"
        print("set dataset to nq")

    training_args = torch.load(os.path.join(checkpoint, "training_args.bin"))
    ########################################################################
    from accelerate.state import PartialState

    training_args._n_gpu = 1  # Don't load in DDP
    training_args.local_rank = -1  # Don't load in DDP
    training_args.distributed_state = PartialState()
    training_args.deepspeed_plugin = None  # For backwards compatibility
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
    try:
        trainer._load_from_checkpoint(checkpoint)
    except RuntimeError:
        # backwards compatibility from adding/removing layernorm
        trainer.model.use_ln = False
        trainer.model.layernorm = None
        # try again without trying to load layernorm
        trainer._load_from_checkpoint(checkpoint)
    if sanity_decode:
        trainer.sanity_decode()
    return experiment, trainer


def load_trainer(
    *args, **kwargs
):  # (can't import due to circluar import) -> trainers.Inversion
    experiment, trainer = load_experiment_and_trainer(*args, **kwargs)
    return trainer


def load_results_from_folder(name: str) -> pd.DataFrame:
    filenames = glob.glob(os.path.join(name, "*.json"))
    data = []
    for f in filenames:
        d = json.load(open(f, "r"))
        if "_eval_args" in d:
            # unnest args for evaluation
            d.update(d.pop("_eval_args"))
        data.append(d)
    return pd.DataFrame(data)
