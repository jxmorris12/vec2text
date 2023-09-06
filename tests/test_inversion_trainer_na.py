import shlex

import pytest
import transformers

from vec2text.experiments import experiment_from_args
from vec2text.run_args import (
    DATASET_NAMES,
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from vec2text.trainers import InversionTrainer

DEFAULT_ARGS_STR = "--per_device_train_batch_size 32 --max_seq_length 128 --model_name_or_path t5-small --embedder_model_name gtr_base --num_repeat_tokens 32 --exp_name test-exp-123"
DEFAULT_ARGS = shlex.split(DEFAULT_ARGS_STR)

DEFAULT_ARGS += ["--use_wandb", "0"]
DEFAULT_ARGS += ["--fp16", "1"]


def load_trainer(model_args, data_args, training_args) -> InversionTrainer:
    ########################################################
    training_args.num_train_epochs = 1.0
    training_args.eval_steps = 4
    data_args.max_eval_samples = 64
    trainer = experiment_from_args(
        model_args=model_args, data_args=data_args, training_args=training_args
    ).load_trainer()
    # make datasets smaller...
    trainer.train_dataset = trainer.train_dataset.select(range(256))
    ########################################################
    return trainer


@pytest.mark.parametrize("dataset_name", DATASET_NAMES)
def test_trainer(dataset_name):
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=DEFAULT_ARGS
    )
    data_args.dataset_name = dataset_name
    data_args.use_less_data = True
    training_args.experiment = "inversion_na"
    trainer = load_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    print("metrics:", metrics)


def test_trainer_luar_data():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=DEFAULT_ARGS
    )
    data_args.dataset_name = "luar_reddit"
    model_args.embedder_model_name = "paraphrase-distilroberta"
    model_args.use_frozen_embeddings_as_input = True
    trainer = load_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    print("metrics:", metrics)
