import os
import shlex
import tempfile

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

DEFAULT_ARGS_STR = "--per_device_train_batch_size 8 --max_seq_length 128 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --exp_name test-exp-123 --use_less_data 1000"
DEFAULT_ARGS = shlex.split(DEFAULT_ARGS_STR)

DEFAULT_ARGS += ["--use_wandb", "0"]
DEFAULT_ARGS += ["--bf16", "1"]


def load_trainer(model_args, data_args, training_args) -> InversionTrainer:
    ########################################################
    training_args.dataloader_num_workers = 0  # no multiprocessing :)
    training_args.num_train_epochs = 13.0
    training_args.eval_steps = 64
    training_args.group_by_length = True
    data_args.max_eval_samples = 64
    training_args.warmup_steps = 0
    trainer = experiment_from_args(
        model_args=model_args, data_args=data_args, training_args=training_args
    ).load_trainer()
    # make datasets smaller...
    trainer.train_dataset = trainer.train_dataset.select(range(256))  # just 8 batches
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
    trainer = load_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        TRAINER_STATE_NAME = "state.json"
        trainer.state.save_to_json(os.path.join(temp_dir, TRAINER_STATE_NAME))
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    assert metrics["train_loss"] > 0

    print("metrics:", metrics)


def test_trainer_openai():
    dataset_name = "msmarco"
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=DEFAULT_ARGS
    )
    model_args.embedder_model_api = "text-embedding-ada-002"
    model_args.use_frozen_embeddings_as_input = True
    data_args.dataset_name = dataset_name
    trainer = load_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        TRAINER_STATE_NAME = "state.json"
        trainer.state.save_to_json(os.path.join(temp_dir, TRAINER_STATE_NAME))
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    assert metrics["train_loss"] > 0

    print("metrics:", metrics)


def test_trainer_decoder():
    dataset_name = "nq"
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=DEFAULT_ARGS
    )
    model_args.model_name_or_path = "t5-base"
    model_args.use_frozen_embeddings_as_input = False
    data_args.dataset_name = dataset_name
    training_args.experiment = "inversion_decoder_only"
    trainer = load_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        TRAINER_STATE_NAME = "state.json"
        trainer.state.save_to_json(os.path.join(temp_dir, TRAINER_STATE_NAME))
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    assert metrics["train_loss"] > 0

    print("metrics:", metrics)


def test_trainer_gpt2():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=DEFAULT_ARGS
    )
    training_args.experiment = "inversion_from_logits"
    model_args.embedder_model_name = "gpt2"
    # model_args.embedder_model_name = "meta-llama/Llama-2-7b-hf"
    model_args.embedder_model_api = None
    model_args.model_name_or_path = "t5-small"
    model_args.use_frozen_embeddings_as_input = False  # too big (1.1 TB for 8M logits)
    data_args.dataset_name = "msmarco"
    trainer = load_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        TRAINER_STATE_NAME = "state.json"
        trainer.state.save_to_json(os.path.join(temp_dir, TRAINER_STATE_NAME))
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    assert metrics["train_loss"] > 0

    print("metrics:", metrics)


def test_trainer_gpt2_with_suffix():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=DEFAULT_ARGS
    )
    training_args.experiment = "inversion_from_logits"
    model_args.embedder_model_name = "gpt2"
    # model_args.embedder_model_name = "meta-llama/Llama-2-7b-hf"
    model_args.embedder_model_api = None
    model_args.model_name_or_path = "t5-small"
    model_args.use_frozen_embeddings_as_input = False  # too big (1.1 TB for 8M logits)
    data_args.dataset_name = "msmarco"
    model_args.suffix_conditioning = True
    trainer = load_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        TRAINER_STATE_NAME = "state.json"
        trainer.state.save_to_json(os.path.join(temp_dir, TRAINER_STATE_NAME))
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    assert metrics["train_loss"] > 0

    print("metrics:", metrics)


# def test_trainer_luar_data():
#     parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments)
#     )
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses(
#         args=DEFAULT_ARGS
#     )
#     data_args.dataset_name = "luar_reddit"
#     model_args.embedder_model_name = "paraphrase-distilroberta"
#     model_args.use_frozen_embeddings_as_input = True
#     trainer = load_trainer(
#         model_args=model_args, data_args=data_args, training_args=training_args
#     )
#     train_result = trainer.train(resume_from_checkpoint=None)
#     metrics = train_result.metrics
#     print("metrics:", metrics)
