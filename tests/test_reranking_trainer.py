import shlex

import transformers

import experiments
from run_args import DataArguments, ModelArguments, TrainingArguments
from trainers import RerankingTrainer

DEFAULT_ARGS_STR = "--per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 32 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --exp_group_name mar17-baselines --learning_rate 0.0003 --freeze_strategy none --embedder_fake_with_zeros False --use_frozen_embeddings_as_input False --num_train_epochs 24 --max_eval_samples 500 --eval_steps 25000 --warmup_steps 100000 --bf16=1 --use_wandb=0"
DEFAULT_ARGS = shlex.split(DEFAULT_ARGS_STR)

DEFAULT_ARGS += ["--use_wandb", "0"]
DEFAULT_ARGS += ["--bf16", "1"]


def load_trainer(model_args, data_args, training_args) -> RerankingTrainer:
    ########################################################
    training_args.num_train_epochs = 1.0
    training_args.eval_steps = 4
    trainer = experiments.experiment_from_args(
        model_args=model_args, data_args=data_args, training_args=training_args
    ).load_trainer()
    # make datasets smaller...
    trainer.train_dataset = trainer.train_dataset.select(range(256))
    trainer.eval_dataset = trainer.eval_dataset.select(range(64))
    ########################################################
    return trainer


def test_trainer():
    dataset_name = "nq"
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=DEFAULT_ARGS
    )
    training_args.experiment = "reranking"
    model_args.max_seq_length = 32
    data_args.dataset_name = dataset_name
    trainer = load_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    print("metrics:", metrics)
