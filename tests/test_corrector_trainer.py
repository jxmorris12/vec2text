import shlex

import transformers

from vec2text import experiments
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments
from vec2text.trainers import Corrector

DEFAULT_ARGS_STR = "--per_device_train_batch_size 16 --per_device_eval_batch_size 16 --max_seq_length 32 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --exp_group_name mar17-baselines --learning_rate 0.0003 --freeze_strategy none --embedder_fake_with_zeros False --use_frozen_embeddings_as_input False --num_train_epochs 24 --eval_steps 25000 --warmup_steps 100000 --bf16=1 --use_wandb=0"
DEFAULT_ARGS = shlex.split(DEFAULT_ARGS_STR)

DEFAULT_ARGS += ["--use_wandb", "0"]
DEFAULT_ARGS += ["--bf16", "1"]


def load_trainer(model_args, data_args, training_args) -> Corrector:
    ########################################################
    training_args.num_train_epochs = 2.0
    training_args.eval_steps = 6400000  # 64
    training_args.use_less_data = 1000
    data_args.max_eval_samples = 64
    training_args.cheat_on_train_hypotheses = True
    trainer = experiments.experiment_from_args(
        model_args=model_args, data_args=data_args, training_args=training_args
    ).load_trainer()
    # make datasets smaller...
    trainer.train_dataset = trainer.train_dataset.select(range(128))
    ########################################################
    return trainer


def test_trainer():
    dataset_name = "msmarco"
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=DEFAULT_ARGS
    )
    # TODO parameterize tests with experiment var
    # training_args.experiment = "corrector"
    training_args.experiment = "corrector_encoder"
    model_args.max_seq_length = 32
    data_args.dataset_name = dataset_name
    trainer = load_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )
    train_result = trainer.train(resume_from_checkpoint=None)
    train_metrics = train_result.metrics
    print("train metrics:", train_metrics)

    for eval_dataset_name, eval_dataset in trainer.eval_dataset.items():
        max_len = min(500, len(eval_dataset))
        eval_metrics = trainer.evaluate(
            eval_dataset=eval_dataset.select(range(max_len)),
            ignore_keys=None,
            metric_key_prefix=f"eval_{eval_dataset_name}",
        )
        print("eval metrics:", eval_metrics)
