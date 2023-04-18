import transformers

from experiments import setup_experiment
from run_args import ModelArguments, DataTrainingArguments, TrainingArguments


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    experiment = setup_experiment(model_args, data_args, training_args)
    experiment.run()


if __name__ == "__main__":
    main()

