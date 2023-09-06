# type: ignore

import argparse
import hashlib
import json
import os
from pprint import pprint

import aliases
from data_helpers import load_beir_datasets


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Argument Parser")

    parser.add_argument("alias", type=str, help="Trained model alias from alias.py")
    parser.add_argument(
        "--num_samples", type=int, default=200, help="Number of evaluation samples"
    )
    parser.add_argument(
        "--return_best_hypothesis",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to return best hypothesis during generation",
    )
    parser.add_argument(
        "--num_gen_recursive_steps",
        type=int,
        default=1,
        help="Number of steps for recursive generation",
    )
    parser.add_argument(
        "--sequence_beam_width", type=int, default=1, help="Sequence-level beam width"
    )
    parser.add_argument("--beam_width", type=int, default=1, help="Regular beam width")
    parser.add_argument(
        "--dataset", type=str, default=None, help="Dataset (if not regular val)"
    )

    return parser


def md5_hash_kwargs(**kwargs) -> str:
    # We ignore special hf args that start with _ like '__cached__setup_devices'.
    safe_kwargs = {k: str(v) for k, v in kwargs.items() if not k.startswith("_")}
    s = json.dumps(safe_kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


def main(args: argparse.ArgumentParser):
    out_file = os.path.join(
        "/home/jxm3/research/retrieval/inversion/results_evaluation",
        md5_hash_kwargs(**vars(args)) + ".json",
    )
    if os.path.exists(out_file):
        print("file exists:", out_file)
        print("args were:", vars(args))
        print("exiting early :-)")
        exit()

    print("return_best_hypothesis:", args.return_best_hypothesis)
    experiment, trainer = aliases.load_experiment_and_trainer_from_alias(args.alias)
    trainer.model.eval()
    trainer.args.per_device_eval_batch_size = int(
        8 / max(args.beam_width, args.sequence_beam_width)
    )
    trainer.return_best_hypothesis = bool(args.return_best_hypothesis)
    trainer.num_gen_recursive_steps = args.num_gen_recursive_steps
    trainer.sequence_beam_width = args.sequence_beam_width
    trainer.gen_kwargs = {
        "early_stopping": False,
        "num_beams": args.beam_width,
        "num_return_sequences": args.beam_width,
        "do_sample": False,
        "no_repeat_ngram_size": 0,
    }

    if args.dataset:
        # load dataset
        beir = load_beir_datasets()
        if hasattr(trainer, "inversion_trainer"):
            model = trainer.inversion_trainer.model
        else:
            model = trainer.model

        beir = experiment._prepare_val_datasets_dict(
            model=model,
            val_datasets_dict=beir,
            tokenizer=trainer.tokenizer,
            embedder_tokenizer=trainer.embedder_tokenizer,
        )
        eval_dataset = beir[args.dataset]
    else:
        eval_dataset = trainer.eval_dataset[experiment.data_args.dataset_name]
    metrics = trainer.evaluate(
        eval_dataset=eval_dataset.select(range(args.num_samples))
    )
    metrics["_eval_args"] = vars(args)
    with open(out_file, "w") as f:
        json.dump(metrics, f)

    pprint(metrics)
    print("wrote metrics to", out_file)


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    main(args=args)
