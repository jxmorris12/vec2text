# type: ignore

import argparse
import hashlib
import json
import os
from pprint import pprint

import vec2text


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Argument Parser")

    parser.add_argument("alias", type=str, help="baseline name")
    parser.add_argument(
        "--dataset",
        type=str,
        default="python_code_alpaca",
        help="Dataset (if not regular val)",
        choices=[
            "one_million_instructions",
            "python_code_alpaca",
            "anthropic_toxic_prompts",
        ],
    )
    parser.add_argument(
        "--num_samples", type=int, default=200, help="Number of evaluation samples"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for 7B model"
    )
    return parser


def md5_hash_kwargs(**kwargs) -> str:
    # We ignore special hf args that start with _ like '__cached__setup_devices'.
    safe_kwargs = {k: str(v) for k, v in kwargs.items() if not k.startswith("_")}
    s = json.dumps(safe_kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


def main(args: argparse.ArgumentParser):
    out_file = os.path.join(
        "/home/jxm3/research/retrieval/inversion/results_evaluation/logits",
        md5_hash_kwargs(**vars(args)) + ".json",
    )
    if os.path.exists(out_file):
        print("file exists:", out_file)
        print("args were:", vars(args))
        print("exiting early :-)")
        exit()

    if os.path.exists(args.alias):
        (
            experiment,
            trainer,
        ) = vec2text.analyze_utils.load_experiment_and_trainer(args.alias)
    else:
        (
            experiment,
            trainer,
        ) = vec2text.analyze_utils.load_experiment_and_trainer_from_pretrained(
            args.alias
        )
    embedder_name = experiment.model_args.embedder_model_name
    assert "7b" in embedder_name

    # This code assumes models were trained on 7b param embedders.
    # It also assumes they have the same tokenizer...
    for embedder_size in ["7b", "13b", "70b"]:  # , "13b", "7b"]:
        #
        trainer.enable_emb_cos_sim_metric()
        trainer.model.use_frozen_embeddings_as_input = False

        this_embedder_name = embedder_name.replace("7b", embedder_size)
        trainer.args.per_device_eval_batch_size = (
            (args.batch_size or 32) if "7b" in this_embedder_name else 1
        )
        args.embedder_model_name = this_embedder_name

        cwd = os.path.dirname(os.path.abspath(__file__))
        out_file = os.path.normpath(
            os.path.join(
                cwd,
                os.pardir,
                os.pardir,
                "results",
                "results_evaluation",
                "logits",
                md5_hash_kwargs(**vars(args)) + ".json",
            )
        )
        if os.path.exists(out_file):
            print("file exists:", out_file)
            continue

        trainer.model.cpu()
        trainer.args.bf16_full_eval = True
        print(
            "\tloading embedder for eval:",
            this_embedder_name,
        )
        trainer.model.embedder = vec2text.models.load_embedder_and_tokenizer(
            this_embedder_name, torch_dtype=trainer.model.config.embedder_torch_dtype
        )[0]
        trainer.model.to(trainer.args.device)

        # import datasets
        ## val
        # trainer.eval_dataset = datasets.load_from_disk("/home/wentingz/.cache/inversion/915245880f7c6efb6fd89ee78d3d91d1.arrow")
        ## train
        # td = datasets.load_from_disk("/home/wentingz/.cache/inversion/a92ab1949d136d6c63fde466c2bdadac.arrow")
        # trainer.eval_dataset["one_million_instructions"] = td["validation"]

        eval_dataset = trainer.eval_dataset[args.dataset].remove_columns(
            "frozen_embeddings"
        )
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
