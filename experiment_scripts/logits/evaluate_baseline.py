# type: ignore

import argparse
import hashlib
import json
import os
from pprint import pprint

from datasets import disable_caching

import vec2text

disable_caching()
print("** DISABLED HF DATASETS CACHING **")


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Argument Parser")

    parser.add_argument(
        "alias", type=str, help="baseline name", choices=["jailbreak", "fewshot"]
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=vec2text.prompts.JAILBREAK_PROMPTS.keys(),
        help="key for prompt to use:",
        default="00_output_simple",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=64, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_samples", type=int, default=200, help="Number of evaluation samples"
    )
    parser.add_argument(
        "--embedder_model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="model to invert",
    )
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
        "--gpt_version",
        type=str,
        default="gpt-3.5-turbo-0613",
        help="gpt version for api calls (for fewshot only)",
    )

    parser.add_argument(
        "--take_first_line",
        type=bool,
        default=False,
        help="whether to only take first line of model response (for jailbreak strings only)",
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

    if args.alias == "jailbreak":
        trainer = vec2text.analyze_utils.load_jailbreak_baseline_trainer(
            prompt=vec2text.prompts.JAILBREAK_PROMPTS[args.prompt],
            embedder_model_name=args.embedder_model_name,
            max_seq_len=args.max_seq_length,
        )
        trainer.take_first_line = args.take_first_line
    elif args.alias == "fewshot":
        trainer = vec2text.analyze_utils.load_gpt_fewshot_baseline_trainer(
            embedder_model_name=args.embedder_model_name,
            num_few_shot_examples=3,
            num_tokens_per_example=100,
        )
        trainer._gpt_version = args.gpt_version
    else:
        raise ValueError(f"unknown alias {args.alias}")
    trainer.args.per_device_eval_batch_size = (
        4 if "13b" in args.embedder_model_name else 32
    )
    trainer.enable_emb_cos_sim_metric()

    eval_dataset = trainer.eval_dataset[args.dataset]
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
