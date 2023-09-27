import collections
import itertools
import os
import random

import tqdm

import vec2text

args = itertools.product(
    ["gpt-3.5-turbo-0613", "gpt-4-0613"],
    ["one_million_instructions", "python_code_alpaca", "anthropic_toxic_prompts"],
    [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-13b-chat-hf",
    ],
)

ArgsList = collections.namedtuple(
    "ArgsList",
    [
        "gpt",
        "dataset",
        "model",
    ],
)

BASE_CMD = """python experiment_scripts/logits/evaluate_baseline.py \
fewshot \
--gpt_version {gpt} --dataset {dataset} \
--embedder_model_name {model}
"""

all_args = list(itertools.product(args))
random.shuffle(all_args)


for args_list in tqdm.tqdm(all_args):
    A = ArgsList(*args_list[0])
    cmd = BASE_CMD.format(
        gpt=A.gpt,
        dataset=A.dataset,
        model=A.model,
    )
    print(cmd)
    os.system(cmd)
