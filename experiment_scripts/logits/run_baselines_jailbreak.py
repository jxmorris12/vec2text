import collections
import itertools
import os
import random

import tqdm

import vec2text

args = itertools.product(
    list(vec2text.prompts.JAILBREAK_PROMPTS.keys()),
    ["one_million_instructions", "python_code_alpaca", "anthropic_toxic_prompts"],
    [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-13b-chat-hf",
    ],
    [True, False],
)

ArgsList = collections.namedtuple(
    "ArgsList", ["prompt", "dataset", "model", "take_first_line"]
)

BASE_CMD = """python experiment_scripts/logits/evaluate_baseline.py \
jailbreak \
--prompt {prompt} --dataset {dataset} \
--embedder_model_name {model} --take_first_line {take_first_line}
"""

all_args = list(itertools.product(args))
random.shuffle(all_args)


for args_list in tqdm.tqdm(all_args):
    A = ArgsList(*args_list[0])
    cmd = BASE_CMD.format(
        prompt=A.prompt,
        dataset=A.dataset,
        model=A.model,
        take_first_line=A.take_first_line,
    )
    print(cmd)
    os.system(cmd)
