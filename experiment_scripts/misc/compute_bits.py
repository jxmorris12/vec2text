# Load model directly

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).to("cuda")


import hashlib
import json
import os
import pickle
from typing import Dict, List

import openai
from tenacity import retry, stop_after_attempt, wait_fixed

os.environ["OPENAI_API_KEY"] = "😎"


class LLM_Chat:
    """Chat models take a different format: https://platform.openai.com/docs/guides/chat/introduction"""

    def __init__(self, checkpoint, seed, role, CACHE_DIR):
        self.cache_dir = os.path.join(
            CACHE_DIR, "cache_openai", f'{checkpoint.replace("/", "_")}___{seed}'
        )
        self.checkpoint = checkpoint
        self.role = role

    # @retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
    def __call__(
        self,
        prompts_list: List[Dict[str, str]],
        max_new_tokens=250,
        stop=None,
        functions: List[Dict] = None,
        return_str=True,
        verbose=False,
        temperature=0.0,
        frequency_penalty=0.0,
    ):
        """
        prompts_list: list of dicts, each dict has keys 'role' and 'content'
            Example: [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        prompts_list: str
            Alternatively, string which gets formatted into basic prompts_list:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": <<<<<prompts_list>>>>},
            ]
        """
        import openai

        if isinstance(prompts_list, str):
            role = self.role
            if role is None:
                role = "You are a helpful assistant."
            prompts_list = [
                {"role": "system", "content": role},
                {"role": "user", "content": prompts_list},
            ]

        assert isinstance(prompts_list, list), prompts_list

        # cache
        os.makedirs(self.cache_dir, exist_ok=True)
        prompts_list_dict = {
            str(i): sorted(v.items()) for i, v in enumerate(prompts_list)
        }
        if not self.checkpoint == "gpt-3.5-turbo":
            prompts_list_dict["checkpoint"] = self.checkpoint
        if functions is not None:
            prompts_list_dict["functions"] = functions
        if temperature > 0.1:
            prompts_list_dict["temperature"] = temperature
        dict_as_str = json.dumps(prompts_list_dict, sort_keys=True)
        hash_str = hashlib.sha256(dict_as_str.encode()).hexdigest()
        cache_file = os.path.join(
            self.cache_dir,
            f"chat__{hash_str}__num_tok={max_new_tokens}.pkl",
        )
        if os.path.exists(cache_file):
            if verbose:
                print("cached!")
                # print(cache_file)
            # print(cache_file)
            return pickle.load(open(cache_file, "rb"))
        if verbose:
            print("not cached")

        kwargs = dict(
            model=self.checkpoint,
            messages=prompts_list,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=frequency_penalty,  # maximum is 2
            presence_penalty=0,
            stop=stop,
            # stop=["101"]
        )
        if functions is not None:
            kwargs["functions"] = functions

        response = openai.ChatCompletion.create(**kwargs)

        if return_str:
            response = response["choices"][0]["message"]["content"]
        # print(response)

        pickle.dump(response, open(cache_file, "wb"))
        return response


gpt = LLM_Chat("gpt-4", 42, None, ".gpt_cache")


import datasets

text = datasets.load_dataset("wikitext", "wikitext-103-v1")["test"].filter(
    lambda ex: len(ex["text"].split()) > 100
)["text"]
text = [" ".join(t.split()) for t in text]

import random

random.choice(text)

first_ten_words = " ".join(text[0].split()[:10])
first_ten_words

prompt = """Please update the sentence by replacing one word sentence with a close synonym. Respond with only the word to swap in the format word1 -> word2.

{s}

Answer:"""

gpt(prompt.format(s=first_ten_words))

import tqdm


def update_sentence(s: str) -> str:
    first_ten_words = " ".join(s.split()[:10])
    r = gpt(prompt.format(s=first_ten_words))
    w1, w2 = r.split(" -> ")
    print("replacing", w1, "with", w2)
    new_first_ten_words = first_ten_words.replace(w1, w2)
    return s.replace(first_ten_words, new_first_ten_words)


new_text = [update_sentence(t) for t in tqdm.tqdm(text[:100])]

import torch


def div(d1: torch.Tensor, d2: torch.Tensor) -> float:
    return torch.nn.functional.kl_div(
        d1.log_softmax(0), d2.log_softmax(0), log_target=True
    ).item()


def get_dist_shift(text: str, new_text: str) -> torch.Tensor:
    t1 = tokenizer([text], padding=False, truncation=False, return_tensors="pt").to(
        "cuda"
    )
    t2 = tokenizer([new_text], padding=False, truncation=False, return_tensors="pt").to(
        "cuda"
    )

    with torch.no_grad():
        l1 = model(**t1).logits.squeeze(0)
        l2 = model(**t2).logits.squeeze(0)

    min_len = min(len(l1), len(l2))
    D = len(l2) - len(l1)
    return [div(l1[i], l2[i + D]) for i in range(min_len)]


shifts = []
for i in tqdm.trange(len(new_text)):
    shifts.append(get_dist_shift(text[i], new_text[i]))

import struct

import torch
from scipy.spatial import distance


def binary(num):
    return "".join("{:0>8b}".format(c) for c in struct.pack("!f", num))


def binary_diff(num1, num2) -> int:
    b1 = list(binary(num1))
    b2 = list(binary(num2))
    return distance.hamming(b1, b2) * len(b1)


import concurrent.futures


def bin_diff(d1: torch.Tensor, d2: torch.Tensor) -> float:
    # d1 = d1[:4]; d2=d2[:4];
    return sum(
        binary_diff(n1, n2)
        for n1, n2 in zip(d1.to(torch.float16), d2.to(torch.float16))
    )


def get_dist_shift(text: str, new_text: str) -> torch.Tensor:
    t1 = tokenizer([text], padding=False, truncation=False, return_tensors="pt").to(
        "cuda"
    )
    t2 = tokenizer([new_text], padding=False, truncation=False, return_tensors="pt").to(
        "cuda"
    )

    with torch.no_grad():
        l1 = model(**t1).logits.squeeze(0)
        l2 = model(**t2).logits.squeeze(0)

    min_len = min(len(l1), len(l2))
    D = len(l2) - len(l1)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(os.sched_getaffinity(0))
    ) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(
                    bin_diff,
                    [l1[i] for i in range(min_len)],
                    [l2[i + D] for i in range(min_len)],
                ),
                total=min_len,
                desc="binarizing",
                leave=False,
            )
        )
    return results
    # return [
    #     bin_diff(l1[i], l2[i+D]) for i in tqdm.trange(min_len, desc='binarizing', leave=False)
    # ]


shifts = []
for i in tqdm.trange(len(new_text)):
    shifts.append(get_dist_shift(text[i], new_text[i]))
pickle.dump(shifts, open("shifts_bits.p", "wb"))

print(f"wrote {len(new_text)} numbers to shifts_bits.p")
