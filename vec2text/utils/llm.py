import json
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
import re
from transformers import LlamaForCausalLM, LlamaTokenizer

import openai

from typing import Any, Dict, List, Mapping, Optional
import numpy as np
import os.path
from os.path import join, dirname
import os
import pickle as pkl
from scipy.special import softmax
import hashlib
import torch
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
CACHE_DIR = "/home/jxm3/.cache/openai"
LLM_REPEAT_DELAY = 5  # how long to wait before recalling a failed llm call



def get_llm(checkpoint, seed=1, role: str = None):
    if checkpoint.startswith("text-"):
        return llm_openai(checkpoint, seed=seed)
    elif checkpoint.startswith("gpt-3") or checkpoint.startswith("gpt-4"):
        return llm_openai_chat(checkpoint, seed=seed, role=role)
    else:
        return llm_hf(
            checkpoint, seed=seed
        )  # warning: this sets torch.manual_seed(seed)

def llm_openai(checkpoint="text-davinci-003", seed=1):
    class LLM_OpenAI:
        def __init__(self, checkpoint, seed):
            self.cache_dir = join(
                CACHE_DIR, "cache_openai", f'{checkpoint.replace("/", "_")}___{seed}'
            )
            self.checkpoint = checkpoint

        @retry(wait=wait_fixed(LLM_REPEAT_DELAY), stop=stop_after_attempt(10))
        def __call__(self, prompt: str, max_new_tokens=250, do_sample=True, stop=None):
            # cache
            os.makedirs(self.cache_dir, exist_ok=True)
            id_str = (prompt + str(max_new_tokens) + (stop or "") + checkpoint)
            hash_str = hashlib.sha256(id_str.encode()).hexdigest()
            cache_file = join(
                self.cache_dir, f"{hash_str}__num_tok={max_new_tokens}.pkl"
            )
            if os.path.exists(cache_file):
                return pkl.load(open(cache_file, "rb"))

            response = openai.Completion.create(
                engine=self.checkpoint,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=0.1,
                top_p=1,
                frequency_penalty=0.25,  # maximum is 2
                presence_penalty=0,
                stop=stop,
                # stop=["101"]
            )
            response_text = response["choices"][0]["text"]

            pkl.dump(response_text, open(cache_file, "wb"))
            return response_text

    return LLM_OpenAI(checkpoint, seed)


def llm_openai_chat(checkpoint="gpt-3.5-turbo", seed=1, role=None):
    class LLM_Chat:
        """Chat models take a different format: https://platform.openai.com/docs/guides/chat/introduction"""

        def __init__(self, checkpoint, seed, role):
            self.cache_dir = join(
                CACHE_DIR, "cache_openai", f'{checkpoint.replace("/", "_")}___{seed}'
            )
            self.checkpoint = checkpoint
            self.role = role

        @retry(wait=wait_fixed(LLM_REPEAT_DELAY), stop=stop_after_attempt(10))
        def __call__(
            self,
            prompts_list: List[Dict[str, str]],
            max_new_tokens=250,
            stop=None,
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
            dict_as_str = json.dumps(prompts_list_dict, sort_keys=True)
            id_str = (dict_as_str + str(max_new_tokens) + (stop or "") + checkpoint)
            hash_str = hashlib.sha256(id_str.encode()).hexdigest()
            cache_file = join(
                self.cache_dir,
                f"chat__{hash_str}__num_tok={max_new_tokens}.pkl",
            )
            if os.path.exists(cache_file):
                print("cached!")
                return pkl.load(open(cache_file, "rb"))
            print("not cached")

            response = openai.ChatCompletion.create(
                model=self.checkpoint,
                messages=prompts_list,
                max_tokens=max_new_tokens,
                temperature=0.1,
                top_p=1,
                frequency_penalty=0.25,  # maximum is 2
                presence_penalty=0,
                stop=stop,
                # stop=["101"]
            )["choices"][0]["message"]["content"]

            pkl.dump(response, open(cache_file, "wb"))
            return response

    return LLM_Chat(checkpoint, seed, role)


def llm_hf(checkpoint="google/flan-t5-xl", seed=1):
    LLAMA_DIR = "/home/chansingh/llama"

    class LLM_HF:
        def __init__(self, checkpoint, seed):
            # set tokenizer
            if "facebook/opt" in checkpoint:
                # opt can't use fast tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint, use_fast=False
                )
            elif "llama_" in checkpoint:
                self._tokenizer = transformers.LlamaTokenizer.from_pretrained(
                    join(LLAMA_DIR, checkpoint)
                )
            elif "PMC_LLAMA" in checkpoint:
                self._tokenizer = transformers.LlamaTokenizer.from_pretrained(
                    "chaoyi-wu/PMC_LLAMA_7B"
                )
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint, use_fast=True
                )

            # set checkpoint
            if "google/flan" in checkpoint:
                self._model = T5ForConditionalGeneration.from_pretrained(
                    checkpoint, device_map="auto", torch_dtype=torch.float16
                )
            elif "llama_" in checkpoint:
                self._model = transformers.LlamaForCausalLM.from_pretrained(
                    join(LLAMA_DIR, checkpoint),
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            elif checkpoint == "gpt-xl":
                self._model = AutoModelForCausalLM.from_pretrained(checkpoint)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    checkpoint, device_map="auto", torch_dtype=torch.float16
                )
            self.checkpoint = checkpoint
            self.cache_dir = join(
                CACHE_DIR, "cache_hf", f'{checkpoint.replace("/", "_")}___{seed}'
            )
            self.seed = seed

        def __call__(
            self,
            prompt: str,
            stop: str = None,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
        ) -> str:
            """Warning: stop not actually used"""
            with torch.no_grad():
                # cache
                os.makedirs(self.cache_dir, exist_ok=True)
                hash_str = hashlib.sha256(prompt.encode()).hexdigest()
                cache_file = join(
                    self.cache_dir, f"{hash_str}__num_tok={max_new_tokens}.pkl"
                )
                if os.path.exists(cache_file) and use_cache:
                    return pkl.load(open(cache_file, "rb"))

                # if stop is not None:
                # raise ValueError("stop kwargs are not permitted.")
                inputs = self._tokenizer(
                    prompt, return_tensors="pt", return_attention_mask=True
                ).to(
                    self._model.device
                )  # .input_ids.to("cuda")
                # stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_tokens)])
                # outputs = self._model.generate(input_ids, max_length=max_tokens, stopping_criteria=stopping_criteria)
                # print('pad_token', self._tokenizer.pad_token)
                if self._tokenizer.pad_token_id is None:
                    self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
                    torch.manual_seed(0)
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    # pad_token=self._tokenizer.pad_token,
                    pad_token_id=self._tokenizer.pad_token_id,
                    # top_p=0.92,
                    # top_k=0
                )
                out_str = self._tokenizer.decode(outputs[0])
                # print('out_str', out_str)
                if "facebook/opt" in self.checkpoint:
                    out_str = out_str[len("</s>") + len(prompt) :]
                elif "google/flan" in self.checkpoint:
                    # print("full", out_str)
                    out_str = out_str[len("<pad>") : out_str.index("</s>")]
                elif "PMC_LLAMA" in self.checkpoint:
                    # print('here!', out_str)
                    out_str = out_str[len("<unk>") + len(prompt) :]
                elif "llama_" in self.checkpoint:
                    out_str = out_str[len("<s>") + len(prompt) :]
                else:
                    out_str = out_str[len(prompt) :]

                if stop is not None and isinstance(stop, str) and stop in out_str:
                    out_str = out_str[: out_str.index(stop)]

                pkl.dump(out_str, open(cache_file, "wb"))
                return out_str

        def _get_logit_for_target_token(
            self, prompt: str, target_token_str: str
        ) -> float:
            """Get logits target_token_str
            This is weird when token_output_ids represents multiple tokens
            It currently will only take the first token
            """
            # Get first token id in target_token_str
            target_token_id = self._tokenizer(target_token_str)["input_ids"][0]

            # get prob of target token
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True,
                padding=False,
                truncation=False,
            ).to(self._model.device)
            # shape is (batch_size, seq_len, vocab_size)
            logits = self._model(**inputs)["logits"].detach().cpu()
            # shape is (vocab_size,)
            probs_next_token = softmax(logits[0, -1, :].numpy().flatten())
            return probs_next_token[target_token_id]

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            """Get the identifying parameters."""
            return vars(self)

        @property
        def _llm_type(self) -> str:
            return "custom_hf_llm_for_langchain"

    return LLM_HF(checkpoint, seed)


if __name__ == "__main__":
    # llm = get_llm("text-davinci-003")
    # text = llm("What do these have in common? Horse, ")
    # print("text", text)

    # llm = get_llm("gpt2")
    # text = llm(
    # """Continue this list
    # - apple
    # - banana
    # -"""
    # )
    # print("text", text)
    # tokenizer = transformers.LlamaTokenizer.from_pretrained("chaoyi-wu/PMC_LLAMA_7B")
    # model = transformers.LlamaForCausalLM.from_pretrained("chaoyi-wu/PMC_LLAMA_7B")

    # llm = get_llm("chaoyi-wu/PMC_LLAMA_7B")
    llm = get_llm("llama_65b")
    text = llm(
        """Continue this list
- red
- orange
- yellow
- green
-""",
        use_cache=False,
    )
    print(text)
    print("\n\n")
    print(repr(text))