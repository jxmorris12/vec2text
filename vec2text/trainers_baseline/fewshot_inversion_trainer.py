import functools
from typing import Dict, Iterable, List

import datasets
import torch
import transformers
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from vec2text.trainers.base import BaseTrainer

client = OpenAI()


@retry(wait=wait_fixed(5), stop=stop_after_attempt(10))
def call_openai_llm(
    prompt: str,
    gpt_version: str,
) -> str:
    full_prompts = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return client.chat.completions.create(
        model=gpt_version,
        messages=full_prompts,
        max_tokens=64,
        temperature=0.0,
        # stop=["\n"],
        presence_penalty=0,
    )["choices"][0]["message"]["content"]


def make_example_str_input_from_train_row(
    embedding: torch.Tensor,
    embedder_tokenizer: transformers.PreTrainedTokenizer,
    k: int,
) -> str:
    topk_tokens = embedding[: embedder_tokenizer.vocab_size].topk(k=k)
    json_str = "{ "
    for tid, p in zip(topk_tokens.indices, topk_tokens.values):
        t = embedder_tokenizer.decode([tid])
        json_str += f"  {t}: {p:.4f}  "
    json_str += " }"
    return f"""Top tokens: {json_str}
Output:"""


def make_example_str_from_train_row(
    input_ids: torch.Tensor,
    embedding: torch.Tensor,
    embedder_tokenizer: transformers.PreTrainedTokenizer,
    k: int,
) -> str:
    input_str = make_example_str_input_from_train_row(
        embedding=embedding, k=k, embedder_tokenizer=embedder_tokenizer
    )
    output = (
        embedder_tokenizer.decode(input_ids, skip_special_tokens=True).strip()
        # .replace("\n", "\\n")
    )
    return input_str + " " + output


class FewshotInversionTrainer(BaseTrainer):
    """This class is a mock 'trainer' that can be used to evaluate how good an LLM is (like GPT-4) at inversion."""

    train_dataset: datasets.Dataset
    num_tokens_per_example: int
    num_few_shot_examples: int
    prompt_header: str = "Given the top-K predicted tokens and log-probabilities from a language model, please predict what the input was. Please follow the examples and don't output anything except the predicted input.\n\n"

    def __init__(
        self,
        *args,
        embedder_tokenizer: transformers.PreTrainedTokenizer,
        train_dataset: datasets.Dataset,
        num_tokens_per_example: int = 10,
        num_few_shot_examples: int = 3,
        **kwargs,
    ):
        super().__init__(*args, model=torch.nn.Linear(1, 1), model_init=None, **kwargs)
        self.num_tokens_per_example = num_tokens_per_example
        self.embedder_tokenizer = embedder_tokenizer
        self.prompt_str = self.prompt_header
        self.num_few_shot_examples = num_few_shot_examples

        self.unigram_embedding = train_dataset["frozen_embeddings"].mean(dim=0)
        for row in train_dataset.select(range(self.num_few_shot_examples)):
            assert (
                "frozen_embeddings" in row
            ), f"need embedding for few shot - got keys {row.keys()}"
            self.prompt_str += make_example_str_from_train_row(
                input_ids=row["embedder_input_ids"],
                embedding=row["frozen_embeddings"] - self.unigram_embedding,
                embedder_tokenizer=self.embedder_tokenizer,
                k=self.num_tokens_per_example,
            )
            self.prompt_str += "\n\n"
        self._gpt_version = "gpt-3.5-turbo"

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        if "frozen_embeddings" in inputs:
            embeddings = inputs["frozen_embeddings"]
            assert len(embeddings.shape) == 2
        else:
            with torch.no_grad():
                embeddings = self.call_embedding_model(
                    input_ids=inputs["embedder_input_ids"],
                    attention_mask=inputs["embedder_attention_mask"],
                )
                embeddings = embeddings - self.unigram_embedding[None, :].to(
                    embeddings.device
                )
        prompt_suffixes = list(
            map(
                functools.partial(
                    make_example_str_input_from_train_row,
                    embedder_tokenizer=self.embedder_tokenizer,
                    k=self.num_tokens_per_example,
                ),
                embeddings.cpu(),
            )
        )
        full_prompts = [self.prompt_str + s for s in prompt_suffixes]
        # print(full_prompts[0])
        response_text = list(self._call_gpt(full_prompts))
        return self.tokenizer(
            response_text, return_tensors="pt", padding="max_length", truncation=False
        ).input_ids.to(inputs["embedder_input_ids"].device)

    def _call_gpt(self, prompts: List[str]) -> Iterable[str]:
        # TODO implement caching...
        for p in prompts:
            yield call_openai_llm(
                prompt=p,
                gpt_version=self._gpt_version,
            )

    def train(self):
        raise NotImplementedError

    def prediction_step(self, *args, **kwargs):
        return None, None, None
