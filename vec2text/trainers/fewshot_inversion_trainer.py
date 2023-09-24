from typing import Dict

import datasets
import torch
import transformers

from vec2text.trainers.base import BaseTrainer


def make_example_str_input_from_train_row(
        embedding: torch.Tensor, 
        embedder_tokenizer: transformers.PreTrainedTokenizer,
        k: int,
    ) -> str:
    topk_tokens = embedding[:embedder_tokenizer.vocab_size].topk(k=k)
    json_str = '{ '
    for tid, p in zip(topk_tokens.indices, topk_tokens.values):
        t = embedder_tokenizer.decode([tid]).encode()
        json_str += f'  {t}: {p:.4f}  '
    json_str += ' }'
    return f"""Top tokens: {json_str}
Output:"""

def make_example_str_from_train_row(
        input_ids: torch.Tensor, 
        embedding: torch.Tensor, 
        embedder_tokenizer: transformers.PreTrainedTokenizer,
        k: int,
    ) -> str:
    input_str = make_example_str_input_from_train_row(
        embedding=embedding,
        k=k,
        embedder_tokenizer=embedder_tokenizer
    )
    output = embedder_tokenizer.decode(input_ids, skip_special_tokens=True).strip().replace("\n", "\\n")
    return input_str + " " + output

class FewshotInversionTrainer(BaseTrainer):
    """This class is a mock 'trainer' that can be used to evaluate how good an LLM is (like GPT-4) at inversion."""

    train_dataset: datasets.Dataset
    num_tokens_per_example: int

    def __init__(self, *args, embedder_tokenizer: transformers.PreTrainedTokenizer, train_dataset: datasets.Dataset, num_tokens_per_example: int = 10, **kwargs):
        super().__init__(*args, model=torch.nn.Linear(1,1), model_init=None, **kwargs)
        self.num_tokens_per_example = num_tokens_per_example
        self.prompt_str = "Given the top-K predicted tokens and log-probabilities from a language model, please predict what the input was.\n\n"
        self.embedder_tokenizer = embedder_tokenizer
        for row in train_dataset:
            assert "frozen_embeddings" in row, f"need embedding for few shot - got keys {row.keys()}"
            self.prompt_str += make_example_str_from_train_row(
                input_ids=row["embedder_input_ids"],
                embedding=row["frozen_embeddings"], 
                embedder_tokenizer=self.embedder_tokenizer,
                k=self.num_tokens_per_example)
            self.prompt_str += "\n\n"

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        import pdb; pdb.set_trace()
        decoded_inputs = self.embedder_tokenizer.batch_decode(
            inputs["embedder_input_ids"], skip_special_tokens=True
        )
        # TODO: Test whether this is behaving properly for LLAMA chat.
        # May need special handling there.
        new_inputs = [d + self.prompt for d in decoded_inputs]
        new_inputs_tokenized = self.embedder_tokenizer(
            new_inputs, return_tensors="pt"
        ).to(self.device)
        generation_kwargs["max_length"] = self.max_length * 2
        generations = self.embedder.generate(
            **new_inputs_tokenized, **generation_kwargs
        )
        # pad away tokens that were in the original input
        is_new_tokens_mask = (
            torch.arange(generations.shape[1], device=self.args.device)[None]
            >= new_inputs_tokenized["attention_mask"].sum(1)[:, None]
        )
        generations = generations.where(
            is_new_tokens_mask, self.embedder_tokenizer.pad_token_id
        )
        # need to swap tokenizers
        bos_tokens = torch.tensor(
            [[self.decoder_start_token_id]] * len(new_inputs),
            dtype=torch.long,
            device=self.device,
        )
        untokenized_generations = self.embedder_tokenizer.batch_decode(
            generations, skip_special_tokens=True
        )
        retokenized_generations = self.tokenizer(
            untokenized_generations,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        retokenized_generations = torch.cat(
            [bos_tokens, retokenized_generations["input_ids"]], dim=1
        )
        return retokenized_generations

    def train(self):
        raise NotImplementedError

    def prediction_step(self, *args, **kwargs):
        return None, None, None
