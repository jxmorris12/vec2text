from typing import Dict

import torch

from vec2text.trainers.base import BaseTrainer


class JailbreakPromptTrainer(BaseTrainer):
    """This class is a mock trainer that can be used to evaluate the usefulness of text prompts for inversion."""

    prompt: str

    def __init__(self, *args, prompt: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = prompt
        self.max_length = 128

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        if "frozen_embeddings" in inputs:
            del inputs["frozen_embeddings"]

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
