from typing import Dict

import torch

from vec2text.trainers.base import BaseTrainer


class JailbreakPromptTrainer(BaseTrainer):
    """This class is a mock trainer that can be used to evaluate the usefulness of text prompts for inversion."""

    prompt: str

    def __init__(self, *args, prompt: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = prompt

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        del inputs["frozen_embeddings"]

        decoded_inputs = self.embedder_tokenizer.batch_decode(
            inputs["embedder_input_ids"], skip_special_tokens=True
        )
        new_inputs = [d + self.prompt for d in decoded_inputs]
        new_inputs_tokenized = self.embedder_tokenizer(
            new_inputs, return_tensors="pt"
        ).to(self.device)
        inputs["embedder_input_ids"] = new_inputs_tokenized["input_ids"]
        inputs["embedder_attention_mask"] = new_inputs_tokenized["attention_mask"]

        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def train(self):
        raise NotImplementedError

    def prediction_step(self, *args, **kwargs):
        return None, None, None
