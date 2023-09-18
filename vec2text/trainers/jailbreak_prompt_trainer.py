from typing import Dict

import torch

from vec2text.trainers.base import BaseTrainer


class JailbreakPromptTrainer(BaseTrainer):
    """This class is a mock trainer that can be used to evaluate the usefulness of text prompts for inversion.
    """
    prompt: str
    def __init__(self, *args, prompt: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = prompt
    

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)


    def train(self):
        raise NotImplementedError