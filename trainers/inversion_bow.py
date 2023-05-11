import math
from typing import Dict

import torch
import transformers

from trainers.base import BaseTrainer


class InversionTrainerBagOfWords(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ######################################################
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model
    
    def compute_metrics_func(self, eval_preds):
        return {} # TODO: fix/implement metrics.

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

