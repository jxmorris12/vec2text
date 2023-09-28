import functools
from typing import Dict, Iterable, List

import datasets
import torch
import transformers

from vec2text.trainers.base import BaseTrainer


class DecodeInversionTrainer(BaseTrainer):
    """This 'trainer' represents a baseline for logits inversion that decodes from
    the language model, then tries to predict (sequence-to-sequence) what the
    prompt was, given only the decoded output.
    """

    language_model: transformers.PreTrainedModel
    inverter: transformers.PreTrainedModel

    def __init__(
        self,
        *args,
        language_model: transformers.PreTrainedModel,
        inverter: transformers.PreTrainedModel,
        **kwargs,
    ):
        super().__init__(*args, model=torch.nn.Linear(1, 1), model_init=None, **kwargs)
        self.language_model = language_model
        self.inverter = inverter
        self.max_length = 64

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        lm_outputs = self.language_model.generate(
            input_ids=inputs["embedder_input_ids"],
            attention_mask=inputs["embedder_attention_mask"],
            do_sample=False,
            max_new_tokens=self.max_length,
        )

        lm_outputs = self.lm_tokenizer(
            self.lm_tokenizer.decode_batch(
                lm_outputs,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True
            ),
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
        ).to(self.args.device)

        return self.inverter.generate(
            **lm_outputs,
            self.gen_kwargs
        )

    def train(self):
        raise NotImplementedError

    def prediction_step(self, *args, **kwargs):
        return None, None, None
