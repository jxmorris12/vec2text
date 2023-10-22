from typing import Dict

import torch
import transformers

from vec2text.trainers.base import BaseTrainer


class DecodeInversionTrainer(BaseTrainer):
    """This 'trainer' represents a baseline for logits inversion that decodes from
    the language model, then tries to predict (sequence-to-sequence) what the
    prompt was, given only the decoded output.
    """

    language_model: transformers.PreTrainedModel
    language_model_tokenizer: transformers.PreTrainedTokenizer
    inverter: transformers.PreTrainedModel

    def __init__(
        self,
        *args,
        language_model: transformers.PreTrainedModel,
        language_model_tokenizer: transformers.PreTrainedTokenizer,
        inverter: transformers.PreTrainedModel,
        **kwargs,
    ):
        super().__init__(*args, model=torch.nn.Linear(1, 1), model_init=None, **kwargs)
        self.language_model = language_model
        self.language_model_tokenizer = language_model_tokenizer
        self.inverter = inverter
        self.inverter = self.inverter.to(self.args.device)
        self.max_length = 64

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        self.embedder_tokenizer.padding_side = "left"
        lm_inputs = self.embedder_tokenizer(
            self.embedder_tokenizer.batch_decode(
                inputs["embedder_input_ids"],
                skip_special_tokens=True,
            ),
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True,
        ).to(self.args.device)

        full_lm_outputs = self.language_model.generate(
            input_ids=lm_inputs.input_ids,
            attention_mask=lm_inputs.attention_mask,
            do_sample=False,
            max_new_tokens=self.max_length,
        )
        lm_outputs = full_lm_outputs[:, lm_inputs.input_ids.shape[1] :]
        # bos_tokens = torch.ones((lm_outputs.shape[0], 1), dtype=torch.long, device=self.args.device)
        # lm_outputs = torch.cat((bos_tokens, lm_outputs), dim=1)
        lm_outputs_str = self.language_model_tokenizer.batch_decode(
            lm_outputs, skip_special_tokens=True
        )

        lm_outputs_for_inverter = self.tokenizer(
            lm_outputs_str,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        ).to(self.args.device)

        gen_kwargs = self.gen_kwargs
        gen_kwargs["min_new_tokens"] = 1
        gen_kwargs["max_new_tokens"] = self.max_length
        return self.inverter.generate(
            **lm_outputs_for_inverter,
            min_new_tokens=1,
            max_new_tokens=64,
            generation_kwargs=gen_kwargs,
        )

    def train(self):
        raise NotImplementedError

    def prediction_step(self, *args, **kwargs):
        return None, None, None
