from typing import Dict

import torch
import transformers

from models import InversionModel


class ContrastiveLogitsProcessor(transformers.LogitsProcessor):
    model: InversionModel
    alpha: float
    gamma: float
    hypothesis_embedding: torch.Tensor

    def __init__(
        self,
        model: InversionModel,
        alpha: float,
        gamma: float,
        hypothesis_temperature: float,
        hypothesis_num_samples: int,
        inputs: Dict[str, torch.Tensor],
    ):
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.hypothesis_temperature = hypothesis_temperature  # 1e-9
        self.hypothesis_num_samples = hypothesis_num_samples
        self.hypothesis_embedding = self._get_hypothesis_embedding(inputs=inputs)
        # print(f"ContrastiveLogitsProcessor alpha={alpha} gamma={gamma}")

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _get_hypothesis_embedding(
        self, inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        batch_size, seq_length = inputs["embedder_input_ids"].shape

        if "decoder_input_ids" in inputs:
            inputs.pop("decoder_input_ids")
        if "decoder_attention_mask" in inputs:
            inputs.pop("decoder_attention_mask")

        do_sample = self.hypothesis_temperature > 0
        hypotheses = self.model.generate(
            inputs=inputs,
            generation_kwargs={
                "max_length": seq_length,
                "early_stopping": True,
                "num_beams": (self.hypothesis_num_samples if not do_sample else 1),
                "do_sample": do_sample,
                "temperature": self.hypothesis_temperature,
                "num_return_sequences": self.hypothesis_num_samples,
            },
        )
        eos_token_id = self.model.embedder_tokenizer.eos_token_id
        eos_tokens = (
            torch.ones((len(hypotheses), 1), dtype=torch.long, device=self.device)
            * eos_token_id
        )
        hypotheses_with_eos = torch.cat((hypotheses[:, 1:], eos_tokens), dim=1)
        hypothesis_attention_mask = torch.ones_like(
            hypotheses_with_eos, device=self.device
        )
        with torch.no_grad():
            hypothesis_embedding = self.model.call_embedding_model(
                input_ids=hypotheses_with_eos,
                attention_mask=hypothesis_attention_mask,
            )
        return hypothesis_embedding

    def __call__(
        self, input_ids: torch.LongTensor, next_token_logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = round(
            self.hypothesis_embedding.shape[0] / self.hypothesis_num_samples
        )
        batch_size_times_beam_width = input_ids.shape[0]
        assert batch_size_times_beam_width % batch_size == 0
        beam_width = int(batch_size_times_beam_width / batch_size)

        hypothesis_embedding = (
            self.hypothesis_embedding[:, None, :]
            .repeat((1, beam_width, 1))
            .reshape((batch_size * beam_width * self.hypothesis_num_samples, -1))
        )
        input_ids = (
            input_ids.reshape((batch_size, beam_width, -1))
            .repeat((1, self.hypothesis_num_samples, 1))
            .reshape((batch_size * beam_width * self.hypothesis_num_samples, -1))
        )
        with torch.no_grad():
            bad_outputs = self.model(
                embedder_input_ids=None,
                embedder_attention_mask=None,
                frozen_embeddings=hypothesis_embedding,
                decoder_input_ids=input_ids,
            )
        bad_token_logits = bad_outputs.logits[:, -1] * self.gamma
        bad_token_logits = (
            bad_token_logits.reshape(
                (batch_size, beam_width, self.hypothesis_num_samples, -1)
            )
            .mean(dim=2)
            .reshape((batch_size * beam_width, -1))
        )

        diff_logits = next_token_logits.log_softmax(1) - bad_token_logits.log_softmax(1)
        #
        next_token_probs = next_token_logits.softmax(-1)
        V_mask = (
            next_token_probs
            >= self.alpha * next_token_probs.max(dim=1, keepdim=True).values
        )
        diff_logits = torch.where(V_mask, diff_logits, diff_logits - 10**10)
        return diff_logits
