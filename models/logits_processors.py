import random
from typing import Dict

import torch
import transformers

from models import InversionModel


class ContrastiveLogitsProcessor(transformers.LogitsProcessor):
    model: InversionModel
    alpha: float
    gamma: float
    beta: float
    batch_size: int
    seq_length: int
    hypothesis_embedding: torch.Tensor  # shape [batch_size, num_hypothesis_embeddings, emb_d]

    def __init__(
        self,
        model: InversionModel,
        alpha: float,
        gamma: float,
        beta: float,
        hypothesis_temperature: float,
        hypothesis_num_samples: int,
        inputs: Dict[str, torch.Tensor],
    ):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.hypothesis_temperature = hypothesis_temperature  # 1e-9
        self.hypothesis_num_samples = hypothesis_num_samples
        self.batch_size, self.seq_length = inputs["input_ids"].shape
        self.hypothesis_embedding = self._generate_hypotheses_and_embed(inputs=inputs)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def embedding_dim(self) -> int:
        return self.hypothesis_embedding.shape[2]

    def _get_hypothesis_embeddings(self, hypotheses: torch.Tensor) -> torch.Tensor:
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

    def _generate_hypotheses_and_embed(
        self, inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if "decoder_input_ids" in inputs:
            inputs.pop("decoder_input_ids")
        if "decoder_attention_mask" in inputs:
            inputs.pop("decoder_attention_mask")

        do_sample = self.hypothesis_temperature > 0
        hypotheses = self.model.generate(
            inputs=inputs,
            generation_kwargs={
                "max_length": self.seq_length,
                "early_stopping": True,
                "num_beams": (self.hypothesis_num_samples if not do_sample else 1),
                "do_sample": do_sample,
                "temperature": self.hypothesis_temperature,
                "num_return_sequences": self.hypothesis_num_samples,
            },
        )
        hypothesis_embedding = self._get_hypothesis_embeddings(hypotheses=hypotheses)
        hypothesis_embedding = hypothesis_embedding.reshape(
            (self.batch_size, self.hypothesis_num_samples, -1)
        )
        return hypothesis_embedding

    def update_hypotheses(self, hypotheses: torch.Tensor) -> None:
        new_hypothesis_embedding = self._get_hypothesis_embeddings(
            hypotheses=hypotheses
        )
        new_hypothesis_embedding = new_hypothesis_embedding.reshape(
            (self.batch_size, -1, self.embedding_dim)
        )
        self.hypothesis_embedding = torch.cat(
            (self.hypothesis_embedding, new_hypothesis_embedding), dim=1
        )

    def __call__(
        self, input_ids: torch.LongTensor, next_token_logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        (
            batch_size,
            num_hypothesis_embeddings,
            embedding_dim,
        ) = self.hypothesis_embedding.shape
        batch_size_times_beam_width = input_ids.shape[0]
        assert batch_size_times_beam_width % batch_size == 0
        beam_width = int(batch_size_times_beam_width / batch_size)

        hypothesis_embedding = (
            self.hypothesis_embedding[:, None, :]
            .reshape((batch_size, -1, self.embedding_dim))
            .repeat((1, beam_width, 1))
            .reshape((batch_size * num_hypothesis_embeddings * beam_width, -1))
        )
        input_ids = (
            input_ids.reshape((batch_size, beam_width, -1))
            .repeat((1, num_hypothesis_embeddings, 1))
            .reshape((batch_size * num_hypothesis_embeddings * beam_width, -1))
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
                (batch_size, num_hypothesis_embeddings, beam_width, -1)
            )
            .log_softmax(dim=3)
            .logsumexp(dim=1)
            .reshape((batch_size * beam_width, -1))
        )
        #
        next_token_logits = next_token_logits.log_softmax(1)
        diff_logits = next_token_logits - (self.beta * bad_token_logits)
        #
        next_token_probs = next_token_logits.softmax(-1)
        V_mask = (
            next_token_probs
            >= self.alpha * next_token_probs.max(dim=1, keepdim=True).values
        )
        diff_logits = torch.where(V_mask, diff_logits, diff_logits - 10**10)
        return diff_logits


################################################################################
class EncourageTrueTokensLogitsProcessor(transformers.LogitsProcessor):
    true_input_ids: torch.LongTensor
    gamma: float

    def __init__(self, true_input_ids: torch.LongTensor):
        self.true_input_ids = true_input_ids
        self.gamma = 10.0

    def __call__(
        self, input_ids: torch.LongTensor, next_token_logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        true_next_tokens = self.true_input_ids[:, input_ids.shape[1] - 1]
        fake_logits = torch.zeros_like(
            next_token_logits, device=next_token_logits.device
        )
        # todo vectorize
        for i in range(input_ids.shape[0]):
            fake_logits[i, true_next_tokens[i]] += random.random()

        # gamma = self.gammas[:, None].to(next_token_logits.device)
        gamma = self.gamma
        logits = (next_token_logits + fake_logits * gamma).log_softmax(1)
        # import pdb; pdb.set_trace()

        return logits


################################################################################
