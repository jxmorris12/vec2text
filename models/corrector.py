import copy
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers


class CorrectorModel(torch.nn.Module):
    """Embeds text and concats with a provided embedding.

    TODO improve comment here.
    """

    # Encoder-decoder model we train to correct embedding hypotheses.
    encoder_decoder: transformers.PreTrainedModel

    def __init__(
        self,
        encoder_decoder: transformers.PreTrainedModel,
        embedder_dim: int = 768,
        num_repeat_tokens: int = 16,
        bottleneck_dim: int = 768,
    ):
        super().__init__()
        self.encoder_decoder = encoder_decoder.to_bettertransformer()
        self.embedder_dim = embedder_dim
        self.num_repeat_tokens = num_repeat_tokens
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedding_transform_1 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),  # TODO consider dropout or normalization here.
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim * num_repeat_tokens),
        )
        self.embedding_transform_2 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),  # TODO consider dropout or normalization here.
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim * num_repeat_tokens),
        )
        self.embedding_transform_3 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),  # TODO consider dropout or normalization here.
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim * num_repeat_tokens),
        )

    def get_encoder_embedding(
        self,
        embedding: torch.Tensor,
        hypothesis_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embeds e, e', and (e-e') using `self.embedding_transform`."""
        batch_size, D = embedding.shape
        assert D == self.encoder_hidden_dim
        assert embedding.shape == (batch_size, D)
        assert hypothesis_embedding.shape == (batch_size, D)
        diff_embedding = embedding - hypothesis_embedding
        #
        embedding = self.embedding_transform(embedding)
        embedding = embedding.reshape((batch_size, self.num_repeat_tokens, D))
        #
        diff_embedding = self.embedding_transform(diff_embedding)
        diff_embedding = diff_embedding.reshape((batch_size, self.num_repeat_tokens, D))
        #
        hypothesis_embedding = self.embedding_transform(hypothesis_embedding)
        hypothesis_embedding = hypothesis_embedding.reshape(
            (batch_size, self.num_repeat_tokens, D)
        )
        #
        ones = torch.ones((batch_size, 1), dtype=torch.long, device=embedding.device)
        sep_token = ones * self.encoder_decoder.config.eos_token_id
        sep_token = self.encoder_decoder.encoder.embed_tokens(sep_token)
        inputs_embeds = torch.cat(
            (
                sep_token,
                embedding,
                sep_token,
                hypothesis_embedding,
                sep_token,
                diff_embedding,
            ),
            dim=1,
        )
        attention_mask = ones.repeat((1, 3 * (1 + self.num_repeat_tokens)))
        return (inputs_embeds, attention_mask)

    def null_hypothesis_embedding(
        self, hypothesis_embedding: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(
            hypothesis_embedding, device=hypothesis_embedding.device
        )

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
        embed_generated_hypothesis_func: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Does two-step generation by generating a hypothesis and then a correction."""
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        max_length = inputs["hypothesis_input_ids"].shape[1]
        # if "max_length" not in generation_kwargs:
        generation_kwargs["max_length"] = max_length
        generation_kwargs["min_length"] = generation_kwargs["max_length"]

        embedding = inputs["frozen_embeddings"]
        #
        # [0/2] Use hypothesis.
        #
        hypothesis_embedding = inputs["hypothesis_embedding"]
        hypothesis_input_ids = inputs["hypothesis_input_ids"]
        hypothesis_attention_mask = inputs["hypothesis_attention_mask"]

        #
        # [1/2] Generate hypothesis.
        #
        # initial_inputs_embeds, initial_attention_mask = self.get_encoder_embedding(
        #     embedding=embedding,
        #     hypothesis_embedding=self.null_hypothesis_embedding(embedding),
        # )
        # hypothesis_input_ids = self.encoder_decoder.generate(
        #     inputs_embeds=initial_inputs_embeds,
        #     attention_mask=initial_attention_mask,
        #     **generation_kwargs,
        # )
        # hypothesis_embedding = embed_generated_hypothesis_func(hypothesis_input_ids)

        # The "start of sequence" token for the second guess is the end-of-sequence
        # token from the hypothesis.
        # batch_size = len(hypothesis_input_ids)
        # bos_tokens = (
        #     torch.ones((batch_size, 1), device=embedding.device, dtype=torch.long)
        #     * self.encoder_decoder.config.decoder_start_token_id
        # )
        # hypothesis_input_ids = torch.cat(
        #     (bos_tokens, hypothesis_input_ids), dim=1
        # )
        hypothesis_attention_mask = (
            hypothesis_input_ids != self.encoder_decoder.config.pad_token_id
        )
        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=embedding,
            hypothesis_embedding=hypothesis_embedding,
        )

        #
        # [2/2] Given generated hypothesis & embedding, generate a correction.
        #
        # We want to generate starting from the hypothesis
        generation_kwargs["max_length"] += max_length
        # Force the model to generate all the tokens
        generation_kwargs["min_length"] = generation_kwargs["max_length"]

        return self.encoder_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=hypothesis_input_ids,
            decoder_attention_mask=hypothesis_attention_mask,
            **generation_kwargs,
        )

    def forward(
        self,
        embedding: torch.Tensor,
        hypothesis_embedding: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=embedding,
            hypothesis_embedding=hypothesis_embedding,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
