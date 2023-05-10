from typing import Dict, Optional

import torch
import torch.nn as nn
import transformers

from .model_utils import mean_pool


class InversionModelNonAutoregressive(nn.Module):
    embedder: torch.nn.Module
    encoder: transformers.AutoModel
    tokenizer: transformers.AutoTokenizer
    embedder_tokenizer: transformers.AutoTokenizer

    def __init__(
        self,
        embedder: torch.nn.Module,
        encoder: transformers.AutoModel,
        embedder_tokenizer: transformers.AutoTokenizer,
        tokenizer: transformers.AutoTokenizer,
    ):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.embedder_tokenizer = embedder_tokenizer
        self.tokenizer = tokenizer
        self.lm_transform = nn.Sequential(
            nn.Linear(self.d_encoder, self.d_encoder),
            nn.GELU(),
            nn.LayerNorm(self.d_encoder),
        )
        self.in_projection = torch.nn.Sequential(
            torch.nn.Linear(self.d_embedder, self.d_encoder),
            torch.nn.GELU(),
            torch.nn.Linear(self.d_encoder, self.d_encoder),
        )

    @property
    def d_encoder(self) -> int:
        return self.encoder.config.d_model

    @property
    def d_embedder(self) -> int:
        return self.embedder.config.d_model

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # TODO respect generation kwargs.
        with torch.no_grad():
            logits = self.masked_lm_logits(**inputs)
        # TODO implement different generation strategies

        return logits.argmax(dim=2)

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        embeddings = mean_pool(hidden_state, attention_mask)
        import pdb

        pdb.set_trace()
        return embeddings

    def masked_lm_logits(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # Drop the first token, which is the sentence embedding input.
        # tokens = outputs.last_hidden_state[:, 1:, :]
        # Project
        projected = self.lm_transform(outputs.last_hidden_state)
        word_embeddings = self.encoder.get_input_embeddings().weight
        # Multiply by vocab
        logits = projected @ word_embeddings.T
        return logits

    def masked_lm_loss(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.masked_lm_logits(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        batch_size, seq_length, v = logits.shape
        logits = logits.reshape((batch_size * seq_length, v))
        labels = labels.reshape((batch_size * seq_length,))

        return torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)

    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_length = embedder_input_ids.shape
        if frozen_embeddings is None:
            with torch.no_grad():
                embedding = self.call_embedding_model(
                    input_ids=embedder_input_ids, attention_mask=embedder_attention_mask
                )
        else:
            embedding = frozen_embeddings
        assert embedding.shape == (batch_size, self.d_embedder)
        embedding = self.in_projection(embedding)
        # TODO: check that it's ok if we make every token '<unk>'.
        input_ids = self.tokenizer.unk_token_id * torch.ones_like(
            labels, device=labels.device
        )
        inputs_embeds = self.encoder.embed_tokens(input_ids)
        # TODO: support & ablate concatenation methods.
        inputs_embeds = torch.cat((embedding[:, None, :], inputs_embeds), dim=1)
        labels = torch.cat(
            (
                -100
                * torch.ones((batch_size, 1), dtype=labels.dtype, device=labels.device),
                labels,
            ),
            dim=1,
        )
        attention_mask = torch.ones_like(labels, device=labels.device)
        loss = self.masked_lm_loss(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"loss": loss}  # hf trainer output format
