from typing import Dict, Optional

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.model_utils import (
    load_embedder_and_tokenizer,
    load_tokenizer,
    mean_pool,
)


class InversionModelBagOfWords(transformers.PreTrainedModel):

    config_class = InversionConfig
    embedder: torch.nn.Module
    encoder: transformers.AutoModel
    tokenizer: transformers.AutoTokenizer
    embedder_tokenizer: transformers.AutoTokenizer

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name, torch_dtype=config.embedder_torch_dtype
        )
        encoder = transformers.AutoModel.from_pretrained(
            config.model_name_or_path,
        ).encoder
        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )

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
        self.config = config

    @property
    def d_encoder(self) -> int:
        return self.encoder.config.d_model

    @property
    def d_embedder(self) -> int:
        if self.config.custom_embedder_name in ["gtr-base", "sbert", "st5"]:
            return 768
        elif self.config.custom_embedder_name=="minilm":
            return 384
        else :
            raise ValueError(f"Unknown model name: {self.config.custom_embedder_name}")


    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # TODO respect generation kwargs.
        with torch.no_grad():
            logits = self.forward(**inputs)["logits"]
        # TODO implement different generation strategies
        max_length = generation_kwargs["max_length"]
        # Take top `seq_length` tokens
        return logits.topk(max_length, dim=1).indices

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        sentences = [self.embedder_tokenizer.decode(ids) for ids in input_ids]
        embeddings = self.embedder.encode(sentences)
        # hidden_state = embeddings.last_hidden_state

        return embeddings

    def bow_logits(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # Drop the first token, which is the sentence embedding input.
        # tokens = outputs.last_hidden_state[:, 1:, :]
        # Project
        output_vector = mean_pool(outputs.last_hidden_state, attention_mask)
        projected = self.lm_transform(output_vector)
        word_embeddings = self.encoder.get_input_embeddings().weight
        # Multiply by vocab
        logits = projected @ word_embeddings.T
        return logits

    def bow_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        vocab_size = self.encoder.get_input_embeddings().weight.shape[0]
        vocab = torch.arange(vocab_size, dtype=labels.dtype, device=labels.device)
        one_hot_labels = (labels[:, :, None] == vocab[None, :]).any(dim=1).float()
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, one_hot_labels
        )

    def forward(
        self,
        embedder_input_ids: torch.Tensor = None,
        embedder_attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # batch_size, seq_length = embedder_input_ids.shape
        if frozen_embeddings is None:
            with torch.no_grad():
                embedding = self.call_embedding_model(
                    input_ids=embedder_input_ids, attention_mask=embedder_attention_mask
                )
        else:
            embedding = frozen_embeddings
        batch_size = embedding.shape[0]
        # assert embedding.shape == (batch_size, self.d_embedder)
        embedding = self.in_projection(embedding)
        # TODO: check that it's ok if we make every token '<unk>'.
        import numpy as np
        input_ids = self.tokenizer.unk_token_id * torch.ones(
            [batch_size, self.config.max_seq_length], device=torch.device("cuda")
        )
        input_ids = input_ids.long()
        inputs_embeds = self.encoder.embed_tokens(input_ids)
        # TODO: support & ablate concatenation methods.
        inputs_embeds = torch.cat((embedding[:, None, :], inputs_embeds), dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[0:2], device=inputs_embeds.device
        )
        logits = self.bow_logits(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        outputs = {"logits": logits}  # hf trainer output format
        if labels is not None:
            labels = torch.cat(
                (
                    -100
                    * torch.ones(
                        (batch_size, 1), dtype=labels.dtype, device=labels.device
                    ),
                    labels,
                ),
                dim=1,
            )
            loss = self.bow_loss(
                logits=logits,
                labels=labels,
            )
            outputs["loss"] = loss
        return outputs
