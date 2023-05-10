from typing import Dict, Optional

import torch
import torch.nn as nn
import transformers

from .model_utils import (
    mean_pool,
)

class InversionModelNonAutoregressive(nn.Module):
    embedder: torch.nn.Module
    encoder: transformers.AutoModel
    tokenizer: transformers.AutoTokenizer
    embedder_tokenizer: transformers.AutoTokenizer
    def __init__(self, 
        embedder: torch.nn.Module, encoder: transformers.AutoModel, embedder_tokenizer: transformers.AutoTokenizer, tokenizer: transformers.AutoTokenizer):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.embedder_tokenizer = embedder_tokenizer
        self.tokenizer = tokenizer
        self.lm_transform = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768),
        )
        self.in_projection = torch.nn.Sequential(
            torch.nn.Linear(768, 768), 
            torch.nn.GELU(), 
            torch.nn.Linear(768, 768)
        )
    
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # TODO respect generation kwargs.
        with torch.no_grad():
            logits = self.masked_lm_logits(
                **inputs
            )
        # TODO implement different generation strategies
        
        return logits.argmax(dim=2)

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        model_output = self.embedder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_state = outputs.last_hidden_state
        embeddings = mean_pool(hidden_state, attention_mask)
        return embeddings
    
    def masked_lm_logits(
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # Drop the first token, which is the sentence embedding input.
        # tokens = outputs.last_hidden_state[:, 1:, :]
        # Project
        projected = self.lm_transform(tokens)
        # Multiply by vocab
        logits = (
            projected @ self.encoder.inputs_embeds.word_embeddings.weight.T
        )
        return logits
    
    def masked_lm_loss(
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        logits = self.masked_lm_logits(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return torch.nn.functional.cross_entropy(
            logits, labels, ignore_index=-100
        )

    
    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_length = input_ids.shape
        assert embedding.shape == (batch_size, 768)
        embedding = self.in_projection(embedding)
        inputs_embeds = self.encoder.embed_tokens(input_ids)
        #
        ones = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
        # TODO: support & ablate concatenation methods.
        inputs_embeds = torch.cat((embedding[:, None, :], inputs_embeds), dim=1)
        labels = torch.cat((
            -100 * torch.ones((batch_size, 1), dtype=labels.dtype, device=labels.device),
            labels,
        ), dim=1)
        return self.masked_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )