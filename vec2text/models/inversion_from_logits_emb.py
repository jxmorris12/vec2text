from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.inversion_from_logits import InversionFromLogitsModel


class InversionFromLogitsEmbModel(InversionFromLogitsModel):
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        self.embedding_proj = nn.Sequential(
            nn.Linear(self.embedder_dim, self.encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim),
        )
        self.num_tokens = num_tokens = 32
        self.num_zeros_to_add = num_zeros_to_add = (
            (num_tokens - (self.embedder_dim % num_tokens)) % num_tokens
        )
        word_embeddings = self.embedder.model.embed_tokens.weight.detach().clone()
        word_embedding_zeros = torch.zeros(
            (num_zeros_to_add, self.embedder_dim),
            dtype=torch.float32, device=word_embeddings.device)
        padded_word_embeddings = torch.cat((
            word_embeddings, word_embedding_zeros
        ), dim=0)
        word_embeddings = padded_word_embeddings.reshape(
            (num_tokens, -1, self.embedder_dim)
        )
        self.word_embeddings =  nn.Parameter(
            word_embeddings, requires_grad=False,
        )

        self.embedder_vocab_size = self.embedder.config.vocab_size
        self.minibatch_size = 128
        self.unigram_beta = 0.01 # How much to update unigram with each batch
        self.unigram = nn.Parameter(
            torch.zeros(
                (1, self.embedder.config.vocab_size + self.num_zeros_to_add),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )   

    def embed_and_project(
        self,
        embedder_input_ids: Optional[torch.Tensor],
        embedder_attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
        suffix_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 2  # batch by d
        elif self.embedder_no_grad:
            with torch.no_grad():
                embeddings = self.call_embedding_model(
                    input_ids=embedder_input_ids,
                    attention_mask=embedder_attention_mask,
                )
        else:
            embeddings = self.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            )
        
        num_tokens = self.num_tokens
        # Remove any extraneous zeros
        embeddings = embeddings[:, :self.embedder_vocab_size] # (B, V)

        if self.training:
            # Update unigram.
            unigram_batch = embeddings.mean(dim=0, keepdim=True)
            self.unigram.data = (
                self.unigram.data * (1 - self.unigram_beta) +
                unigram_batch * (self.unigram_beta)
            )
        embeddings -= self.unigram

        batch_size = embeddings.shape[0]
        logits_zeros = torch.zeros(
            (batch_size, self.num_zeros_to_add), 
            dtype=embeddings.dtype, 
            device=embeddings.device
        )
        logits = torch.cat(
            (embeddings, logits_zeros), dim=1
        ).to(self.sequence_weights.dtype)
        logits = logits.reshape((batch_size, num_tokens, -1))
        # logits = logits.softmax(dim=-1) # softmax for weighted sum

        # TODO compute cross-vocab alignment, etc.
        with torch.no_grad():
            # Minibatch 
            embeddings_list = []
            i = 0
            while i < batch_size:
                batch_logits = logits[i : i+self.minibatch_size, ...]
                batch_embeddings = torch.einsum('smd,bsm -> bsd', self.word_embeddings, batch_logits)
                embeddings_list.append(batch_embeddings)
                i += self.minibatch_size
            embeddings = torch.cat(embeddings_list, dim=0)
        
        embeddings = self.embedding_proj(embeddings)
        assert embeddings.shape == (
            batch_size,
            num_tokens,
            self.encoder_hidden_dim,
        )
        attention_mask = torch.ones(
            (batch_size, num_tokens), dtype=torch.long, device=embeddings.device
        )
        return embeddings, attention_mask
