from typing import Optional, Tuple

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel


class InversionFromLogitsModel(InversionModel):

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        # hacky way of checking if model is a pre-trained HF decoder
        assert (
            ("CausalLM" in str(type(self.embedder))) or 
             ("LMHead" in str(type(self.embedder)))
        )
        encoder_hidden_dim = self.encoder_decoder.config.hidden_size 
        self.embedder_dim = 768
        self.embedder_is_decoder = True
        bottleneck_dim = (
            768  # 768 * 30k = 23m params. TODO: is this rank reduction harmful?
        )
        # todo: make prettier
        self.num_zeros_to_add = 768 - ((self.embedder.config.vocab_size + 768) % 768)
        self.num_repeat_tokens = round(
            (self.embedder.config.vocab_size + self.num_zeros_to_add) / 768
        )
        self.sequence_weights = torch.nn.Parameter(
            torch.randn(
                (self.num_repeat_tokens, self.embedder_dim, self.embedder_dim),
                dtype=torch.float32,
            ),
            requires_grad=True,
        )
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim),
        )
    
    def embed_and_project(
            self,
            embedder_input_ids: Optional[torch.Tensor],
            embedder_attention_mask: Optional[torch.Tensor],
            frozen_embeddings: Optional[torch.Tensor] = None,
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
        embeddings = embeddings.reshape(
            (embeddings.shape[0], self.num_repeat_tokens, -1)
        )
        embeddings = torch.einsum("bsd,sdw->bsw", embeddings, self.sequence_weights)
        embeddings = self.embedding_transform(embeddings)
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )
        return embeddings, attention_mask

    def _process_embedder_output(
        self,
        outputs: transformers.modeling_outputs.BaseModelOutput,
        attention_mask: torch.Tensor,
        ) -> torch.Tensor:
        # embeddings = outputs.logits[:, -1, :]# (batch, sequence_length, vocab_size)
        embeddings = outputs.logits[:, -1, :]  # .log_softmax(dim=2)
        zeros = torch.zeros(
            (embeddings.shape[0], self.num_zeros_to_add),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        embeddings = torch.cat((embeddings, zeros), dim=1)
        # hidden_state = outputs.hidden_states[-1]
        return embeddings