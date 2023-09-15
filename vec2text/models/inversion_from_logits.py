from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel


class InversionFromLogitsModel(InversionModel):
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        # hacky way of checking if model is a pre-trained HF decoder
        assert ("CausalLM" in str(type(self.embedder))) or (
            "LMHead" in str(type(self.embedder))
        )
        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedder_dim = 768
        self.embedder_is_decoder = True
        bottleneck_dim = (
            768  # 768 * 30k = 23m params. TODO: is this rank reduction harmful?
        )
        # TODO: Make prettier & remove hardcoded values
        self.num_zeros_to_add = 768 - ((self.embedder.config.vocab_size + 768) % 768)
        self.num_repeat_tokens = round(
            (self.embedder.config.vocab_size + self.num_zeros_to_add) / 768
        )
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim),
        )
        if self.config.suffix_conditioning:
            # TODO remove hardcoded max sequence length
            self.suffix_position_embedding = torch.nn.Parameter(
                torch.randn(
                    (256, self.num_repeat_tokens, self.embedder_dim),
                    dtype=torch.float32,
                ),
                requires_grad=True,
            )
            self.suffix_transform = nn.Sequential(
                nn.Linear(encoder_hidden_dim, bottleneck_dim),
                nn.Dropout(self.encoder_decoder.config.dropout_rate),
                nn.GELU(),
                nn.Linear(bottleneck_dim, encoder_hidden_dim),
            )
        self.sequence_weights = torch.nn.Parameter(
            torch.randn(
                (self.num_repeat_tokens, self.embedder_dim, self.embedder_dim),
                dtype=torch.float32,
            ),
            requires_grad=True,
        )

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        model_output = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
        return self._process_embedder_output(
            model_output, attention_mask, return_sequence=return_sequence
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
                    return_sequence=True,
                )
        else:
            embeddings = self.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
                return_sequence=True,
            )

        if self.config.suffix_conditioning:
            # below message will go away when we get data with suffixes
            if suffix_ids is None:
                print("warning: suffix-conditioning enabled but no suffix passed")
                suffix_ids = torch.tensor(
                    [[0]] * len(embeddings), dtype=torch.long, device=self.device
                )
                # embeddings = embeddings[:, -1, :]  # next-token logits
            suffix_length = suffix_ids.shape[1]
            embeddings = embeddings[:, -suffix_length:, :]
            suffix_embeddings = self.encoder_decoder.encoder.embed_tokens(suffix_ids)
            suffix_embeddings = self.suffix_transform(suffix_embeddings)
            #
            suffix_length = suffix_ids.shape[1]
            suffix_position_embedding = self.suffix_position_embedding[
                None, :suffix_length, ...
            ]
            embeddings = embeddings.reshape(
                (
                    embeddings.shape[0],
                    suffix_length,
                    self.num_repeat_tokens,
                    self.embedder_dim,
                )
            )
            embeddings = embeddings + suffix_position_embedding
            embeddings = embeddings.mean(dim=1)
            embeddings = torch.einsum("bsd,sdw->bsw", embeddings, self.sequence_weights)
            #
            embeddings = self.embedding_transform(embeddings)
            attention_mask = torch.ones(
                (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
            )
            #
            embeddings = torch.cat((embeddings, suffix_embeddings), dim=1)
            suffix_attention_mask = (suffix_ids != 0).int()
            attention_mask = torch.cat((attention_mask, suffix_attention_mask), dim=1)
        else:
            embeddings = embeddings[:, -1, :]  # next-token logits
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
        return_sequence: bool = False,
    ) -> torch.Tensor:
        embeddings = outputs.logits
        zeros = torch.zeros(
            (*embeddings.shape[0:2], self.num_zeros_to_add),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        embeddings = torch.cat((embeddings, zeros), dim=2)
        # hidden_state = outputs.hidden_states[-1]

        if return_sequence:
            return embeddings
        else:
            return embeddings[:, -1, :]

    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask
        if self.config.suffix_conditioning:
            assert labels is not None
            batch_size, seq_length = labels.shape
            true_seq_length = (labels >= 0).sum(1).min()
            if self.training:
                # Randomly create a suffix from the input.
                # TODO: Pass in suffix directly from (prompted) data.
                # # Remove this hackiness!
                prefix_length = torch.randint(
                    low=1,  # inclusive
                    high=true_seq_length,  # exclusive
                    size=(1,),
                    dtype=torch.long,
                ).item()
            else:
                prefix_length = true_seq_length // 2

            if labels is not None:
                suffix_ids = labels[:, prefix_length:]
                suffix_ids = suffix_ids.clamp(min=0)  # replace -100 with 0.
                labels = labels.where(
                    torch.arange(seq_length, device=self.device)[None, :]
                    < prefix_length,
                    -100,
                )
            else:
                suffix_ids = None
        else:
            prefix_length = None

        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=frozen_embeddings,
            suffix_ids=suffix_ids,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )
