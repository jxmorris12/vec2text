from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel


def zero_embedding_except_topk(
    embeddings: torch.Tensor, vocab_size: int, k: torch.Tensor, default_val: float
) -> torch.Tensor:
    # return embeddings
    topk = embeddings[:, :vocab_size].topk(k=k, dim=1)
    new_emb = torch.zeros_like(embeddings, device=embeddings.device) + default_val
    return new_emb.scatter_add(1, topk.indices, topk.values)


class InversionFromLogitsModel(InversionModel):
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        # hacky way of checking if model is a pre-trained HF decoder
        assert ("CausalLM" in str(type(self.embedder))) or (
            "LMHead" in str(type(self.embedder))
        )
        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.encoder_hidden_dim = encoder_hidden_dim
        self.embedder_is_decoder = True
        bottleneck_dim = self.bottleneck_dim

        # TODO: Make prettier & remove hardcoded values
        embedder_dim = self.embedder_dim
        self.num_zeros_to_add = embedder_dim - (
            (self.embedder.config.vocab_size + embedder_dim) % embedder_dim
        )
        self.num_repeat_tokens = round(
            (self.embedder.config.vocab_size + self.num_zeros_to_add) / embedder_dim
        )
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim),
        )
        if self.config.suffix_conditioning:
            self.suffix_transform = nn.Sequential(
                nn.Linear(encoder_hidden_dim, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, encoder_hidden_dim),
            )
        self.sequence_weights = nn.Parameter(
            torch.randn(
                (self.num_repeat_tokens, self.embedder_dim, self.embedder_dim),
                dtype=torch.float32,
            ),
            requires_grad=True,
        )
        self._zero_except_topk = vars(config).get("embedding_zero_except_topk")
        print("Set zero-except-top-K value =", self._zero_except_topk)

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedder = self.embedder
        model_output = embedder(input_ids=input_ids, attention_mask=attention_mask)
        return self._process_embedder_output(model_output, attention_mask)

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

        embeddings = embeddings.to(self.sequence_weights.dtype)

        if self._zero_except_topk is not None:
            embeddings = zero_embedding_except_topk(
                embeddings,
                vocab_size=self.embedder.config.vocab_size,
                k=self._zero_except_topk,
                default_val=-30.0,
            )

        if self.config.suffix_conditioning:
            # below message will go away when we get data with suffixes. it only happens during eval anyway.
            if suffix_ids is None:
                suffix_ids = torch.tensor(
                    [[0]] * len(embeddings), dtype=torch.long, device=self.device
                )
            #
            # Get embeddings for each token in suffix.
            #
            suffix_attention_mask = (
                suffix_ids != self.encoder_decoder.config.pad_token_id
            ).int()
            # add pad token so we can shift.
            suffix_embeddings = self.encoder_decoder.encoder.embed_tokens(suffix_ids)
            suffix_embeddings = self.suffix_transform(suffix_embeddings)
            #
            # Get embeddings for each next-token logit from suffix.
            #
            embeddings = embeddings.reshape(
                (embeddings.shape[0], self.num_repeat_tokens, self.embedder_dim)
            )
            embeddings = torch.einsum("bsd,sdw->bsw", embeddings, self.sequence_weights)
            embeddings = self.embedding_transform(embeddings)
            logit_attention_mask = torch.ones(
                (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
            )
            #
            # TODO add positional embeddings :-)
            #
            embeddings = torch.cat(
                (suffix_embeddings, embeddings),
                dim=1,
            )
            attention_mask = torch.cat(
                (suffix_attention_mask, logit_attention_mask), dim=1
            )
        else:
            embeddings = embeddings.reshape(
                (embeddings.shape[0], self.num_repeat_tokens, self.embedder_dim)
            )
            embeddings = torch.einsum("bsd,sdw->bsw", embeddings, self.sequence_weights)
            embeddings = self.embedding_transform(embeddings)
            attention_mask = torch.ones(
                (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
            )

        assert embeddings.shape == (
            attention_mask.shape[0],
            attention_mask.shape[1],
            self.encoder_hidden_dim,
        )
        return embeddings, attention_mask

    def _process_embedder_output(
        self,
        outputs: transformers.modeling_outputs.BaseModelOutput,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        try:
            embeddings = outputs.logits.log_softmax(dim=2)
        except AttributeError:
            embeddings = outputs
        zeros = torch.zeros(
            (*embeddings.shape[0:2], self.num_zeros_to_add),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        embeddings = torch.cat((embeddings, zeros), dim=2)
        # import pdb; pdb.set_trace()
        # return embeddings[:, -1, :]
        return embeddings[torch.arange(len(embeddings)), attention_mask.sum(1) - 1]

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
                if true_seq_length == 1:
                    prefix_length = 1
                else:
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
            suffix_ids = None
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
