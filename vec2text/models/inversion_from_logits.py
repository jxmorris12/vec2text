from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel

LOGIT_FILTER_VALUE = -1 * 10**7

# TODO: Remove conflicting duplicate features: zero-except-top-k and
# emb-top-k.


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
        self._emb_top_p = None
        self._emb_top_k = None
        self._emb_temp = None
        self._softmax_in_log_space = True

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
            if suffix_ids is None:
                suffix_ids = (
                    torch.ones(
                        (len(embeddings), 1), dtype=torch.long, device=self.device
                    )
                    * self.encoder_decoder.config.eos_token_id
                )

            # print("suffix_ids =", suffix_ids)
            assert len(suffix_ids) == len(
                embeddings
            ), f"got {len(suffix_ids)} suffixes and {len(embeddings)} embeddings?"
            #
            # Get embeddings for each token in suffix.
            #
            suffix_attention_mask = (
                suffix_ids != self.encoder_decoder.config.pad_token_id
            ).int()
            # add pad token so we can shift.
            # print("suffix_ids:", suffix_ids)
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
            embeddings = embeddings.to(self.sequence_weights.dtype)
            embeddings = embeddings.reshape(
                (embeddings.shape[0], self.num_repeat_tokens, self.embedder_dim)
            )
            embeddings = torch.einsum("bsd,sdw->bsw", embeddings, self.sequence_weights)
            embeddings = embeddings.to(
                next(self.embedding_transform.parameters()).dtype
            )
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
        B = len(attention_mask)
        logits = outputs.logits[torch.arange(B), attention_mask.sum(1) - 1]

        logit_filter_value = logits.min()

        if self._emb_top_k is not None:
            topk = logits.topk(k=min(logits.shape[1], self._emb_top_k), dim=1)
            logits = torch.zeros_like(logits, device=logits.device)
            logits = logits.scatter_add(1, topk.indices, topk.values)
            logits = logits.where(logits != 0, logit_filter_value)

        if self._emb_top_p is not None:
            for j in range(len(logits)):
                sorted_logits, sorted_indices = logits[j].sort(descending=True)
                cumulative_probs = sorted_logits.softmax(dim=0).cumsum(dim=0)
                topp_idxs = sorted_indices[cumulative_probs >= self._emb_top_p]
                logits[j] = logits[j].scatter(
                    dim=0, index=topp_idxs, value=logit_filter_value
                )

        if self._emb_temp is not None:
            logits /= self._emb_temp

        if self._softmax_in_log_space:
            embeddings = logits.log_softmax(dim=1)
        else:
            embeddings = (logits.log_softmax(dim=1).exp() + 1e-9).log()
        zeros = torch.zeros(
            (B, self.num_zeros_to_add),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        return torch.cat((embeddings, zeros), dim=1)

    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        suffix_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask
        if (suffix_ids is None) and self.config.suffix_conditioning:
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
                prefix_length = 1
                print("prefix_length:", prefix_length)

                if labels is not None:
                    # create suffix based on the labels and selected prefix_length.
                    suffix_ids = labels[:, prefix_length:]
                    suffix_ids = suffix_ids.where(
                        suffix_ids >= 0, self.encoder_decoder.config.pad_token_id
                    )  # replace -100 with 0.
                    labels = labels.where(
                        torch.arange(seq_length, device=self.device)[None, :]
                        < prefix_length,
                        -100,
                    )
                    ignore_token_id = -100
                    eos_token_id = self.tokenizer.eos_token_id
                    first_ignore_token_id = (
                        ((labels == ignore_token_id) | (labels == eos_token_id))
                        .long()
                        .argmax(dim=1)
                    )
                    eos_tokens = (
                        torch.ones(
                            (batch_size, 1), dtype=torch.long, device=self.device
                        )
                        * eos_token_id
                    )
                    labels = labels.scatter(
                        dim=1, index=first_ignore_token_id[:, None], src=eos_tokens
                    )
                else:
                    suffix_ids = None
            else:
                suffix_ids = None

        # print("suffix_ids:", suffix_ids, "labels:", labels)
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
            past_key_values=past_key_values,
        )
