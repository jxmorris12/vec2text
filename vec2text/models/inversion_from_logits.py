import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel


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

        self.num_zeros_to_add = encoder_hidden_dim - (
            (self.embedder.config.vocab_size + encoder_hidden_dim) % encoder_hidden_dim
        )
        self.num_repeat_tokens = round(
            (self.embedder.config.vocab_size + self.num_zeros_to_add)
            / encoder_hidden_dim
        )
        self.embedding_transform = nn.Sequential(
            nn.Linear(encoder_hidden_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim),
        )
        self.sequence_weights = nn.Parameter(
            torch.randn(
                (self.num_repeat_tokens, encoder_hidden_dim, encoder_hidden_dim),
                dtype=torch.float32,
            ),
            requires_grad=True,
        )

        self.unigram_beta = 0.01  # How much to update unigram with each batch
        self.unigram = nn.Parameter(
            torch.zeros(
                (1, self.embedder.config.vocab_size + self.num_zeros_to_add),
                dtype=torch.float32,
            ),
            requires_grad=False,
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

        inputs_str = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        emb_input_ids = self.embedder_tokenizer(
            inputs_str,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(next(self.parameters()).device)

        model_output = embedder(**emb_input_ids)
        return self._process_embedder_output(model_output, emb_input_ids.attention_mask)

    def embed_and_project(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 2  # batch by d
        elif self.embedder_no_grad:
            with torch.no_grad():
                embeddings = self.call_embedding_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            embeddings = self.call_embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        embeddings = embeddings.to(self.sequence_weights.dtype)

        if self.training:
            # Update unigram.
            unigram_batch = embeddings.mean(dim=0, keepdim=True)
            self.unigram.data = self.unigram.data * (
                1 - self.unigram_beta
            ) + unigram_batch * (self.unigram_beta)
        embeddings -= self.unigram

        if self._zero_except_topk is not None:
            embeddings = zero_embedding_except_topk(
                embeddings,
                vocab_size=self.embedder.config.vocab_size,
                k=self._zero_except_topk,
                default_val=-30.0,
            )

        embeddings = embeddings.to(self.sequence_weights.dtype)
        embeddings = embeddings.reshape(
            (embeddings.shape[0], self.num_repeat_tokens, self.encoder_hidden_dim)
        )
        embeddings = torch.einsum("bsd,sdw->bsw", embeddings, self.sequence_weights)
        embeddings = embeddings.to(next(self.embedding_transform.parameters()).dtype)
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

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        inputs_embeds, attention_mask = self.embed_and_project(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                # decoder_attention_mask=inputs["decoder_attention_mask"],
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                **generation_kwargs,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask

        inputs_embeds, attention_mask = self.embed_and_project(
            input_ids=input_ids,
            attention_mask=attention_mask,
            frozen_embeddings=frozen_embeddings,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
        )
