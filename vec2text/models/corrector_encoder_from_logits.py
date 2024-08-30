from typing import Tuple

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig

from .corrector_encoder import CorrectorEncoderModel


class CorrectorEncoderFromLogitsModel(CorrectorEncoderModel):
    config_class = InversionConfig
    encoder_decoder: transformers.PreTrainedModel

    def __init__(
        self,
        config: InversionConfig,
    ):
        super().__init__(config=config)

        config.embedder_dim = 768  # TODO: Pipe this in.
        config.num_zeros_to_add = self.num_zeros_to_add = 512  # TODO: Compute this.
        config.num_repeat_tokens = (
            self.num_repeat_tokens
        ) = 42  # TODO: Compute this properly.

        # TODO: Calculate this explicitly from trainer.
        # self.unigram = torch.load(
        # "/home/jxm3/research/retrieval/inversion/llama_unigram.pt"
        #)

        self.embedder_dim = config.embedder_dim
        bottleneck_dim = config.embedder_dim

        self.sequence_weights_1 = nn.Parameter(
            torch.randn(
                (self.num_repeat_tokens, self.embedder_dim, self.embedder_dim),
                dtype=torch.float32,
            ),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.sequence_weights_1)
        self.sequence_layernorm_1 = nn.LayerNorm(self.embedder_dim)

        self.sequence_weights_2 = nn.Parameter(
            torch.randn(
                (self.num_repeat_tokens, self.embedder_dim, self.embedder_dim),
                dtype=torch.float32,
            ),
            requires_grad=True,
        )
        self.sequence_layernorm_2 = nn.LayerNorm(self.embedder_dim)
        nn.init.xavier_uniform_(self.sequence_weights_2)

        self.sequence_weights_3 = nn.Parameter(
            torch.randn(
                (self.num_repeat_tokens, self.embedder_dim, self.embedder_dim),
                dtype=torch.float32,
            ),
            requires_grad=True,
        )
        self.sequence_layernorm_3 = nn.LayerNorm(self.embedder_dim)
        nn.init.xavier_uniform_(self.sequence_weights_3)

        self.embedding_transform_1 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(
                self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim),
        )
        self.embedding_transform_2 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(
                self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim),
        )
        self.embedding_transform_3 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(
                self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim),
        )

    def get_encoder_embedding(
        self,
        embedding: torch.Tensor,
        hypothesis_embedding: torch.Tensor,
        hypothesis_input_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, D = embedding.shape
        if (self.training) and (self.training_embedding_noise_level > 0):
            embedding += self.training_embedding_noise_level * torch.randn(
                embedding.shape, device=embedding.device
            )
            hypothesis_embedding += self.training_embedding_noise_level * torch.randn(
                hypothesis_embedding.shape, device=hypothesis_embedding.device
            )

        unigram = self.unigram.to(embedding.device)
        embedding = embedding - unigram
        hypothesis_embedding = hypothesis_embedding - unigram

        embedding = embedding[:, :32256]  # (b, 32768) -> (b, 32256)
        hypothesis_embedding = hypothesis_embedding[
            :, :32256
        ]  # (b, 32768) -> (b, 32256)

        diff_embedding = embedding - hypothesis_embedding
        embedding = embedding.to(torch.float32)
        embedding = embedding.reshape(
            (embedding.shape[0], self.num_repeat_tokens, self.embedder_dim)
        )
        embedding = torch.einsum(
            "bsd,sdw->bsw", embedding, self.sequence_weights_1.to(torch.float32)
        )
        embedding = embedding.to(next(self.sequence_layernorm_1.parameters()).dtype)
        embedding = self.sequence_layernorm_1(embedding)
        embedding = self.embedding_transform_1(embedding)
        #
        diff_embedding = diff_embedding.to(torch.float32)
        diff_embedding = diff_embedding.reshape(
            (diff_embedding.shape[0], self.num_repeat_tokens, self.embedder_dim)
        )
        diff_embedding = torch.einsum(
            "bsd,sdw->bsw", diff_embedding, self.sequence_weights_2.to(torch.float32)
        )
        diff_embedding = diff_embedding.to(
            next(self.sequence_layernorm_2.parameters()).dtype
        )
        diff_embedding = self.sequence_layernorm_2(diff_embedding)
        diff_embedding = self.embedding_transform_2(diff_embedding)
        #
        hypothesis_embedding = hypothesis_embedding.to(torch.float32)
        hypothesis_embedding = hypothesis_embedding.reshape(
            (hypothesis_embedding.shape[0], self.num_repeat_tokens, self.embedder_dim)
        )
        hypothesis_embedding = torch.einsum(
            "bsd,sdw->bsw",
            hypothesis_embedding,
            self.sequence_weights_3.to(torch.float32),
        )
        hypothesis_embedding = hypothesis_embedding.to(
            next(self.sequence_layernorm_3.parameters()).dtype
        )
        hypothesis_embedding = self.sequence_layernorm_3(hypothesis_embedding)
        hypothesis_embedding = self.embedding_transform_3(hypothesis_embedding)
        inputs_embeds = self.encoder_decoder.encoder.embed_tokens(hypothesis_input_ids)
        #
        ones = torch.ones(
            (batch_size, 1), dtype=torch.long, device=hypothesis_input_ids.device
        )
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
                sep_token,
                inputs_embeds,
            ),
            dim=1,
        )
        inputs_embeds = self.layernorm(inputs_embeds)
        attention_mask = torch.cat(
            (ones.repeat(1, 4 + 3 * self.num_repeat_tokens), hypothesis_attention_mask),
            dim=1,
        )

        if self.training:
            import wandb

            try:
                wandb.log(
                    {
                        "emb_norm/emb": embedding.abs().mean(),
                        "emb_norm/hypothesis": hypothesis_embedding.abs().mean(),
                        "emb_norm/diff": diff_embedding.abs().mean(),
                        "emb_norm/input_length": attention_mask.shape[1],
                    }
                )
            except Exception:
                pass
        return (inputs_embeds, attention_mask)
