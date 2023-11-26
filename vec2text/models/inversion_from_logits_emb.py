from typing import Optional, Tuple

import torch
import torch.nn as nn

from vec2text.models.config import InversionConfig
from vec2text.models.inversion_from_logits import InversionFromLogitsModel
from vec2text.tokenize_data import get_tokenizer_mapping


class InversionFromLogitsEmbModel(InversionFromLogitsModel):
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        self.embedding_proj = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, self.embedder_dim),
            nn.GELU(),
            nn.Linear(self.embedder_dim, self.encoder_hidden_dim),
        )
        word_embeddings = (
            self.encoder_decoder.encoder.embed_tokens.weight.detach().clone()
        )
        inverter_vocab_size = word_embeddings.shape[0]
        self.num_tokens = num_tokens = 64
        self.num_zeros_to_add = num_zeros_to_add = (
            num_tokens - (word_embeddings.shape[0] % num_tokens)
        ) % num_tokens
        word_embedding_zeros = torch.zeros(
            (num_zeros_to_add, word_embeddings.shape[1]),
            dtype=torch.float32,
            device=word_embeddings.device,
        )
        padded_word_embeddings = torch.cat(
            (word_embeddings, word_embedding_zeros), dim=0
        )
        word_embeddings = padded_word_embeddings.reshape(
            (num_tokens, -1, word_embeddings.shape[1])
        )
        self.word_embeddings = nn.Parameter(
            word_embeddings,
            requires_grad=False,
        )

        self.embedder_vocab_size = self.embedder.config.vocab_size
        self.minibatch_size = 128
        self.unigram_beta = 0.01  # How much to update unigram with each batch
        self.unigram = nn.Parameter(
            torch.zeros(
                (1, inverter_vocab_size),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.tokenizer_mapping = get_tokenizer_mapping(
            config.embedder_model_name,
            config.model_name_or_path,
            self.encoder_decoder.config.vocab_size,
        )

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

        num_tokens = self.num_tokens
        # Remove any extraneous zeros
        embeddings = embeddings[:, : self.tokenizer_mapping.numel()]  # (B, V)

        # Map embeddings to our space.
        batch_size = embeddings.shape[0]
        new_embeddings = torch.zeros(
            (batch_size, self.encoder_decoder.config.vocab_size),
            device=embeddings.device,
            dtype=torch.double,
        )
        mapping = (
            self.tokenizer_mapping[None]
            .repeat((batch_size, 1))
            .to(new_embeddings.device)
        )
        embeddings = new_embeddings.scatter_add(
            dim=1, index=mapping, src=embeddings.to(torch.double).exp()
        ).log()
        embeddings = (
            embeddings.nan_to_num()
        )  # replace empty values from -inf to tiny neg number

        if self.training:
            unigram_batch = embeddings.mean(dim=0, keepdim=True)
            # Update unigram.
            if self.unigram.sum() == 0:
                print("INFO: resetting unigram.")
                self.unigram.data = unigram_batch
            else:
                self.unigram.data = self.unigram.data * (
                    1 - self.unigram_beta
                ) + unigram_batch * (self.unigram_beta)
        embeddings = embeddings - self.unigram
        embeddings = embeddings.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        logits_zeros = torch.zeros(
            (batch_size, self.num_zeros_to_add),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        logits = torch.cat((embeddings, logits_zeros), dim=1).to(
            self.sequence_weights.dtype
        )
        logits = logits.reshape((batch_size, num_tokens, -1))

        with torch.no_grad():
            # Minibatch
            embeddings_list = []
            i = 0
            while i < batch_size:
                batch_logits = logits[i : i + self.minibatch_size, ...]
                batch_embeddings = torch.einsum(
                    "smd,bsm -> bsd", self.word_embeddings, batch_logits
                )
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
