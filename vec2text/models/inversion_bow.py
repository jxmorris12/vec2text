from typing import Dict, Optional

import torch
import torch.nn as nn
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.model_utils import (
    load_embedder_and_tokenizer,
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

        self.embedder = embedder
        self.tokenizer = embedder_tokenizer
        self.encoder = embedder
        self.embedder_tokenizer = embedder_tokenizer
        self.vocab_size = embedder_tokenizer.vocab_size

        self.mlp = nn.Sequential(
            nn.Linear(self.d_embedder, self.d_embedder*8),
            nn.ReLU(),
            nn.Linear(self.d_embedder*8, self.vocab_size),
        )
        
        self.config = config


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
        with torch.no_grad():
            logits = self.forward(**inputs)["logits"]
        max_length = generation_kwargs["max_length"]
        return logits.topk(max_length, dim=1).indices

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        sentences = [self.embedder_tokenizer.decode(ids) for ids in input_ids]
        embeddings = self.embedder.encode(sentences)
        return embeddings


    def bow_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        vocab_size = self.vocab_size
        # vocab = torch.arange(vocab_size, dtype=labels.dtype, device=labels.device)

        batch_labels = torch.zeros_like(logits, dtype=torch.float32)
        for i, batch in enumerate(labels):
            length = 128 # Assume the max generated length is 128
            for j, label in enumerate(batch):
                if label == 0: continue
                if label == -100: break
                batch_labels[i, label] = length-j

        return torch.nn.functional.cross_entropy(
            logits, batch_labels
        )

    def forward(
        self,
        embedder_input_ids: torch.Tensor = None,
        embedder_attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if frozen_embeddings is None:
            with torch.no_grad():
                embedding = self.call_embedding_model(
                    input_ids=embedder_input_ids, attention_mask=embedder_attention_mask
                )
        else:
            embedding = frozen_embeddings
        batch_size = embedding.shape[0]
        logits = self.mlp(embedding)
        outputs = {"logits": logits}  # hf trainer output format
        if labels is not None:
            labels = torch.cat(
                (
                    torch.zeros(
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
