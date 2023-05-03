from typing import Dict, Optional
import torch
import transformers

from .model_utils import mean_pool


class JointEmbeddingTextEncoder(torch.nn.Module):
    # TODO: Rename? It's not just an encoder anymore so this name doesn't fit
    # super well
    """Embeds text and concats with a provided embedding."""

    encoder_decoder: transformers.PreTrainedModel

    def __init__(self, encoder_decoder: transformers.PreTrainedModel):
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.in_projection = torch.nn.Sequential(
            torch.nn.Linear(768, 768), torch.nn.GELU(), torch.nn.Linear(768, 768)
        )
        
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        if "max_length" not in generation_kwargs:
            generation_kwargs["max_length"] = (
                inputs["input_ids"].shape[1] + 1
            )
        
        batch_size, seq_length = inputs["input_ids"].shape
        assert embedding.shape == (batch_size, 768)
        embedding = self.in_projection(embedding)
        inputs_embeds = self.encoder_decoder.encoder.embed_tokens(input_ids)
        #
        ones = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
        # TODO: support & ablate concatenation methods.
        inputs_embeds = torch.cat((embedding[:, None, :], inputs_embeds), dim=1)
        attention_mask = torch.cat((ones, attention_mask), dim=1)

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
        embedding: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        batch_size, seq_length = input_ids.shape
        assert embedding.shape == (batch_size, 768)
        embedding = self.in_projection(embedding)
        inputs_embeds = self.encoder_decoder.encoder.embed_tokens(input_ids)
        #
        ones = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
        # TODO: support & ablate concatenation methods.
        inputs_embeds = torch.cat((embedding[:, None, :], inputs_embeds), dim=1)
        attention_mask = torch.cat((ones, attention_mask), dim=1)
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )