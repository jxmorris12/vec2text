import torch
import transformers

from .model_utils import mean_pool


class JointEmbeddingTextEncoder(torch.nn.Module):
    """Embeds text and concats with a provided embedding."""

    encoder: transformers.PreTrainedModel

    def __init__(self, encoder: transformers.PreTrainedModel):
        super().__init__()
        self.encoder = encoder
        self.in_projection = torch.nn.Sequential(
            torch.nn.Linear(768, 768), torch.nn.GELU(), torch.nn.Linear(768, 768)
        )
        self.out_projection = torch.nn.Sequential(
            torch.nn.Linear(768, 768), torch.nn.GELU(), torch.nn.Linear(768, 768)
        )

    def forward(
        self,
        embedding: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        batch_size, seq_length = input_ids.shape
        assert embedding.shape == (batch_size, 768)
        embedding = self.in_projection(embedding)
        inputs_embeds = self.encoder.encoder.embed_tokens(input_ids)
        #
        ones = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
        # TODO: support & ablate concatenation methods.
        inputs_embeds = torch.cat((embedding[:, None, :], inputs_embeds), dim=1)
        attention_mask = torch.cat((ones, attention_mask), dim=1)
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden_state = outputs.last_hidden_state
        # TODO consider different pooling strategies? At least first-token.
        pooled = mean_pool(hidden_state, attention_mask)
        return self.out_projection(pooled)
