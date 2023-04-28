import torch
import torch.nn as nn


class PrefixReranker(nn.Module):
    """Given an embedding prefix, predicts the final embedding of the completion
    of that prefix.
    """

    prefix_embedder: nn.Module  # embeds a prefix
    embedding_projection: nn.Module  # projects sentence embedding to same
    # space as a prefix embedding

    def __init__(
        self,
        prefix_embedder: nn.Module,
    ):
        super().__init__()
        self.prefix_embedder = prefix_embedder
        self.embedding_projection = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Linear(2048, 768),
        )
        self.score = nn.Linear(768, 1)

    def score_prefix_and_embedding(
        self,
        prefix_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, prefix_length = prefix_ids.shape
        embeddings = self.embedding_projection(embeddings)
        inputs_embeds = self.prefix_embedder.encoder.embed_tokens(prefix_ids)
        all_embeddings = torch.cat((embeddings[:, None], inputs_embeds), dim=1)
        ones = torch.ones((batch_size, 1), device=attention_mask.device)
        attention_mask = torch.cat((ones, attention_mask), dim=1)
        model_output = self.prefix_embedder(
            inputs_embeds=all_embeddings, attention_mask=attention_mask
        )
        hidden_state = model_output.last_hidden_state
        output_embeddings = hidden_state[:, 0, :]
        # output_embeddings = mean_pool(hidden_state, attention_mask)
        scores = self.score(output_embeddings)
        scores = scores.flatten()  # (batch_size, 1) -> (batch_size,)
        return scores
