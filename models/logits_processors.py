from typing import Dict
import torch
import transformers
from models import InversionModel


class ContrastiveLogitsProcessor(transformers.LogitsProcessor):
    model: InversionModel
    alpha: float
    gamma: float
    hypothesis_embedding: torch.Tensor

    def __init__(
        self,
        model: InversionModel,
        alpha: float,
        gamma: float,
        inputs: Dict[str, torch.Tensor],
    ):
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.hypothesis_embedding = self._get_hypothesis_embedding(inputs=inputs)
        print(f"ContrastiveLogitsProcessor alpha={alpha} gamma={gamma}")
    
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
    
    def _get_hypothesis_embedding(self, inputs: Dict[str, torch.Tensor]) -> None:
        batch_size, seq_length = inputs["embedder_input_ids"].shape

        with torch.no_grad():
            embedding = self.model.call_embedding_model(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            )
        #
        hypotheses = self.model.generate(
            inputs=inputs,
            generation_kwargs={
                "max_length": seq_length,
                "early_stopping": True,
                "num_beams": 1,
                "do_sample": False,
            },
        )
        eos_token_id = self.model.embedder_tokenizer.eos_token_id
        eos_tokens = (
            torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
            * eos_token_id
        )
        hypotheses_with_eos = torch.cat((hypotheses[:, 1:], eos_tokens), dim=1)
        hypothesis_attention_mask = torch.ones_like(
            hypotheses_with_eos, device=self.device
        )
        with torch.no_grad():
            hypothesis_embedding = self.model.call_embedding_model(
                input_ids=hypotheses_with_eos,
                attention_mask=hypothesis_attention_mask,
            )
        return hypothesis_embedding

    def __call__(
        self, input_ids: torch.LongTensor, next_token_logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        assert len(input_ids) == len(self.hypothesis_embedding) == len(next_token_logits)
        batch_size, vocab_size = next_token_logits.shape
        
        with torch.no_grad():
            bad_outputs = self.model(
                embedder_input_ids=None,
                embedder_attention_mask=None,
                frozen_embeddings=self.hypothesis_embedding,
                decoder_input_ids=input_ids,
            )
        bad_token_logits = bad_outputs.logits[:, -1] * self.gamma
        #
        diff_logits = next_token_logits.log_softmax(
            1
        ) - bad_token_logits.log_softmax(1)
        #
        next_token_probs = next_token_logits.softmax(-1)
        V_mask = (
            next_token_probs
            >= self.alpha * next_token_probs.max(dim=1, keepdim=True).values
        )
        diff_logits = torch.where(V_mask, diff_logits, diff_logits - 10**10)
        return diff_logits
