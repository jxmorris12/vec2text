from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers

from sentence_transformers import SentenceTransformer

def mean_pool(outputs: transformers.modeling_outputs.BaseModelOutput, attention_mask: torch.Tensor) -> torch.Tensor:
    if outputs.pooler_output is not None:
        return outputs.pooler_output
    B, S, D = outputs.last_hidden_state.shape
    unmasked_outputs = (outputs.last_hidden_state * attention_mask[..., None])
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


# TODO: can we make this class a HF pretrained model so it works nicely with
# .push_to_hub(), etc.?
class InversionModel(nn.Module):
    embedder: nn.Module
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    embedding_transform: nn.Module
    num_repeat_tokens: int
    embedder_no_grad: bool
    bottleneck_dim: int

    def __init__(
            self,
            embedder: nn.Module,
            encoder_decoder: transformers.AutoModelForSeq2SeqLM,
            num_repeat_tokens: int,
            embedder_no_grad: bool,
            bottleneck_dim: int = 128,
        ):
        super().__init__()
        self.embedder = embedder
        self.encoder_decoder = encoder_decoder
        ######################################################
        self.num_repeat_tokens = num_repeat_tokens
        if isinstance(self.embedder, SentenceTransformer):
            hidden_size = self.embedder.get_sentence_embedding_dimension()
        else:
            hidden_size = self.embedder.config.hidden_size
        embedder_dim = hidden_size
        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedder_no_grad = embedder_no_grad
        self.bottleneck_dim = bottleneck_dim
        self.embedding_transform = nn.Sequential(
            nn.Linear(embedder_dim, bottleneck_dim),
            nn.GELU(),
            # TODO dropout here?
            nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens)
        )
        ######################################################
    
    def _call_embedding_model(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
        if isinstance(self.embedder, SentenceTransformer):
            # really annoying 
            return self.embedder({ 'input_ids': input_ids, 'attention_mask': attention_mask})
        else:
            return self.embedder(input_ids=input_ids, attention_mask=attention_mask)

    def embed(
            self, 
            embedder_input_ids: torch.Tensor,
            embedder_attention_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: implement mean-pooling so we can test BERT sentence
        # embeddings vs SimCSE vs Contriever etc fairly.
        # TODO: should we allow dropout from the embedding model?
        # assert not self.embedder.training
        if self.embedder_no_grad:
            with torch.no_grad():
                model_output = self._call_embedding_model(
                    input_ids=embedder_input_ids,
                    attention_mask=embedder_attention_mask,
                )
        else:
            model_output = self._call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            )
        
        if isinstance(self.embedder, SentenceTransformer):
            embeddings = model_output['sentence_embedding']
        else:
            embeddings = mean_pool(model_output, embedder_attention_mask)
        embeddings = self.embedding_transform(embeddings)
        batch_size = embedder_input_ids.shape[0]
        # linear outputs a big embedding, reshape into a sequence of regular size embeddings.
        embeddings = embeddings.reshape((batch_size, self.num_repeat_tokens, -1))
        attention_mask = torch.ones((embeddings.shape[0], self.num_repeat_tokens), device=embeddings.device)
        return embeddings, attention_mask
    
    def generate(
            self,
            inputs: Dict[str, torch.Tensor],
            generation_kwargs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        inputs_embeds, attention_mask = self.embed(
            embedder_input_ids=inputs["embedder_input_ids"],
            embedder_attention_mask=inputs["embedder_attention_mask"],
        )
        return self.encoder_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
    
    def forward(
            self, 
            embedder_input_ids: torch.Tensor,
            embedder_attention_mask: torch.Tensor,
            # embedder_token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            **kwargs
        ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask, embedder_token_type_ids
        inputs_embeds, attention_mask = self.embed(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )


def load_embedder_and_tokenizer(name: str):
    # TODO make abstract/argparse for it etc.
    if name == "dpr":
        model = transformers.DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    elif name == "contriever":
        model = transformers.AutoModel.from_pretrained("facebook/contriever")
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/contriever")
    elif name == "gtr_base":
        model = SentenceTransformer("sentence-transformers/gtr-t5-base")
        tokenizer = model.tokenizer
    elif name == "ance_tele":
        model = transformers.AutoModel.from_pretrained("OpenMatch/ance-tele_nq_psg-encoder")
        tokenizer = transformers.AutoTokenizer.from_pretrained("OpenMatch/ance-tele_nq_psg-encoder")
    else:
        raise ValueError(f'unknown embedder {name}')
    return model, tokenizer


def load_encoder_decoder(model_name: str) -> transformers.AutoModelForSeq2SeqLM:
    return transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name) # for testing