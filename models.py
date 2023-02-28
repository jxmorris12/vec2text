from typing import Dict, Tuple

import torch
import torch.nn as nn
import transformers


# TODO: can we make this class a HF pretrained model so it works nicely with
# .push_to_hub(), etc.?
class InversionModel(nn.Module):
    embedder: nn.Module
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    embedding_transform: nn.Module
    num_repeat_tokens: int

    def __init__(
            self,
            embedder: nn.Module,
            encoder_decoder: transformers.AutoModelForSeq2SeqLM,
            num_repeat_tokens: int,
        ):
        super().__init__()
        self.embedder = embedder
        self.encoder_decoder = encoder_decoder
        ######################################################
        self.num_repeat_tokens = num_repeat_tokens
        embedder_dim = self.embedder.config.hidden_size
        encoder_hidden_dim = self.encoder_decoder.config.hidden_size

        bottleneck_dim = 128
        self.embedding_transform = nn.Sequential(
            nn.Linear(embedder_dim, bottleneck_dim),
            nn.GELU(),
            # TODO dropout here?
            nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens)
        )
        ######################################################

    def embed(
            self, 
            inputs: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: implement mean-pooling so we can test BERT sentence
        # embeddings vs SimCSE vs Contriever etc fairly.
        assert "embedder_input_ids" in inputs
        assert "embedder_attention_mask" in inputs

        # TODO should we allow dropout from the embedding model?
        # assert not self.embedder.training
        with torch.no_grad():
            embeddings = self.embedder(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            ).pooler_output
        embeddings = self.embedding_transform(embeddings)
        batch_size = inputs["embedder_input_ids"].shape[0]
        # linear outputs a big embedding, reshape into a sequence of regular size embeddings.
        embeddings = embeddings.reshape((batch_size, self.num_repeat_tokens, -1))
        attention_mask = torch.ones((embeddings.shape[0], self.num_repeat_tokens), device=embeddings.device)
        return embeddings, attention_mask
    
    def generate(
            self,
            inputs: Dict[str, torch.Tensor],
            generation_kwargs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        inputs_embeds, attention_mask = self.embed(inputs)
        return self.encoder_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
    
    def forward(
            self, inputs: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
        inputs_embeds, attention_mask = self.embed(inputs)
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=inputs.get("labels", None),
        )


def load_embedder_and_tokenizer(name: str):
    # TODO make abstract/argparse for it etc.
    if name == "dpr":
        model = transformers.DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    elif name == "contriever":
        model = transformers.AutoModel.from_pretrained("facebook/contriever")
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/contriever")
    return model, tokenizer


def load_encoder_decoder(model_name: str) -> transformers.AutoModelForSeq2SeqLM:
    return transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name) # for testing