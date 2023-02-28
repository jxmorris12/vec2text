from typing import Dict, Tuple

import torch
import transformers


# TODO: can we make this class a HF pretrained model so it works nicely with
# .push_to_hub(), etc.?
class InversionModel(torch.nn.Module):
    embedder: torch.nn.Module
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    embedding_transform: torch.nn.Module
    num_repeat_tokens: int

    def __init__(
            self,
            embedder: torch.nn.Module,
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
        self.embedding_transform = torch.nn.Linear(
            embedder_dim, encoder_hidden_dim
        )
        ######################################################

    def embed(
            self, 
            inputs: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert "embedder_input_ids" in inputs
        assert "embedder_attention_mask" in inputs
        assert not self.embedder.training
        with torch.no_grad():
            embeddings = self.embedder(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            ).pooler_output
        embeddings = self.embedding_transform(embeddings)
        inputs_embeds = embeddings.unsqueeze(dim=1).repeat((1, self.num_repeat_tokens, 1)) # TODO make this fancier/abstracted.
        attention_mask = torch.ones((inputs_embeds.shape[0], self.num_repeat_tokens), device=inputs_embeds.device)
        return inputs_embeds, attention_mask
    
    def generate(
            self,
            inputs: Dict[str, torch.Tensor],
            generation_kwargs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        inputs_embeds, attention_mask = self.embed(inputs)
        # TODO: implement mean-pooling so we can test BERT sentence
        # embeddings vs SimCSE vs Contriever etc fairly.
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
            # 
            # decoder_input_ids=inputs["input_ids"],
            # decoder_attention_mask=inputs["attention_mask"],
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