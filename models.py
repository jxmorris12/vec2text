from typing import Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import transformers
from transformers.generation_logits_process import LogitsProcessor,LogitsProcessorList

from utils import embed_all_tokens

MODEL_NAMES =  ["contriever", "dpr", "gtr_base", "gtr_large", "ance_tele"]
FREEZE_STRATEGIES = ["decoder", "encoder_and_decoder", "encoder", "none"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def freeze_params(model: nn.Module):
    total_num_params = 0
    for name, params in model.named_parameters():
        params.requires_grad = False
        total_num_params += params.numel()

    print(f'Froze {total_num_params} params from model type {type(model)}')


def mean_pool(outputs: transformers.modeling_outputs.BaseModelOutput, attention_mask: torch.Tensor) -> torch.Tensor:
    if outputs.pooler_output is not None:
        return outputs.pooler_output
    B, S, D = outputs.last_hidden_state.shape
    unmasked_outputs = (outputs.last_hidden_state * attention_mask[..., None])
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


class TokenLogitsProcessor(LogitsProcessor):
    embedded_tokens: torch.Tensor
    alpha: float
    def __init__(self, token_scores: torch.Tensor, alpha: float):
        self.token_scores = token_scores
        self.alpha = alpha

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        #   pad to a length of 2
        num_zeros =  scores.shape[1] - self.token_scores.shape[1]
        token_scores = torch.cat((self.token_scores, torch.tensor([[0]], device='cuda').repeat(128, 28)), dim=1)
        #  scores.shape - torch.Size([128, 32128])
        #  self.embedded_tokens.shape - torch.Size([30522, 768])
        print('alpha =', self.alpha)
        return scores + (token_scores * self.alpha)


# TODO: can we make this class a HF pretrained model so it works nicely with
# .push_to_hub(), etc.?
class InversionModel(nn.Module):
    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer # embedder's tokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    tokenizer: transformers.PreTrainedTokenizer # encoder_decoder's tokenizer
    embedding_transform: nn.Module
    num_repeat_tokens: int
    embedder_no_grad: bool
    bottleneck_dim: int
    token_decode_alpha: float # Alpha to apply to token embedding sims during decoding.
    embedded_tokens: torch.Tensor

    def __init__(
            self,
            embedder: nn.Module,
            embedder_tokenizer: transformers.PreTrainedTokenizer,
            encoder_decoder: transformers.AutoModelForSeq2SeqLM,
            tokenizer: transformers.PreTrainedTokenizer,
            num_repeat_tokens: int,
            embedder_no_grad: bool,
            freeze_strategy: str = "none",
            bottleneck_dim: int = 768,
            token_decode_alpha: float = 0.0,
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
        self.tokenizer = tokenizer
        self.embedder_tokenizer = embedder_tokenizer
        self.freeze(freeze_strategy=freeze_strategy)
        self.token_decode_alpha = token_decode_alpha
        if token_decode_alpha > 0:
            assert embedder_tokenizer is not None
            self.embedded_tokens = embed_all_tokens(self, embedder_tokenizer).to(device)
        else:
            self.embedded_tokens = None
    
    def _freeze_encoder(self):
        freeze_params(self.encoder_decoder.encoder)
    
    def _freeze_decoder(self):
        # github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L1229-L1231
        freeze_params(self.encoder_decoder.decoder)
        freeze_params(self.encoder_decoder.lm_head)
    
    def freeze(self, freeze_strategy: str):
        assert freeze_strategy in FREEZE_STRATEGIES

        if freeze_strategy == "decoder":
            self._freeze_decoder()
        elif freeze_strategy == "encoder":
            self._freeze_encoder()
        elif freeze_strategy == "encoder_and_decoder":
            self._freeze_encoder()
            self._freeze_decoder()
            # in this case, freeze embeddings too
            freeze_params(self.encoder_decoder.shared)
        elif freeze_strategy == "none":
            pass
        else:
            raise ValueError(f'invalid freezing strategy {freeze_strategy}')
    
    def call_embedding_model(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None, # not used
        ) -> torch.Tensor:
        self.embedder.eval()
        if isinstance(self.embedder, SentenceTransformer):
            # sentence-transformers is kind of really annoying 
            model_output = self.embedder({ 'input_ids': input_ids, 'attention_mask': attention_mask})
            embeddings = model_output['sentence_embedding']
        else:
            model_output = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = mean_pool(model_output, attention_mask)
        return embeddings

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
                embeddings = self.call_embedding_model(
                    input_ids=embedder_input_ids,
                    attention_mask=embedder_attention_mask,
                )
        else:
            embeddings = self.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            )
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

        if self.token_decode_alpha > 0:
            ########################################################################
            # TODO: optimize to avoid re-embedding.
            initial_embeddings = self.call_embedding_model(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            )
            token_scores = initial_embeddings @ self.embedded_tokens.T
            embedded_tokens_logits_processor = TokenLogitsProcessor(
                token_scores=token_scores,
                alpha=self.token_decode_alpha,
            )
            generation_kwargs["logits_processor"] = LogitsProcessorList([
                embedded_tokens_logits_processor,
            ])
            ########################################################################
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
    elif name == "gtr_large":
        model = SentenceTransformer("sentence-transformers/gtr-t5-large")
        tokenizer = model.tokenizer
    elif name == "ance_tele":
        model = transformers.AutoModel.from_pretrained("OpenMatch/ance-tele_nq_psg-encoder")
        tokenizer = transformers.AutoTokenizer.from_pretrained("OpenMatch/ance-tele_nq_psg-encoder")
    else:
        raise ValueError(f'unknown embedder {name}')
    return model, tokenizer


def load_encoder_decoder(model_name: str) -> transformers.AutoModelForSeq2SeqLM:
    return transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name) # for testing