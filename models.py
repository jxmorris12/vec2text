from typing import Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import transformers
from transformers import LogitsProcessor, LogitsProcessorList

from utils import embed_all_tokens

MODEL_NAMES =  ["bert", "contriever", "dpr", "gtr_base", "gtr_base__random_init", "gtr_large", "ance_tele", "dpr_st", "gtr_base_st", "paraphrase-distilroberta"]
FREEZE_STRATEGIES = ["decoder", "encoder_and_decoder", "encoder", "none"]
EMBEDDING_TRANSFORM_STRATEGIES = ["repeat", "nearest_neighbors"]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def disable_dropout(model: nn.Module):
    dropout_modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in dropout_modules: m.p = 0.0
    print(f'Disabled {len(dropout_modules)} dropout modules from model type {type(model)}')


def freeze_params(model: nn.Module):
    total_num_params = 0
    for name, params in model.named_parameters():
        params.requires_grad = False
        total_num_params += params.numel()

    print(f'Froze {total_num_params} params from model type {type(model)}')


def mean_pool(outputs: transformers.modeling_outputs.BaseModelOutput, attention_mask: torch.Tensor) -> torch.Tensor:
    if hasattr(outputs, 'pooler_output') and (outputs.pooler_output is not None):
        return outputs.pooler_output
    B, S, D = outputs.last_hidden_state.shape
    unmasked_outputs = (outputs.last_hidden_state * attention_mask[..., None])
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def max_pool(outputs: transformers.modeling_outputs.BaseModelOutput, attention_mask: torch.Tensor) -> torch.Tensor:
    if hasattr(outputs, 'pooler_output') and (outputs.pooler_output is not None):
        return outputs.pooler_output
    B, S, D = outputs.last_hidden_state.shape
    unmasked_outputs = (outputs.last_hidden_state * attention_mask[..., None]) # Set masked activations to zero
    pooled_outputs = unmasked_outputs.max(dim=1).values
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def stack_pool(outputs: transformers.modeling_outputs.BaseModelOutput, attention_mask: torch.Tensor) -> torch.Tensor:
    if hasattr(outputs, 'pooler_output') and (outputs.pooler_output is not None):
        return outputs.pooler_output
    B, S, D = outputs.last_hidden_state.shape
    unmasked_outputs = (outputs.last_hidden_state * attention_mask[..., None])
    pooled_outputs = unmasked_outputs.reshape((B, S*D)) # stack along seq length
    assert pooled_outputs.shape == (B, S*D)
    return pooled_outputs


class TokenLogitsProcessor(LogitsProcessor):
    embeddings: torch.Tensor
    embedded_tokens: torch.Tensor
    alpha: float
    def __init__(self, embeddings: torch.Tensor, embedded_tokens: torch.Tensor, alpha: float, tokenizer: transformers.PreTrainedTokenizer):
        self.embeddings = embeddings
        self.embedded_tokens = embedded_tokens
        self.alpha = alpha
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        assert len(input_ids) == len(self.embeddings) == len(scores)
        
        # Compute token scores based on the currently generated tokens
        if input_ids.shape[1] == 1:
            # first token, nothing to embed yet.
            current_embedding = torch.zeros_like(self.embeddings, device=self.embeddings.device)
        else:
            # subtract the embedding of the currently embedded tokens.
            current_token_embeddings = self.embedded_tokens[input_ids]
            current_embedding = current_token_embeddings[:, 1:, :].mean(dim=1)
        token_scores = (self.embeddings - current_embedding) @ self.embedded_tokens.T

        # Softmax the actual model scores for next tokens.
        batch_size = scores.shape[0]
        scores = scores.log_softmax(dim=-1)

        # Weird detail: have to add zeros for 'fake' tokens that don't receive scores but are in the vocabulary.
        num_zeros = scores.shape[1] - token_scores.shape[1]
        if num_zeros > 0:
            zeros_to_add = torch.tensor([[1e-10]], device=device).repeat(batch_size, num_zeros)
            token_scores = torch.cat((token_scores, zeros_to_add), dim=1)
        
        # Second weird detail: Need to set negative cosine similarities to zero before log.
        # These terms have essentially a probability of zero anyway.
        token_scores = token_scores.clamp(min=1e-10)
        assert not token_scores.isnan().any()
        assert not token_scores.isinf().any()

        # Last weird detail: we normalize.
        token_scores /= token_scores.max(dim=-1, keepdim=True).values

        # convert logits to log-probs.
        token_scores = token_scores.log()

        return (scores * self.alpha) + (token_scores * (1.0 - self.alpha))


# TODO: can we make this class a HF pretrained model so it works nicely with
# .push_to_hub(), etc.?
# TODO: Need config to subclass transformers.PreTrainedModel.
class InversionModel(nn.Module):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """
    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer # embedder's tokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    encoder_decoder_lora: bool # Whether to use LoRA for the encoder-decoder model
    tokenizer: transformers.PreTrainedTokenizer # encoder_decoder's tokenizer
    embedding_transform: nn.Module # Module that transformers embedder output into encoder-decoder input
    bottleneck_dim: int # Bottleneck dimension for embedding_transform
    num_repeat_tokens: int # Sequence length for repeating embedder embedding for encoder-decoder input
    embedder_dim: int # Hidden dimension of embedding model
    embedder_no_grad: bool # Disable gradients for embedding model
    embedder_fake_with_zeros: bool # Whether to just provide zeros as input for encoder-decoder (unconditional)
    embedding_transform_strategy: str # Way to transform bottleneck embedding into input for encoder-decoder
    use_frozen_embeddings_as_input: bool # Whether to train/evaluate on frozen embeddings 
    use_frozen_whitened_embeddings_as_input: bool # Whether to train/evaluate on frozen and whitened embeddings
    token_decode_alpha: Optional[float] # Alpha to apply to token embedding sims during decoding.
    embedding_batch_norm: Optional[nn.Module] # Optionally apply batchnorm to batches of embeddings
    embedded_tokens: torch.Tensor # used for decoding

    def __init__(
            self,
            embedder: nn.Module,
            embedder_tokenizer: transformers.PreTrainedTokenizer,
            encoder_decoder: transformers.AutoModelForSeq2SeqLM,
            tokenizer: transformers.PreTrainedTokenizer,
            num_repeat_tokens: int,
            embedder_no_grad: bool,
            freeze_strategy: str = "none",
            embedder_fake_with_zeros: bool = False,
            encoder_dropout_disabled: bool = False,
            decoder_dropout_disabled: bool = False,
            use_frozen_embeddings_as_input: bool = False,
            use_frozen_whitened_embeddings_as_input: bool = False,
            use_embedding_batch_norm: bool = False,
            encoder_decoder_lora: bool = False, # TODO: thoroughly test this option.
            embedding_transform_strategy: str = "repeat",
            bottleneck_dim: int = 768, # 128,
            token_decode_alpha: Optional[float] = None,
        ):
        super().__init__()
        self.embedder = embedder
        self.encoder_decoder = encoder_decoder
        if encoder_decoder_lora:
            from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
            print("Initializing LORA model with config:", peft_config)
            self.encoder_decoder = get_peft_model(self.encoder_decoder, peft_config)
        ######################################################
        self.num_repeat_tokens = num_repeat_tokens
        # if use_frozen_embeddings_as_input:
            # pass
            # self.embedder_dim = 512
            # temp hack to set fixed sentence embedding size to 512.
            # TODO do this in a smarter way (figure it out from data? or make it an arg.)
        if isinstance(self.embedder, SentenceTransformer):
            self.embedder_dim = self.embedder.get_sentence_embedding_dimension()
        else:
            self.embedder_dim = self.embedder.config.hidden_size
            
        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedder_no_grad = embedder_no_grad
        assert not (use_frozen_embeddings_as_input and use_frozen_whitened_embeddings_as_input)
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        self.use_frozen_whitened_embeddings_as_input = use_frozen_whitened_embeddings_as_input
        self.bottleneck_dim = bottleneck_dim
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim*1, bottleneck_dim),
            nn.GELU(),   # TODO consider dropout or normalization here.
            nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens)
        )
        if use_embedding_batch_norm:
            print('Embedding batch norm enabled')
            self.embedding_batch_norm = torch.nn.BatchNorm1d(num_features=self.embedder_dim)
        else:
            self.embedding_batch_norm = None
        if encoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.encoder)
        if decoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.decoder)
            disable_dropout(self.encoder_decoder.lm_head)
        ######################################################
        self.tokenizer = tokenizer
        self.embedder_tokenizer = embedder_tokenizer
        self.freeze(freeze_strategy=freeze_strategy)
        self.embedder_fake_with_zeros = embedder_fake_with_zeros
        assert embedding_transform_strategy in EMBEDDING_TRANSFORM_STRATEGIES
        self.embedding_transform_strategy = embedding_transform_strategy
        self.token_decode_alpha = token_decode_alpha
        if token_decode_alpha is not None:
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
    
    @property
    def embedder_device(self) -> torch.device:
        return next(self.embedder.parameters()).device
    
    def call_embedding_model(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None, # not used
        ) -> torch.Tensor:

        if self.embedder_fake_with_zeros:
            batch_size = input_ids.shape[0]
            return torch.zeros((batch_size, self.embedder_dim), dtype=torch.float32, device=self.embedder_device)

        if self.embedder_no_grad:
            self.embedder.eval()
        if isinstance(self.embedder, SentenceTransformer):
            # sentence-transformers is kind of really annoying
            model_output = self.embedder({ 'input_ids': input_ids, 'attention_mask': attention_mask})
            embeddings = model_output['sentence_embedding']
        else:
            model_output = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = mean_pool(model_output, attention_mask)
            # embeddings = max_pool(model_output, attention_mask)
            # embeddings = stack_pool(model_output, attention_mask)
        return embeddings

    def embed(
            self, 
            embedder_input_ids: torch.Tensor,
            embedder_attention_mask: torch.Tensor,
            frozen_embeddings: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.use_frozen_embeddings_as_input:
            assert frozen_embeddings is not None, "specified to train on frozen embeddings but none were provided"
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 2 # batch by d
        elif self.embedder_no_grad:
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
        
        if self.embedding_batch_norm:
            embeddings = self.embedding_batch_norm(embeddings)
        
        if self.embedding_transform_strategy == "repeat":
            embeddings = self.embedding_transform(embeddings)
            batch_size = embedder_input_ids.shape[0]
            # linear outputs a big embedding, reshape into a sequence of regular size embeddings.
            embeddings = embeddings.reshape((batch_size, self.num_repeat_tokens, -1))
        elif self.embedding_transform_strategy == "nearest_neighbors":
            # TODO
            raise NotImplementedError()
        else:
            raise ValueError(f'unknown embedding transformation strategy {self.embedding_transform_strategy}')
        attention_mask = torch.ones((embeddings.shape[0], self.num_repeat_tokens), device=embeddings.device)
        return embeddings, attention_mask
    
    def generate(
            self,
            inputs: Dict[str, torch.Tensor],
            generation_kwargs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:

        generation_kwargs['max_length'] = inputs.get('input_ids', inputs['embedder_input_ids']).shape[1]

        if self.token_decode_alpha is not None:
            ########################################################################
            # TODO: optimize this code to avoid re-embedding.
            initial_embeddings = self.call_embedding_model(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            )
            initial_embeddings /= initial_embeddings.norm(p=2, dim=1, keepdim=True)
            embedded_tokens_logits_processor = TokenLogitsProcessor(
                embeddings=initial_embeddings,
                embedded_tokens=self.embedded_tokens,
                alpha=self.token_decode_alpha,
                tokenizer=self.embedder_tokenizer,
            )
            generation_kwargs["logits_processor"] = LogitsProcessorList([
                embedded_tokens_logits_processor,
            ])
            generation_kwargs["renormalize_logits"] = True
            ########################################################################
        
        inputs_embeds, attention_mask = self.embed(
            embedder_input_ids=inputs["embedder_input_ids"],
            embedder_attention_mask=inputs["embedder_attention_mask"],
            frozen_embeddings=inputs.get("frozen_embeddings"),
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
            frozen_embeddings: Optional[torch.Tensor] = None,
            **kwargs
        ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask, embedder_token_type_ids
        inputs_embeds, attention_mask = self.embed(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=frozen_embeddings,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )


def load_embedder_and_tokenizer(name: str):
    # TODO make abstract/argparse for it etc.
    model_kwargs = {
        "low_cpu_mem_usage": True,
    }

    if name == "dpr":
        # model = SentenceTransformer("sentence-transformers/facebook-dpr-question_encoder-multiset-base")
        model = transformers.DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    elif name == "dpr_st":
        # TODO figure out why model w/ sentence transformers gives different results.
        model = SentenceTransformer("sentence-transformers/facebook-dpr-question_encoder-multiset-base")
        tokenizer = model.tokenizer
    elif name == "contriever":
        model = transformers.AutoModel.from_pretrained("facebook/contriever", **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/contriever")
    elif name == "bert":
        model = transformers.AutoModel.from_pretrained("bert-base-uncased", **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    elif name == "gtr_base":
        model = transformers.AutoModel.from_pretrained("sentence-transformers/gtr-t5-base", **model_kwargs).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    elif name == "gtr_base__random_init":
        config = transformers.AutoConfig.from_pretrained("sentence-transformers/gtr-t5-base")
        model = transformers.AutoModel.from_config(config, **model_kwargs).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    elif name == "gtr_base_st":
        # TODO figure out why model w/ sentence transformers gives different results.
        model = SentenceTransformer("sentence-transformers/gtr-t5-base")
        tokenizer = model.tokenizer
    elif name == "gtr_large":
        model = SentenceTransformer("sentence-transformers/gtr-t5-large")
        tokenizer = model.tokenizer
    elif name == "ance_tele":
        model = transformers.AutoModel.from_pretrained("OpenMatch/ance-tele_nq_psg-encoder", **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained("OpenMatch/ance-tele_nq_psg-encoder")
    elif name == "paraphrase-distilroberta":
        model = transformers.AutoModel.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v1", **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-distilroberta-base-v1")
    else:
        raise ValueError(f'unknown embedder {name}')
    
    model = torch.compile(model)
    
    return model, tokenizer


class FutureEmbeddingPredictor(nn.Module):
    """Given an embedding prefix, predicts the final embedding of the completion
    of that prefix.
    """
    embedder: nn.Module # embeds a prefix
    encoder: nn.Module # encodes prefix, outputs encoding
    mlp: nn.Module # maps encoding to 
    def __init__(self, embedder, encoder):
        self.embedder = embedder
        self.encoder = encdoer
        embedder_hidden_size = self.embedder.config.hidden_size
        encoder_hidden_size = self.encoder.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.GELU(),
            nn.Linear(encoder_hidden_size, embedder_hidden_size)
        )


def load_encoder_decoder(model_name: str) -> transformers.AutoModelForSeq2SeqLM:
    model_kwargs = {
        "low_cpu_mem_usage": True,
    }
    return transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
