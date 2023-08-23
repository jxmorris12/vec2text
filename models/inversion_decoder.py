import copy
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import tqdm
import transformers
from sentence_transformers import SentenceTransformer

from utils import embed_all_tokens, embed_api

from .model_utils import (
    EMBEDDING_TRANSFORM_STRATEGIES,
    FREEZE_STRATEGIES,
    device,
    disable_dropout,
    freeze_params,
    mean_pool,
)

logger = logging.getLogger(__name__)


# TODO: can we make this class a HF pretrained model so it works nicely with
# .push_to_hub(), etc.?
# TODO: Need config to subclass transformers.PreTrainedModel.
class InversionDecoderModel(InversionModel):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """

    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer  # embedder's tokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    encoder_decoder_lora: bool  # Whether to use LoRA for the encoder-decoder model
    tokenizer: transformers.PreTrainedTokenizer  # encoder_decoder's tokenizer
    embedding_transform: nn.Module  # Module that transformers embedder output into encoder-decoder input
    bottleneck_dim: int  # Bottleneck dimension for embedding_transform
    num_repeat_tokens: int  # Sequence length for repeating embedder embedding for encoder-decoder input
    embedder_dim: int  # Hidden dimension of embedding model
    embedder_no_grad: bool  # Disable gradients for embedding model
    embedder_fake_with_zeros: bool  # Whether to just provide zeros as input for encoder-decoder (unconditional)
    embedding_transform_strategy: str  # Way to transform bottleneck embedding into input for encoder-decoder
    use_frozen_embeddings_as_input: bool  # Whether to train/evaluate on frozen embeddings
    whiten_embeddings: bool  # Preprocess all embeddings using 'whitening'
    embedded_tokens: torch.Tensor  # used for decoding
    embedder_model_api: Optional[str]

    def __init__(
        self,
        embedder: nn.Module,
        embedder_tokenizer: transformers.PreTrainedTokenizer,
        decoder: transformers.AutoModelForSeq2SeqLM,
        tokenizer: transformers.PreTrainedTokenizer,
        num_repeat_tokens: int,
        embedder_no_grad: bool,
        freeze_strategy: str = "none",
        embedder_model_api: Optional[str] = None,
        embedder_fake_with_zeros: bool = False,
        encoder_dropout_disabled: bool = False,
        decoder_dropout_disabled: bool = False,
        use_frozen_embeddings_as_input: bool = False,
        whiten_embeddings: bool = False,
        embedding_transform_strategy: str = "repeat",
        bottleneck_dim: int = 768,  # 128,
        token_decode_alpha: Optional[float] = None,
        embeddings_from_layer_n: Optional[int] = None,
    ):
        super().__init__()
        self.embedder = embedder
        self.decoder = decoder  # .to_bettertransformer()
        self.num_repeat_tokens = num_repeat_tokens

        if embedder_model_api:
            assert use_frozen_embeddings_as_input, "must precompute embeddings w/ api"
            # Hard-code OpenAI embedding dim
            self.embedder_dim = 1536
            bottleneck_dim = 1536
        # elif use_frozen_embeddings_as_input:
        #     # temp hack to set fixed sentence embedding size to 512 for luar.
        #     # TODO do this in a smarter way (figure it out from data? or make it an arg.)
        #     self.embedder_dim = 512
        elif isinstance(self.embedder, SentenceTransformer):
            self.embedder_dim = self.embedder.get_sentence_embedding_dimension()
        else:
            self.embedder_dim = self.embedder.config.hidden_size

        encoder_hidden_dim = self.decoder.config.hidden_size
        self.embedder_no_grad = embedder_no_grad
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        self.whiten_embeddings = whiten_embeddings
        self.bottleneck_dim = bottleneck_dim
        self.embedding_transform = nn.Linear(self.embedder_dim, decoder_hidden_dim)
        ######################################################
        self.tokenizer = tokenizer
        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        self.freeze(freeze_strategy=freeze_strategy)
        self.embedder_fake_with_zeros = embedder_fake_with_zeros
        assert embedding_transform_strategy in EMBEDDING_TRANSFORM_STRATEGIES
        self.embedding_transform_strategy = "repeat"  # "none" # "repeat"
        self.token_decode_alpha = token_decode_alpha
        if token_decode_alpha is not None:
            assert embedder_tokenizer is not None
            self.embedded_tokens = embed_all_tokens(self, embedder_tokenizer).to(device)
        else:
            self.embedded_tokens = None
        self.embeddings_from_layer_n = embeddings_from_layer_n
        self.noise_level = 0

    def embed_and_project(
        self,
        embedder_input_ids: Optional[torch.Tensor],
        embedder_attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("** embed_and_project")
        assert not ((embedder_input_ids is None) and (frozen_embeddings is None))
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 2  # batch by d
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
        embeddings = self.consider_whitening(embeddings)

        if self.embedding_transform_strategy == "none":
            pass
        elif self.embedding_transform_strategy == "repeat":
            embeddings = self.embedding_transform(embeddings)
            batch_size = embeddings.shape[0]
            # linear outputs a big embedding, reshape into a sequence of regular size embeddings.
            embeddings = embeddings.reshape((batch_size, self.num_repeat_tokens, -1))
        elif self.embedding_transform_strategy == "nearest_neighbors":
            # TODO
            raise NotImplementedError()
        else:
            raise ValueError(
                f"unknown embedding transformation strategy {self.embedding_transform_strategy}"
            )
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )
        return embeddings, attention_mask

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        # if "max_length" not in generation_kwargs:
        # generation_kwargs["max_length"] = generation_kwargs["min_length"] = inputs.get(
        #     "input_ids", inputs["embedder_input_ids"]
        # ).shape[1]
        # print("IM.generate:", generation_kwargs)
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs["embedder_input_ids"],
            embedder_attention_mask=inputs["embedder_attention_mask"],
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )
        if "decoder_input_ids" in inputs:
            return self.decoder.generate(
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
            return self.decoder.generate(
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
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        # embedder_token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask, embedder_token_type_ids
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=frozen_embeddings,
        )
        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )
