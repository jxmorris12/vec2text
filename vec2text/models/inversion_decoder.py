import copy
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from sentence_transformers import SentenceTransformer

from vec2text.models import InversionModel
from vec2text.models.config import InversionConfig
from vec2text.models.model_utils import load_embedder_and_tokenizer, load_tokenizer

logger = logging.getLogger(__name__)


# TODO: can we make this class a HF pretrained model so it works nicely with
# .push_to_hub(), etc.?
# TODO: Need config to subclass transformers.PreTrainedModel.
class InversionModelDecoderOnly(InversionModel):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.

    This class is how we train a baseline for our paper that's just GPT-2 conditioned on a single token
    embedding.
    """

    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer  # embedder's tokenizer
    decoder: transformers.AutoModelForCausalLM
    tokenizer: transformers.PreTrainedTokenizer  # encoder_decoder's tokenizer
    embedding_transform: nn.Module  # Module that transformers embedder output into encoder-decoder input
    bottleneck_dim: int  # Bottleneck dimension for embedding_transform
    embedder_dim: int  # Hidden dimension of embedding model
    embedder_no_grad: bool  # Disable gradients for embedding model
    embedder_fake_with_zeros: bool  # Whether to just provide zeros as input for encoder-decoder (unconditional)
    embedding_transform_strategy: str  # Way to transform bottleneck embedding into input for encoder-decoder
    use_frozen_embeddings_as_input: bool  # Whether to train/evaluate on frozen embeddings
    embedded_tokens: torch.Tensor  # used for decoding
    embedder_model_api: Optional[str]

    def __init__(
        self,
        config: InversionConfig,
    ):
        super(InversionModel, self).__init__(config=config)

        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name
        )
        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )
        embedder_model_api = config.embedder_model_api

        if "t5" in config.model_name_or_path:
            # special handling for loading decoder of t5 (just decoder from encoder-decoder model).
            decoder = transformers.T5ForConditionalGeneration.from_pretrained(
                config.model_name_or_path
            )
        else:
            decoder = transformers.AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path
            )
        self.embedder = embedder
        self.decoder = decoder

        embedder_no_grad = config.embedder_no_grad
        embedder_fake_with_zeros = config.embedder_fake_with_zeros
        use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input

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

        self.embedder_no_grad = embedder_no_grad
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        self.bottleneck_dim = bottleneck_dim
        self.embedding_transform = nn.Linear(
            self.embedder_dim, self.decoder.config.hidden_size
        )
        ######################################################
        self.tokenizer = tokenizer
        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        self.embedder_fake_with_zeros = embedder_fake_with_zeros
        self.embedding_transform_strategy = "repeat"  # "none" # "repeat"
        self.noise_level = 0
        self.embeddings_from_layer_n = None

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

        if self.embedding_transform_strategy == "none":
            pass
        elif self.embedding_transform_strategy == "repeat":
            embeddings = self.embedding_transform(embeddings)
            batch_size = embeddings.shape[0]
            # linear outputs a big embedding, reshape into a sequence of regular size embeddings.
            embeddings = embeddings.reshape((batch_size, 1, -1))
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
                input_ids=inputs["decoder_input_ids"],
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

    def forward(  # type: ignore
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # use gpt for seq2seq: chop last tokens. shift will
        # automatically happen since the prepended embedding
        # takes up a single slot on the left.
        if labels is not None:
            input_ids = input_ids[:, :-1]  # type: ignore
            attention_mask = attention_mask[:, :-1]  # type: ignore

        embed_inputs_embeds, embed_attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=frozen_embeddings,
        )

        input_embeddings_table = self.decoder.get_input_embeddings()
        inputs_embeds = torch.cat(
            (embed_inputs_embeds, input_embeddings_table(input_ids)), dim=1
        )
        attention_mask = torch.cat((embed_attention_mask, attention_mask), dim=1)

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
