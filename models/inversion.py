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
class InversionModel(nn.Module):
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
        encoder_decoder: transformers.AutoModelForSeq2SeqLM,
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
        encoder_decoder_lora: bool = False,
        embedding_transform_strategy: str = "repeat",
        bottleneck_dim: int = 768,  # 128,
        token_decode_alpha: Optional[float] = None,
        embeddings_from_layer_n: Optional[int] = None,
    ):
        super().__init__()
        self.embedder = embedder
        self.encoder_decoder = encoder_decoder  # .to_bettertransformer()
        if encoder_decoder_lora:
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
                prepare_model_for_int8_training,
            )

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            print("Initializing LORA model with config:", peft_config)
            self.encoder_decoder = prepare_model_for_int8_training(self.encoder_decoder)
            self.encoder_decoder = get_peft_model(self.encoder_decoder, peft_config)
        ######################################################
        self.num_repeat_tokens = num_repeat_tokens

        if embedder_model_api:
            assert use_frozen_embeddings_as_input, "must precompute embeddings w/ api"
            # Hard-code OpenAI embedding dim
            self.embedder_dim = 1536
            bottleneck_dim = 1536
        elif use_frozen_embeddings_as_input:
            # temp hack to set fixed sentence embedding size to 512.
            # TODO do this in a smarter way (figure it out from data? or make it an arg.)
            self.embedder_dim = 512
        elif isinstance(self.embedder, SentenceTransformer):
            self.embedder_dim = self.embedder.get_sentence_embedding_dimension()
        else:
            self.embedder_dim = self.embedder.config.hidden_size

        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedder_no_grad = embedder_no_grad
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        self.whiten_embeddings = whiten_embeddings
        self.bottleneck_dim = bottleneck_dim
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),  # TODO consider dropout or normalization here.
            nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens),
        )
        if encoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.encoder)
        if decoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.decoder)
            disable_dropout(self.encoder_decoder.lm_head)
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

    def precompute_whitening_params(self, train_dataloader):
        if not self.whiten_embeddings:
            return
        self.embedder.to(device)
        n_sample = 500_000  # TODO argparse for this.
        n_points = 0
        embeddings = []
        for inputs in tqdm.tqdm(
            train_dataloader, desc="computing initial embeddings for whitening"
        ):
            n_points += len(inputs["embedder_input_ids"])
            if self.use_frozen_embeddings_as_input:
                frozen_embedding = inputs["frozen_embeddings"]
            else:
                with torch.no_grad():
                    frozen_embedding = self.call_embedding_model(
                        input_ids=inputs["embedder_input_ids"].to(device),
                        attention_mask=inputs["embedder_attention_mask"].to(device),
                    )
            embeddings.append(frozen_embedding.cpu())
            if n_points >= 200_000:  # TODO argparse for this
                break
        embeddings = torch.cat(embeddings, dim=0)
        logger.info("[whitening] mean & sample")
        mu = torch.mean(embeddings, dim=0, keepdim=True)
        embeddings_sample = embeddings[:n_sample]
        logger.info("[whitening] cov")
        cov = torch.mm((embeddings_sample - mu).t(), embeddings_sample - mu)
        logger.info("[whitening] SVD")
        u, s, vt = torch.svd(cov)
        logger.info("[whitening] computing W")
        W = torch.mm(u, torch.diag(1 / torch.sqrt(s)))
        self.whitening_mu = mu.to(device)
        self.whitening_W = W.to(device)

    def consider_whitening(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.whiten_embeddings:
            return embeddings
        return torch.mm(embeddings - self.whitening_mu, self.whitening_W)

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
            raise ValueError(f"invalid freezing strategy {freeze_strategy}")

    @property
    def embedder_device(self) -> torch.device:
        return next(self.embedder.parameters()).device

    def _process_embedder_output(
        self,
        outputs: transformers.modeling_outputs.BaseModelOutput,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and (outputs.pooler_output is not None):
            return outputs.pooler_output
        else:
            if self.embeddings_from_layer_n is not None:
                assert hasattr(
                    outputs, "hidden_states"
                ), "output missing hidden states - did you remember to initialize the model with output_hidden_states=True?"
                hidden_state = outputs.hidden_states[self.embeddings_from_layer_n]
            else:
                hidden_state = outputs.last_hidden_state
            # embeddings = model_output
            embeddings = mean_pool(hidden_state, attention_mask)
            # embeddings = max_pool(model_output, attention_mask)
            # embeddings = stack_pool(model_output, attention_mask)
            return embeddings

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        # token_type_ids: Optional[torch.Tensor] = None, # not used
    ) -> torch.Tensor:
        # print("** call_embedding_model")
        if self.embedder_no_grad:
            self.embedder.eval()

        if self.embedder_fake_with_zeros:
            batch_size = input_ids.shape[0]
            return torch.zeros(
                (batch_size, self.embedder_dim),
                dtype=torch.float32,
                device=self.embedder_device,
            )
        elif self.embedder_model_api:
            return embed_api(
                input_ids=input_ids,
                embedder_tokenizer=self.embedder_tokenizer,
                api_name=self.embedder_model_api,
            )
        elif isinstance(self.embedder, SentenceTransformer):
            # sentence-transformers is kind of really annoying
            model_output = self.embedder(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            embeddings = model_output["sentence_embedding"]
        else:
            model_output = self.embedder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            embeddings = self._process_embedder_output(model_output, attention_mask)
        return embeddings

    def embed_and_project(
        self,
        embedder_input_ids: Optional[torch.Tensor],
        embedder_attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("** embed_and_project")
        assert not ((embedder_input_ids is None) and (frozen_embeddings is None))
        if self.use_frozen_embeddings_as_input or (embedder_input_ids is None):
            assert (
                frozen_embeddings is not None
            ), "specified to train on frozen embeddings but none were provided"
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
            return self.encoder_decoder.generate(
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
            return self.encoder_decoder.generate(
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
        # print("** calling encoder_Decoder()")
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )
