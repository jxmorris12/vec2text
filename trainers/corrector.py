from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from models import JointEmbeddingTextEncoder
from models.model_utils import freeze_params
from run_args import TrainingArguments

from .base import BaseTrainer
from .inversion import InversionTrainer


class CorrectorTrainer(BaseTrainer):
    """Trains an encoder model to generate embeddings that recursively correct of an
    InversionTrainer.

    TODO don't assume that the encoder has to have the same tokenizer as the encoder_decoder
    or embedder model.
    """

    _hypothesis_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    def __init__(
        self,
        model: JointEmbeddingTextEncoder,
        inversion_trainer: InversionTrainer,
        args: TrainingArguments,
    ):
        # Freeze other model params
        freeze_params(inversion_trainer.model)
        # We're training this corrector model to correct outputs from
        # a model trained & loaded via the inversion trainer.
        self.inversion_trainer = inversion_trainer
        self.inversion_trainer.model.use_frozen_embeddings_as_input = True
        super().__init__(
            model=model,
            args=args,
            train_dataset=self.inversion_trainer.train_dataset,
            eval_dataset=self.inversion_trainer.eval_dataset,
            data_collator=self.inversion_trainer.data_collator,
        )
        self.tokenizer = self.inversion_trainer.model.tokenizer
        self.embedder_tokenizer = self.inversion_trainer.model.embedder_tokenizer
        self.call_embedding_model = self.inversion_trainer.model.call_embedding_model

        self._hypothesis_cache = {}

        # Need to train with same device as the inversion model to avoid weird errors.
        assert self.args.fp16 == self.inversion_trainer.args.fp16
        assert self.args.bf16 == self.inversion_trainer.args.bf16

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        with torch.no_grad():
            frozen_embeddings = self.inversion_trainer.model.call_embedding_model(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            )
            new_embeddings = self.model(
                embedding=frozen_embeddings,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        inputs["frozen_embeddings"] = new_embeddings

        return self.inversion_trainer.generate(
            inputs=inputs, generation_kwargs=generation_kwargs
        )

    def _get_frozen_embeddings(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        with torch.no_grad():
            frozen_embeddings = self.inversion_trainer.model.call_embedding_model(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            )
        return frozen_embeddings

    def _get_hypothesis_uncached(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_length = inputs["input_ids"].shape
        fake_embedder_input_ids = torch.ones(
            (batch_size, seq_length), device=self.args.device
        )
        fake_embedder_attention_mask = torch.ones(
            (batch_size, seq_length), device=self.args.device
        )
        frozen_embeddings = self._get_frozen_embeddings(inputs=inputs)
        # TODO: support generated outputs of varying length.
        hypothesis_input_ids = self.inversion_trainer.model.generate(
            inputs={
                "embedder_input_ids": fake_embedder_input_ids,
                "embedder_attention_mask": fake_embedder_attention_mask,
                "frozen_embeddings": frozen_embeddings,
            },
            generation_kwargs={
                "early_stopping": False,
                "num_beams": 1,
                "do_sample": False,
                "no_repeat_ngram_size": 3,
            },
        )
        eos_token_id = self.inversion_trainer.model.embedder_tokenizer.eos_token_id
        eos_tokens = (
            torch.ones((batch_size, 1), dtype=torch.long, device=self.args.device)
            * eos_token_id
        )
        # get rid of EOS token, add BOS token.
        hypothesis_input_ids = torch.cat(
            (hypothesis_input_ids[:, 1:], eos_tokens), dim=1
        )
        hypothesis_attention_mask = (
            hypothesis_input_ids != self.embedder_tokenizer.pad_token_id
        )
        return frozen_embeddings, hypothesis_input_ids, hypothesis_attention_mask

    def _get_hypothesis_cached(
        self,
        idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = [
            self._hypothesis_cache[i.item()] for i in idx
        ]  # [(a1, b1, c1), (a2, b2, c2)]
        restacked_values = list(zip(*values))
        restacked_values = [
            torch.stack(a).to(self.args.device) for a in restacked_values
        ]
        (
            frozen_embeddings,
            hypothesis_input_ids,
            hypothesis_attention_mask,
        ) = restacked_values
        return frozen_embeddings, hypothesis_input_ids, hypothesis_attention_mask

    def _cache_hypothesis(
        self,
        idx: torch.Tensor,
        frozen_embeddings: torch.Tensor,
        hypothesis_input_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
    ) -> None:
        for idx, a, b, c in zip(
            idx, frozen_embeddings, hypothesis_input_ids, hypothesis_attention_mask
        ):
            key = idx.cpu().item()
            val = (a.cpu(), b.cpu(), c.cpu())
            self._hypothesis_cache[key] = val

    def get_hypothesis_with_caching(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            (
                frozen_embeddings,
                hypothesis_input_ids,
                hypothesis_attention_mask,
            ) = self._get_hypothesis_cached(idx=inputs["idx"])
        except KeyError:
            (
                frozen_embeddings,
                hypothesis_input_ids,
                hypothesis_attention_mask,
            ) = self._get_hypothesis_uncached(inputs=inputs)
            self._cache_hypothesis(
                idx=inputs["idx"],
                frozen_embeddings=frozen_embeddings,
                hypothesis_input_ids=hypothesis_input_ids,
                hypothesis_attention_mask=hypothesis_attention_mask,
            )

        return frozen_embeddings, hypothesis_input_ids, hypothesis_attention_mask

    def compute_loss(
        self,
        model: JointEmbeddingTextEncoder,
        inputs: Dict[str, torch.Tensor],
        training: bool = True,
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """Computes contrastive loss using model generations and real text."""
        batch_size, seq_length = inputs["input_ids"].shape

        fake_embedder_input_ids = torch.ones(
            (batch_size, seq_length), device=self.args.device
        )
        fake_embedder_attention_mask = torch.ones(
            (batch_size, seq_length), device=self.args.device
        )

        if training:
            (
                frozen_embeddings,
                hypothesis_input_ids,
                hypothesis_attention_mask,
            ) = self.get_hypothesis_with_caching(inputs=inputs)
        else:
            (
                frozen_embeddings,
                hypothesis_input_ids,
                hypothesis_attention_mask,
            ) = self._get_hypothesis_uncached(inputs=inputs)

        # NOTE TO SELF: can't put embedder_input_ids here, that's cheating.
        new_embeddings = self.model(
            embedding=frozen_embeddings,
            input_ids=hypothesis_input_ids,
            attention_mask=hypothesis_attention_mask,
        )

        # TODO: support passing embedder_input_ids/attention_mask as None.
        outputs = self.inversion_trainer.model(
            embedder_input_ids=fake_embedder_input_ids,
            embedder_attention_mask=fake_embedder_attention_mask,
            labels=inputs["labels"],
            frozen_embeddings=new_embeddings,
        )
        return outputs.loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`. Called during self.evalaute()
        """
        inputs = {key: value.to(self.args.device) for key, value in inputs.items()}
        with torch.no_grad():
            loss = self.compute_loss(model=model, inputs=inputs, training=False)

        logits, labels = None, None
        return loss, logits, labels
