import copy
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

import aliases
from models import PrefixReranker
from run_args import TrainingArguments

from .base import BaseTrainer


class RerankingTrainer(BaseTrainer):
    def __init__(self, model: PrefixReranker, args: TrainingArguments):
        # We're training this reranking model to rerank outputs from
        # a model trained via the inversion trainer.
        # TODO argparse for alias.
        self.inversion_trainer = aliases.load_inversion_trainer_from_alias(
            alias="dpr_nq__msl32_beta"
        )
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
        self.rerank_length = 4  # TODO argparse & ablate
        self.beam_width = 4  # TODO argparse & ablate.
        self.reranking_method = "model"  # reranking with model (self) or embedder, or generate_before_embedder
        self.gen_kwargs = {
            "do_sample": False,
            "no_repeat_ngram_size": 3,
        }
        # TODO support gc
        # Need to train with same device as the inversion model to avoid weird errors.
        assert self.args.fp16 == self.inversion_trainer.args.fp16
        assert self.args.bf16 == self.inversion_trainer.args.bf16

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        return self.generate_with_reranking(
            inputs=inputs,
            L=self.rerank_length,
            B=self.beam_width,
            reranking_method=self.reranking_method,
        )

    def generate_continuations(
        self,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        frozen_embeddings: torch.Tensor,
        continuation_length: int,
        B: int,
    ) -> torch.Tensor:
        """Generates continuations for a prefix."""
        batch_size, prefix_length = prefix_input_ids.shape
        full_length = prefix_length + continuation_length

        bos_token_id = (
            self.embedder_tokenizer.bos_token_id or self.embedder_tokenizer.pad_token_id
        )
        bos_tokens = (
            torch.ones(
                (batch_size, 1), dtype=prefix_input_ids.dtype, device=self.args.device
            )
            * bos_token_id
        )
        prefix_input_ids = torch.cat((bos_tokens, prefix_input_ids), dim=1)
        ones = torch.ones(
            (batch_size, 1), dtype=prefix_attention_mask.dtype, device=self.args.device
        )
        prefix_attention_mask = torch.cat((ones, prefix_attention_mask), dim=1)
        # TODO properly handle decoder_attention_mask everywhere.
        # TODO properly handle min_length (don't force max length).
        with torch.no_grad():
            generated_text_ids = self.inversion_trainer.model.generate(
                inputs={
                    "decoder_input_ids": prefix_input_ids,
                    "decoder_attention_mask": prefix_attention_mask,
                    "embedder_input_ids": embedder_input_ids,
                    "embedder_attention_mask": embedder_attention_mask,
                    "frozen_embeddings": frozen_embeddings,
                },
                # TODO consider other gen_kwargs,
                # such as no_ngram_repeat or repetition penalty.
                generation_kwargs={
                    "min_length": full_length + 1,
                    "max_length": full_length + 1,
                    "early_stopping": False,
                    "do_sample": False,
                    "no_repeat_ngram_size": 3,
                    "num_beams": B,
                    "num_return_sequences": B,
                },
            )
        generated_text_ids = generated_text_ids[:, 1:]  # trim bos
        assert generated_text_ids.shape == (batch_size * B, full_length)
        return generated_text_ids

    def _contrastive_loss(
        self, scores: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Computes contrastive loss between two lists of vectors, e1 and e2
        -- float torch.Tensors.

        e1 may contain extra 'hard negative' tensors.
        """
        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=0.0)
        if loss.isnan():
            raise RuntimeError("Loss is nan!")

        # outputs = {
        #     "query_embedding": e1.detach().cpu(),
        #     "document_embedding": e2.detach().cpu(),
        # }
        # accuracy = (scores.argmax(dim=1) == diagonal_idxs).float().mean()
        # if self.args.use_wandb:
        #     # Log model-internal
        #     # todo: use self.log() instead?
        #     import wandb
        #     wandb.log({**metrics_q, **metrics_d, "accuracy": accuracy})

        return loss

    def sanity_decode(self):
        pass  # TODO implement with reranking :-)

    def generate_with_reranking(
        self,
        inputs: Dict[str, torch.Tensor],
        L: int,
        B: int,
        reranking_method: str,
    ) -> torch.Tensor:
        """Generates using inversion model and reranks using reranking model.

        (TODO rename L to rerank_length and B to beam width everywhere)

        Parameters (from rankgen):
            L - rerank length L,
            B - beam size B
            (N - number of samples per beam - not applicable with greedy)
        """
        batch_size, max_length = inputs["input_ids"].shape
        with torch.no_grad():
            frozen_embeddings = self.inversion_trainer.model.call_embedding_model(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            )

        # store copies for each thing in the beam so we can feed to cross-encoder
        frozen_embeddings = frozen_embeddings[:, None, :]
        frozen_embeddings = frozen_embeddings.repeat((1, B, 1))
        frozen_embeddings = frozen_embeddings.reshape((batch_size * B, 768))

        # first step
        bos_token_id = (
            self.inversion_trainer.model.embedder_tokenizer.bos_token_id
            or self.embedder_tokenizer.pad_token_id
        )
        eos_token_id = self.inversion_trainer.model.embedder_tokenizer.eos_token_id
        all_inputs = (
            torch.ones(
                (batch_size, 1),
                dtype=inputs["input_ids"].dtype,
                device=self.args.device,
            )
            * bos_token_id
        )
        # TMP: remove next line after debugging. this is cheating.
        # all_inputs = torch.cat(
        #     (all_inputs, inputs["embedder_input_ids"][:, 0:1]), dim=1
        # )
        # next steps
        while all_inputs.shape[1] < max_length:
            # add L to length
            hypothesis_length = min(max_length, all_inputs.shape[1] + L)
            gen_kwargs = copy.copy(self.gen_kwargs)
            gen_kwargs.update(
                {
                    "min_length": hypothesis_length,
                    "max_length": hypothesis_length,
                    "num_beams": B,
                    "num_return_sequences": B,
                }
            )
            # generate hypotheses
            attention_mask = torch.ones_like(all_inputs, device=self.args.device)
            hypotheses = self.inversion_trainer.model.generate(
                inputs={
                    "embedder_input_ids": inputs["embedder_input_ids"],
                    "embedder_attention_mask": inputs["embedder_attention_mask"],
                    # "frozen_embeddings": frozen_embeddings,
                    "decoder_input_ids": all_inputs,
                    "decoder_attention_mask": attention_mask,
                },
                generation_kwargs=gen_kwargs,
            )
            assert hypotheses.shape == (batch_size * B, hypothesis_length)
            hypothesis_attention_mask = torch.ones_like(
                hypotheses, device=hypotheses.device
            )
            # embed hypotheses
            eos_tokens = (
                torch.ones(
                    (batch_size * B, 1), dtype=torch.long, device=self.args.device
                )
                * eos_token_id
            )
            hypotheses_with_eos = torch.cat((hypotheses[:, 1:], eos_tokens), dim=1)

            if reranking_method == "model":
                scores = self.model.score_prefix_and_embedding(
                    prefix_ids=hypotheses_with_eos,
                    attention_mask=hypothesis_attention_mask,
                    embeddings=frozen_embeddings,
                )
            elif reranking_method == "generate_before_embedder":
                repeated_embedder_input_ids = (
                    inputs["embedder_input_ids"][:, None, :]
                    .repeat((1, B, 1))
                    .reshape(batch_size * B, -1)
                )
                repeated_embedder_attention_mask = (
                    inputs["embedder_attention_mask"][:, None, :]
                    .repeat((1, B, 1))
                    .reshape(batch_size * B, -1)
                )
                finished_hypotheses = self.inversion_trainer.model.generate(
                    inputs={
                        "embedder_input_ids": repeated_embedder_input_ids,
                        "embedder_attention_mask": repeated_embedder_attention_mask,
                        # "frozen_embeddings": frozen_embeddings,
                        "decoder_input_ids": hypotheses,
                        "decoder_attention_mask": hypothesis_attention_mask,
                    },
                    generation_kwargs={
                        "early_stopping": False,
                        "num_beams": 1,
                        "do_sample": False,
                        "no_repeat_ngram_size": 3,
                    },
                )
                finished_hypothesis_attention_mask = torch.ones_like(
                    finished_hypotheses, device=self.args.device
                )
                with torch.no_grad():
                    new_embeddings = self.inversion_trainer.model.call_embedding_model(
                        input_ids=finished_hypotheses,
                        attention_mask=finished_hypothesis_attention_mask,
                    )
                new_embeddings = new_embeddings.reshape(((batch_size, B, -1)))
                scores = torch.nn.CosineSimilarity(dim=2)(
                    new_embeddings.reshape((batch_size, B, -1)),
                    frozen_embeddings.reshape((batch_size, B, -1)),
                )
                scores = scores.reshape((batch_size * B,))
            elif reranking_method == "embedder":
                with torch.no_grad():
                    new_embeddings = self.inversion_trainer.model.call_embedding_model(
                        input_ids=hypotheses_with_eos,
                        attention_mask=hypothesis_attention_mask,
                    )
                new_embeddings = new_embeddings.reshape(((batch_size, B, -1)))
                scores = torch.nn.CosineSimilarity(dim=2)(
                    new_embeddings.reshape((batch_size, B, -1)),
                    frozen_embeddings.reshape((batch_size, B, -1)),
                )
                scores = scores.reshape((batch_size * B,))
            elif reranking_method == "none":
                scores = torch.zeros(
                    (batch_size, B), dtype=torch.float32, device=self.args.device
                )
                scores[:, 0] += 1
                scores = scores.reshape((batch_size * B,))
            else:
                raise ValueError(f"unknown reranking method {reranking_method}")
            assert scores.shape == (batch_size * B,)
            scores = scores.reshape((batch_size, B))

            # truncate beams
            hypotheses = hypotheses.reshape((batch_size, B, hypothesis_length))
            # all_inputs = hypotheses[:, 0, :]
            # print(scores.argmax(1).tolist())
            all_inputs = hypotheses[torch.arange(len(hypotheses)), scores.argmax(1)]

        return all_inputs

    def compute_loss(
        self,
        model: PrefixReranker,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """Computes contrastive loss using model generations and real text."""
        batch_size, seq_length = inputs["input_ids"].shape

        beam_width = self.beam_width

        # The 10 here comes from rankgen (arxiv.org/pdf/2205.09726.pdf)
        # where they generate continuations of a random length
        # from 10 to msl.
        min_continuation_length = 1
        assert min_continuation_length < seq_length

        # TODO consider sampling this from zipf distribution (shorter prefixes are more common?).
        prefix_length = random.randint(0, seq_length - min_continuation_length)
        max_continuation_length = seq_length - prefix_length

        inputs = {key: value.to(self.args.device) for key, value in inputs.items()}

        continuation_length = random.randint(
            min_continuation_length, max_continuation_length
        )

        prefixes = inputs["input_ids"][:, :prefix_length]
        prefix_attention_mask = inputs["attention_mask"][:, :prefix_length]
        true_continuations = inputs["input_ids"][
            :, : prefix_length + continuation_length
        ]

        with torch.no_grad():
            frozen_embeddings = self.inversion_trainer.model.call_embedding_model(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            )

        # TODO: Should we use beam search or nucleus sampling here?
        fake_continuations = self.generate_continuations(
            prefix_input_ids=prefixes,
            prefix_attention_mask=prefix_attention_mask,
            embedder_input_ids=inputs["embedder_input_ids"],
            embedder_attention_mask=inputs["embedder_attention_mask"],
            frozen_embeddings=frozen_embeddings,
            continuation_length=continuation_length,
            B=beam_width,
        )

        assert true_continuations.shape == (
            batch_size,
            prefix_length + continuation_length,
        )
        assert fake_continuations.shape == (
            beam_width * batch_size,
            prefix_length + continuation_length,
        )

        # Add EOS tokens and generate attention masks.
        pad_token_id = self.inversion_trainer.model.embedder_tokenizer.pad_token_id
        num_pad_tokens = seq_length - (prefix_length + continuation_length)
        pad_tokens = (
            torch.ones(
                (batch_size, num_pad_tokens),
                dtype=true_continuations.dtype,
                device=self.args.device,
            )
            * pad_token_id
        )
        eos_token_id = self.inversion_trainer.model.embedder_tokenizer.eos_token_id
        eos_tokens = (
            torch.ones(
                (batch_size, 1), dtype=true_continuations.dtype, device=self.args.device
            )
            * eos_token_id
        )
        true_continuations = torch.cat(
            (true_continuations, pad_tokens, eos_tokens), dim=1
        )

        continuations_attention_mask = torch.ones_like(
            true_continuations,
            device=self.args.device,
        )
        true_prefix_scores = self.model.score_prefix_and_embedding(
            prefix_ids=true_continuations,
            attention_mask=continuations_attention_mask,
            embeddings=frozen_embeddings,
        )
        pad_tokens = pad_tokens.repeat((beam_width, 1))
        eos_tokens = eos_tokens.repeat((beam_width, 1))
        fake_continuations = torch.cat(
            (fake_continuations, pad_tokens, eos_tokens), dim=1
        )
        continuations_attention_mask = torch.ones_like(
            fake_continuations, device=self.args.device
        )
        frozen_embeddings_repeated = (
            frozen_embeddings[:, None]
            .repeat((1, beam_width, 1))
            .reshape((batch_size * beam_width, 768))
        )
        fake_prefix_scores = self.model.score_prefix_and_embedding(
            prefix_ids=fake_continuations,
            attention_mask=continuations_attention_mask,
            embeddings=frozen_embeddings_repeated,
        )

        # create score matrix.
        true_prefix_scores = true_prefix_scores.reshape((batch_size, 1))
        fake_prefix_scores = fake_prefix_scores.reshape((batch_size, beam_width))
        scores = torch.cat((true_prefix_scores, fake_prefix_scores), dim=1)
        assert scores.shape == (
            batch_size,
            (1 + beam_width),
        )

        # TODO: fix.
        # This is so we don't penalize the model when one of the fake continuations is equal
        # to one of the true continuations. check for this and compute labels, then
        # pass those labels to contrastive loss.
        # all_continuations = torch.cat((true_continuations, fake_continuations), dim=0)
        # labels = (
        #     (true_continuations[:, None] == all_continuations[None, :])
        #     .all(dim=2)
        #     .float()
        # )
        # labels = labels / labels.sum(dim=1, keepdim=True)
        labels = torch.zeros((batch_size,), dtype=torch.long, device=self.args.device)

        return self._contrastive_loss(scores, labels)

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
        with torch.no_grad():
            loss = self.compute_loss(model=model, inputs=inputs)

        logits, labels = None, None
        return loss, logits, labels
