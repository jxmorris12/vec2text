import functools
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import torch
import torch.nn as nn
import transformers

from vec2text.models import CorrectorEncoderModel
from vec2text.models.model_utils import freeze_params
from vec2text.run_args import TrainingArguments
from vec2text.utils import dataset_map_multi_worker

from .base import BaseTrainer
from .inversion import InversionTrainer

logger = logging.getLogger(__name__)


class Corrector(BaseTrainer):
    """Trains an encoder model to generate embeddings that recursively correct of an
    InversionTrainer.
    """

    train_dataset: datasets.Dataset
    eval_dataset: Dict[str, datasets.Dataset]
    # TODO: don't assume that the encoder has to have the same tokenizer as the encoder_decoder
    # or embedder model.

    _hypothesis_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    # If set, only take hypothesis if it improves our distance to ground-truth.
    return_best_hypothesis: bool = False

    # Initialize from this hypothesis, if set
    initial_hypothesis_str: Optional[str] = None

    def __init__(
        self,
        model: CorrectorEncoderModel,
        inversion_trainer: InversionTrainer,
        args: Optional[TrainingArguments],
        **kwargs,
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
            **kwargs,
        )
        self.tokenizer = self.inversion_trainer.model.tokenizer
        self.embedder_tokenizer = self.inversion_trainer.model.embedder_tokenizer
        self.embedder = self.inversion_trainer.embedder
        self.call_embedding_model = self.inversion_trainer.model.call_embedding_model

        self.initial_hypothesis_str = None

        # Number of steps of self-correction
        self.num_gen_recursive_steps = 1
        self.sequence_beam_width = 1

        # If set, return closest (in embedding space) hypothesis we see during generation
        self.return_best_hypothesis = False

        # Need to train with same device as the inversion model to avoid weird errors.
        assert self.args.fp16 == self.inversion_trainer.args.fp16
        assert self.args.bf16 == self.inversion_trainer.args.bf16

    def evaluation_loop(
        self, dataloader: torch.utils.data.DataLoader, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics.

        Override to compute ppl from eval loss.
        """
        self.inversion_trainer.model.to(self.args.device)

        metric_key_prefix = kwargs["metric_key_prefix"]
        output = super().evaluation_loop(dataloader=dataloader, *args, **kwargs)  # type: ignore
        if metric_key_prefix in {"eval_msmarco", "eval_nq"}:
            n_rounds = 5
            self.num_gen_recursive_steps = n_rounds
            multi_round_generation_metrics = self.eval_generation_metrics(
                dataloader=dataloader
            )
            multiround_generation_metrics = {
                f"{metric_key_prefix}_{n_rounds}round_{k}": v
                for k, v in multi_round_generation_metrics.items()
            }
            output.metrics.update(multiround_generation_metrics)
            self.num_gen_recursive_steps = 1

        self.inversion_trainer.model.cpu()
        return output

    def _precompute_hypothesis_and_embedding(
        self,
        ds_inputs: Dict[str, torch.Tensor],
        collator=None,
    ) -> Dict[str, torch.Tensor]:
        assert not self.model.training
        inputs = collator.tokenizer.pad(
            {k: v for k, v in ds_inputs.items() if k != "labels"},
            padding=collator.padding,
            max_length=collator.max_length,
            pad_to_multiple_of=collator.pad_to_multiple_of,
            return_tensors=collator.return_tensors,
        ).to(self.args.device)

        (
            frozen_embeddings,
            hypothesis_input_ids,
            hypothesis_attention_mask,
            hypothesis_embedding,
        ) = self._get_hypothesis_uncached(inputs=inputs)
        ds_inputs["frozen_embeddings"] = frozen_embeddings.cpu()
        ds_inputs["hypothesis_embedding"] = hypothesis_embedding.cpu()

        # cut padding so we can batch by length later
        ds_inputs["hypothesis_input_ids"] = []
        ds_inputs["hypothesis_attention_mask"] = []
        for input_ids, attention_mask in zip(
            hypothesis_input_ids.cpu(), hypothesis_attention_mask.cpu()
        ):
            num_tokens = attention_mask.sum()
            ds_inputs["hypothesis_input_ids"].append(input_ids[: num_tokens + 1])
            ds_inputs["hypothesis_attention_mask"].append(
                attention_mask[: num_tokens + 1]
            )
        print("input_ids[0]:", self.tokenizer.decode(ds_inputs["input_ids"][0]))
        print(
            "hypothesis_input_ids[0]:",
            self.tokenizer.decode(ds_inputs["hypothesis_input_ids"][0]),
        )
        return ds_inputs

    def _preprocess_dataset_hypotheses(
        self, dataset: datasets.Dataset, filter_correct_examples: bool = False
    ) -> Tuple[datasets.Dataset, str]:
        #
        # In each model directory, we store a copy of the dataset with hypotheses
        # generated by the model that's checkpointed in this directory. This
        # won't scale well, but hopefully we don't do this with too many models,
        # and precomputing 5M hypotheses on A100 takes ~8 hours, so they're worth
        # storing.
        #
        # Note that the dataset fingerprint changes with calls to select()
        # so we won't overwrite the big dataset files when we use tiny subsets
        # during testing.
        cache_dir = os.environ["VEC2TEXT_CACHE"]
        assert os.path.exists(cache_dir)
        ####
        cache_path = os.path.join(cache_dir, f"{dataset._fingerprint}_hypotheses.cache")
        if not os.path.exists(cache_path):
            print(f"\t[{dataset.builder_name}] Saving hypotheses to path {cache_path}")

            dataset = dataset_map_multi_worker(
                dataset=dataset,
                map_fn=functools.partial(
                    self._precompute_hypothesis_and_embedding,
                    collator=self.data_collator,
                ),
                batched=True,
                batch_size=(self.args.train_batch_size * 2),
                desc="Precomputing hypotheses for data",
            )

            if filter_correct_examples:
                old_length = len(dataset)

                def embedding_is_not_correct(ex):
                    return (
                        ~torch.isclose(
                            ex["frozen_embeddings"].to(self.args.device),
                            ex["hypothesis_embedding"].to(self.args.device),
                        ).all(dim=1)
                    ).tolist()

                dataset = dataset.filter(
                    embedding_is_not_correct,
                    batched=True,
                    batch_size=1024,
                )
                print(f"filtered {old_length} datapoints to {len(dataset)}")
            dataset.save_to_disk(cache_path)
        else:
            logging.info("Loading hypotheses from path %s", cache_path)
            print(
                f"\t[{dataset.builder_name}] Loading hypotheses from path {cache_path}"
            )
            dataset = datasets.load_from_disk(cache_path)
        dataset.set_format("pt")
        return dataset, cache_path

    def precompute_hypotheses(self) -> None:
        """Generates and embeds hypotheses using `self.inversion_trainer`.

        Returns path to precomputed-and-saved train dataset, which is sometimes
        useful for outside processes.
        """
        logger.info("Precomputing frozen embedding & hypotheses before training")

        self.train_dataset, train_cache_path = self._preprocess_dataset_hypotheses(
            dataset=self.train_dataset, filter_correct_examples=True
        )
        for k, v in self.eval_dataset.items():
            self.eval_dataset[k], _ = self._preprocess_dataset_hypotheses(
                dataset=v, filter_correct_examples=False
            )

    def _inner_training_loop(self, *args, **kwargs):
        # Don't let tokenizers run in parallel mode.
        # os.environ["TOKENIZERS_PARALLELISM"] = "False"

        self.model.eval()
        self.model.to(self.args.device)
        self.inversion_trainer.model.to(next(self.model.parameters()).device)
        self.precompute_hypotheses()
        self.model.train()
        self.inversion_trainer.model.cpu()
        return super()._inner_training_loop(*args, **kwargs)

    def generate(
        self,
        inputs: Dict,
        generation_kwargs: Dict,
        num_recursive_steps: int = None,
        sequence_beam_width: int = None,
    ) -> torch.Tensor:
        """Generates text using self-correction.

        Args:
            inputs (Dict[str, torch.Tensor]): inputs for generation, like the input embedding, hypothesis,
                and hypothesis embedding
            generation_kwargs (Dict): dictionary of parameters for generation, will be passed on to the model
            sequence_beam_width (int): beam width for sequence-level beam search
        Returns:
            generated_ids (torch.Tensor): ids of generated text
        """
        try:
            frozen_embeddings = inputs["frozen_embeddings"]
            hypothesis_input_ids = inputs["hypothesis_input_ids"]
            hypothesis_attention_mask = inputs["hypothesis_attention_mask"]
            hypothesis_embedding = inputs["hypothesis_embedding"]
        except KeyError:
            (
                frozen_embeddings,
                hypothesis_input_ids,
                hypothesis_attention_mask,
                hypothesis_embedding,
            ) = self._get_hypothesis_uncached(inputs=inputs)

        # Add beam dimension:
        #       (batch, ...) -> (batch, beam, ...)
        inputs["frozen_embeddings"] = frozen_embeddings
        inputs["hypothesis_input_ids"] = hypothesis_input_ids
        inputs["hypothesis_attention_mask"] = hypothesis_attention_mask
        inputs["hypothesis_embedding"] = hypothesis_embedding
        # print("generating with sequence_beam_width:", (sequence_beam_width or self.sequence_beam_width))

        num_recursive_steps = num_recursive_steps or self.num_gen_recursive_steps
        sequence_beam_width = sequence_beam_width or self.sequence_beam_width
        num_recursive_steps_so_far = 0

        total_best_scores_seen = None  # Track best scores for early stopping

        while num_recursive_steps >= 1:
            gen_text_ids, hypothesis_embedding, best_scores = self._generate_with_beam(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                num_recursive_steps=num_recursive_steps,
                num_recursive_steps_so_far=num_recursive_steps_so_far,
                sequence_beam_width=sequence_beam_width,
            )
            inputs["hypothesis_input_ids"] = gen_text_ids
            inputs["hypothesis_attention_mask"] = (
                gen_text_ids != self.model.encoder_decoder.config.pad_token_id
            ).int()
            inputs["hypothesis_embedding"] = hypothesis_embedding
            # step counters
            num_recursive_steps -= 1
            num_recursive_steps_so_far += 1
            # early stopping
            if best_scores is not None:
                if (total_best_scores_seen is not None) and torch.isclose(
                    best_scores, total_best_scores_seen, atol=1e-3
                ):
                    print(
                        "scores stopped increasing! stopping early after",
                        num_recursive_steps_so_far,
                        "steps",
                    )
                    break
                best_scores = total_best_scores_seen

        return gen_text_ids

    def generate_with_hypotheses(
        self,
        inputs: Dict,
        generation_kwargs: Dict,
        num_recursive_steps: int = None,
        sequence_beam_width: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates text using self-correction. Works exactly like generate(), but returns all the intermediate hypotheses steps.

        Args:
            inputs (Dict[str, torch.Tensor]): inputs for generation, like the input embedding, hypothesis,
                and hypothesis embedding
            generation_kwargs (Dict): dictionary of parameters for generation, will be passed on to the model
            sequence_beam_width (int): beam width for sequence-level beam search
        Returns:
            generated_ids (List[torch.Tensor]): ids of generated text, for each hypothesis sequence
            hypothesis_embeddings (List[torch.Tensor]): embeddings of each hypothesis sequence
        """
        try:
            frozen_embeddings = inputs["frozen_embeddings"]
            hypothesis_input_ids = inputs["hypothesis_input_ids"]
            hypothesis_attention_mask = inputs["hypothesis_attention_mask"]
            hypothesis_embedding = inputs["hypothesis_embedding"]
        except KeyError:
            (
                frozen_embeddings,
                hypothesis_input_ids,
                hypothesis_attention_mask,
                hypothesis_embedding,
            ) = self._get_hypothesis_uncached(inputs=inputs)

        # Add beam dimension:
        #       (batch, ...) -> (batch, beam, ...)
        inputs["frozen_embeddings"] = frozen_embeddings
        inputs["hypothesis_input_ids"] = hypothesis_input_ids
        inputs["hypothesis_attention_mask"] = hypothesis_attention_mask
        inputs["hypothesis_embedding"] = hypothesis_embedding

        num_recursive_steps = num_recursive_steps or self.num_gen_recursive_steps
        sequence_beam_width = sequence_beam_width or self.sequence_beam_width
        num_recursive_steps_so_far = 0

        total_best_scores_seen = None  # Track best scores for early stopping

        ground_truth_embedding = inputs["hypothesis_embedding"]
        hypothesis_embeddings = [ground_truth_embedding]  # Track hypothesis embeddings

        hypothesis_ids = [inputs["hypothesis_input_ids"]]  # Track hypothesis ids

        while num_recursive_steps >= 1:
            gen_text_ids, hypothesis_embedding, best_scores = self._generate_with_beam(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                num_recursive_steps=num_recursive_steps,
                num_recursive_steps_so_far=num_recursive_steps_so_far,
                sequence_beam_width=sequence_beam_width,
            )
            inputs["hypothesis_input_ids"] = gen_text_ids
            inputs["hypothesis_attention_mask"] = (
                gen_text_ids != self.model.encoder_decoder.config.pad_token_id
            ).int()
            inputs["hypothesis_embedding"] = hypothesis_embedding
            # step counters
            num_recursive_steps -= 1
            num_recursive_steps_so_far += 1
            # early stopping

            if best_scores is not None:
                closest_idx = torch.argmax(best_scores)
                if (total_best_scores_seen is not None) and torch.isclose(
                    best_scores, total_best_scores_seen, atol=1e-3
                ):
                    print(
                        "scores stopped increasing! stopping early after",
                        num_recursive_steps_so_far,
                        "steps",
                    )
                    break
                best_scores = total_best_scores_seen
            else:
                closest_idx = 0

            hypothesis_embeddings.append(hypothesis_embedding[closest_idx].unsqueeze(0))
            hypothesis_ids.append(gen_text_ids[closest_idx].unsqueeze(0))

        return hypothesis_ids, hypothesis_embeddings

    def _generate_with_beam(
        self,
        inputs: Dict,
        generation_kwargs: Dict,
        num_recursive_steps: int,
        num_recursive_steps_so_far: int,
        sequence_beam_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates text using self-correction.

        Args:
            inputs (Dict[str, torch.Tensor]): inputs for generation, like the input embedding, hypothesis,
                and hypothesis embedding
            generation_kwargs (Dict): dictionary of parameters for generation, will be passed on to the model
            num_recursive_steps (int): Number of remaining steps of recursion, used to know when to stop
            num_recusive_steps_so_far (int): Number of steps of recursion performed so far. This is how we
                can check if it's the initial hypothesis or not.
            sequence_beam_width (int): beam width for sequence-level beam search
        Returns:
            generated_ids (torch.Tensor): ids of generated text
        """
        assert num_recursive_steps >= 1
        frozen_embeddings = inputs["frozen_embeddings"]
        ################################################################################
        if not generation_kwargs["do_sample"]:
            num_return_sequences = max(
                sequence_beam_width, generation_kwargs.get("num_beams", 1)
            )
            generation_kwargs["num_beams"] = num_return_sequences
            generation_kwargs["num_return_sequences"] = num_return_sequences

        if (num_recursive_steps_so_far == 0) and (
            self.initial_hypothesis_str is not None
        ):
            # Support setting a string as the initial hypothesis (for ablations)
            logger.info(f"Using initial hypothesis: {self.initial_hypothesis_str}")
            # If set, uses this string as the hypothesis for step 0 of self-correction
            batch_size = frozen_embeddings.shape[0]
            gen_text_ids = (
                self.embedder_tokenizer(
                    [self.initial_hypothesis_str],
                    return_tensors="pt",
                    max_length=inputs["hypothesis_input_ids"].shape[1],
                    truncation=True,
                    padding="max_length",
                )["input_ids"]
                .repeat((batch_size, 1))
                .to(self.args.device)
            )
            # gen_text_ids = (
            #     torch.randint(
            #         low=1,
            #         high=self.embedder_tokenizer.vocab_size,
            #         size=(1, inputs["hypothesis_input_ids"].shape[1]),
            #         dtype=torch.long,
            #     )
            #     .repeat((batch_size, 1))
            #     .to(self.args.device)
            # )
            bos_token_id = self.model.encoder_decoder.config.decoder_start_token_id
            bos_token_ids = (
                torch.ones(
                    (batch_size, 1), dtype=torch.long, device=gen_text_ids.device
                )
                * bos_token_id
            )
            gen_text_ids = torch.cat((bos_token_ids, gen_text_ids[:, :-1]), dim=1)
        else:
            outputs = self.model.generate(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                return_dict_in_generate=True,
            )
            gen_text_ids = outputs.sequences

            # get scores for sequences to compute sequence-level likelihood.
            # https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075
            if "beam_indices" in outputs:
                with torch.no_grad():
                    transition_scores = (
                        self.model.encoder_decoder.compute_transition_scores(
                            outputs.sequences,
                            outputs.scores,
                            outputs.beam_indices,
                            normalize_logits=True,
                        )
                    )
            else:
                with torch.no_grad():
                    transition_scores = (
                        self.model.encoder_decoder.compute_transition_scores(
                            outputs.sequences, outputs.scores, normalize_logits=True
                        )
                    )
            length_penalty = self.model.encoder_decoder.generation_config.length_penalty
            output_length = (transition_scores < 0).sum(1)
            del outputs.scores
            gen_text_scores = transition_scores.sum(axis=1) / (
                output_length**length_penalty
            )  # log probs

        # Re-embed generated text so we can rerank, and track the best we've seen so far.
        hypothesis_embedding = self.embed_generated_hypothesis(input_ids=gen_text_ids)

        if num_recursive_steps_so_far == 0:
            batch_size = frozen_embeddings.shape[0]
        else:
            # after the first step, we've already copied frozen embeddings across the beam
            batch_size = int(frozen_embeddings.shape[0] / sequence_beam_width)

        best_scores = None
        #
        #   BEAM SEARCH
        #
        if gen_text_ids.shape[0] > batch_size:
            if sequence_beam_width == 1:
                # This is "regular" beam search.
                beam_width = int(gen_text_ids.shape[0] / batch_size)
                distances_per_beam = torch.nn.CosineSimilarity(dim=2)(
                    hypothesis_embedding.reshape((batch_size, beam_width, -1)),
                    inputs["frozen_embeddings"][:, None, :],
                )
                if self.return_best_hypothesis:
                    scores = distances_per_beam
                else:
                    scores = gen_text_scores.reshape((batch_size, beam_width))
                best_idx_in_beam = scores.argmax(1)
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size), best_idx_in_beam]
                gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))[
                    torch.arange(batch_size), best_idx_in_beam
                ]
                # Flatten again so we can do normal operations.
                gen_text_ids = gen_text_ids.reshape(
                    (batch_size * sequence_beam_width, -1)
                )
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size * sequence_beam_width, -1)
                )
            elif num_recursive_steps == 1:
                # Base case for sequence-level beam search.
                beam_width = int(gen_text_ids.shape[0] / batch_size)
                frozen_embeddings_per_beam = (
                    inputs["frozen_embeddings"][:, None, :]
                    .repeat((1, num_return_sequences, 1))
                    .reshape((batch_size, beam_width, -1))
                )
                distances_per_beam = torch.nn.CosineSimilarity(dim=2)(
                    hypothesis_embedding.reshape((batch_size, beam_width, -1)),
                    frozen_embeddings_per_beam,
                )
                if self.return_best_hypothesis:
                    scores = distances_per_beam
                else:
                    scores = gen_text_scores.reshape((batch_size, beam_width))
                best_idx_in_beam = scores.argmax(dim=1)
                # print("best_idx_in_beam:", best_idx_in_beam)
                # print("avg_distances:", distances_per_beam.mean(1).tolist(), "max_distances:", distances_per_beam.max(1).values.tolist())
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size), best_idx_in_beam]
                gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))[
                    torch.arange(batch_size), best_idx_in_beam
                ]
            else:
                # Now get top things in the beam like normal.
                beam_width = int(gen_text_ids.shape[0] / batch_size)
                assert (
                    beam_width % sequence_beam_width == 0
                ), "inner beam width must divide sequence beam width"

                if num_recursive_steps_so_far == 0:
                    # This is the first return for sequence-level beam search.
                    # First we have to copy the frozen embedding
                    frozen_embeddings_per_beam = (
                        inputs["frozen_embeddings"][:, None, :]
                        .repeat((1, num_return_sequences, 1))
                        .reshape((batch_size, num_return_sequences, -1))
                    )
                    inputs["frozen_embeddings"] = (
                        inputs["frozen_embeddings"][:, None, :]
                        .repeat((1, sequence_beam_width, 1))
                        .reshape((batch_size * sequence_beam_width, -1))
                    )
                else:
                    frozen_embeddings_per_beam = (
                        inputs["frozen_embeddings"][:, None, :]
                        .repeat((1, num_return_sequences, 1))
                        .reshape(
                            (batch_size, sequence_beam_width * num_return_sequences, -1)
                        )
                    )

                distances_per_beam = torch.nn.CosineSimilarity(dim=2)(
                    hypothesis_embedding.reshape((batch_size, beam_width, -1)),
                    frozen_embeddings_per_beam,
                )

                if self.return_best_hypothesis:
                    scores = distances_per_beam
                else:
                    scores = gen_text_scores.reshape((batch_size, beam_width))

                # print("scores:")
                # for t, s in zip(self.tokenizer.batch_decode(gen_text_ids, skip_special_tokens=True), scores.flatten().tolist()):
                #     print(f"\t- {s:2f}", t)
                # print()

                # take top *unique* things in beam.
                best_idx_in_beam_total = scores.topk(dim=1, k=beam_width).indices
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size, beam_width, -1)
                )
                gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))
                best_idx_in_beam = []
                for batch_idx in range(len(best_idx_in_beam_total)):
                    gen_text_set = set()  # track uniqueness
                    best_idx_in_beam.append([])
                    for j in best_idx_in_beam_total[batch_idx].tolist():
                        gen_text_i = tuple(gen_text_ids[batch_idx, j].tolist())
                        if gen_text_i not in gen_text_set:
                            gen_text_set.add(gen_text_i)
                            best_idx_in_beam[batch_idx].append(j)
                        if len(best_idx_in_beam[batch_idx]) == sequence_beam_width:
                            break
                best_idx_in_beam = torch.tensor(
                    best_idx_in_beam, device=best_idx_in_beam_total.device
                )
                # now take top unique things
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size)[:, None], best_idx_in_beam]
                gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))[
                    torch.arange(batch_size)[:, None], best_idx_in_beam
                ]

                # Flatten again so we can do normal operations.
                gen_text_ids = gen_text_ids.reshape(
                    (batch_size * sequence_beam_width, -1)
                )
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size * sequence_beam_width, -1)
                )

            # print scores for any type of beam search
            best_scores = scores.max(1).values.cpu()
        # make sure we reshape correctly
        # (can't do a shape check on gen_text_ids because of the dynamic length.)
        assert hypothesis_embedding.shape[-1] == inputs["frozen_embeddings"].shape[-1]

        return gen_text_ids, hypothesis_embedding, best_scores

    def get_frozen_embeddings(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            frozen_embeddings = self.inversion_trainer.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            )

        return frozen_embeddings.to(self.args.device)

    def embed_generated_hypothesis(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embeds a generated hypothesis. Has to remove EOS token and add BOS token
        at the beginning.
        """
        inputs_str = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        emb_input_ids = self.embedder_tokenizer(
            inputs_str,
            max_length=self.model.config.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(input_ids.device)
        return self.get_frozen_embeddings(
            embedder_input_ids=emb_input_ids.input_ids,
            embedder_attention_mask=emb_input_ids.attention_mask,
        )

    def _get_hypothesis_uncached(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "frozen_embeddings" in inputs:
            frozen_embeddings = inputs["frozen_embeddings"]
        elif "embedder_input_ids" in inputs:
            frozen_embeddings = self.get_frozen_embeddings(
                embedder_input_ids=inputs["embedder_input_ids"],
                embedder_attention_mask=inputs["embedder_attention_mask"],
            )
        else:
            assert (
                "input_ids" in inputs
            ), f"cannot generate hypothesis with input keys: {inputs.keys()}"
            frozen_embeddings = self.embed_generated_hypothesis(
                input_ids=inputs["input_ids"]
            )

        generation_kwargs = {
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 0,
            "max_length": self.model.config.max_seq_length,
        }

        hypothesis_input_ids = self.inversion_trainer.model.generate(
            inputs={
                "frozen_embeddings": frozen_embeddings,
            },
            generation_kwargs=generation_kwargs,
        )
        hypothesis_attention_mask = (
            hypothesis_input_ids != self.model.encoder_decoder.config.pad_token_id
        )
        hypothesis_embedding = self.embed_generated_hypothesis(
            input_ids=hypothesis_input_ids
        )
        return (
            frozen_embeddings,
            hypothesis_input_ids,
            hypothesis_attention_mask,
            hypothesis_embedding,
        )

    def compute_loss(
        self,
        model: CorrectorEncoderModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        batch_size, seq_length = inputs["input_ids"].shape

        try:
            frozen_embeddings = inputs["frozen_embeddings"]
            hypothesis_input_ids = inputs["hypothesis_input_ids"]
            hypothesis_attention_mask = inputs["hypothesis_attention_mask"]
            hypothesis_embedding = inputs["hypothesis_embedding"]
        except KeyError:
            (
                frozen_embeddings,
                hypothesis_input_ids,
                hypothesis_attention_mask,
                hypothesis_embedding,
            ) = self._get_hypothesis_uncached(inputs=inputs)

        labels = inputs["labels"]
        outputs = self.model(
            embedding=frozen_embeddings,
            hypothesis_embedding=hypothesis_embedding,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
            labels=labels,
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
            loss = self.compute_loss(model=model, inputs=inputs)

        logits, labels = None, None
        return loss, logits, labels

    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        """Edit keys posthumously on model load."""
        # Rename keys for backward compatibility w/ model trained before
        # we stopped sharing params between the ff layers
        if {
            "embedding_transform.3.weight",
            "embedding_transform.3.bias",
        } <= state_dict.keys():
            print(
                "Renaming keys",
                {"embedding_transform.2.weight", "embedding_transform.2.bias"},
                "for backward compatibility.",
            )
            state_dict["embedding_transform_1.0.weight"] = state_dict.pop(
                "embedding_transform.0.weight"
            )
            state_dict["embedding_transform_1.0.bias"] = state_dict.pop(
                "embedding_transform.0.bias"
            )
            state_dict["embedding_transform_1.3.weight"] = state_dict.pop(
                "embedding_transform.3.weight"
            )
            state_dict["embedding_transform_1.3.bias"] = state_dict.pop(
                "embedding_transform.3.bias"
            )
            #
            state_dict["embedding_transform_2.0.weight"] = state_dict[
                "embedding_transform_1.0.weight"
            ]
            state_dict["embedding_transform_2.0.bias"] = state_dict[
                "embedding_transform_1.0.bias"
            ]
            state_dict["embedding_transform_2.3.weight"] = state_dict[
                "embedding_transform_1.3.weight"
            ]
            state_dict["embedding_transform_2.3.bias"] = state_dict[
                "embedding_transform_1.3.bias"
            ]
            #
            state_dict["embedding_transform_3.0.weight"] = state_dict[
                "embedding_transform_1.0.weight"
            ]
            state_dict["embedding_transform_3.0.bias"] = state_dict[
                "embedding_transform_1.0.bias"
            ]
            state_dict["embedding_transform_3.3.weight"] = state_dict[
                "embedding_transform_1.3.weight"
            ]
            state_dict["embedding_transform_3.3.bias"] = state_dict[
                "embedding_transform_1.3.bias"
            ]
        return state_dict
