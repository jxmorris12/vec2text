import functools
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import torch
import torch.nn as nn
import transformers

from models import CorrectorEncoderModel, CorrectorModel
from models.logits_processors import EncourageTrueTokensLogitsProcessor
from models.model_utils import freeze_params
from run_args import TrainingArguments

from .base import BaseTrainer
from .inversion import InversionTrainer

logger = logging.getLogger(__name__)


class CorrectorTrainer(BaseTrainer):
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
        model: Union[CorrectorEncoderModel, CorrectorModel],
        inversion_trainer: InversionTrainer,
        args: TrainingArguments,
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
        self.call_embedding_model = self.inversion_trainer.model.call_embedding_model

        self.initial_hypothesis_str = None

        # Number of steps of self-correction
        self.num_gen_recursive_steps = 1
        self.sequence_beam_width = 1

        # If set, return closest (in embedding space) hypothesis we see during generation
        self.return_best_hypothesis = False

        # Initialize our model with pre-trained model params
        missing_keys, unexpected_keys = self.model.load_state_dict(
            self.inversion_trainer.model.state_dict(), strict=False
        )
        self.model.embedding_transform_1.load_state_dict(
            self.inversion_trainer.model.embedding_transform.state_dict(),
        )
        self.model.embedding_transform_2.load_state_dict(
            self.inversion_trainer.model.embedding_transform.state_dict(),
        )
        self.model.embedding_transform_3.load_state_dict(
            self.inversion_trainer.model.embedding_transform.state_dict(),
        )

        # Need to train with same device as the inversion model to avoid weird errors.
        assert self.args.fp16 == self.inversion_trainer.args.fp16
        assert self.args.bf16 == self.inversion_trainer.args.bf16

    def _precompute_hypothesis_and_embedding(
        self, ds_inputs: Dict[str, torch.Tensor], collator=None,
        cheat: bool = False
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
        ) = self._get_hypothesis_uncached(inputs=inputs, cheat=cheat)
        ds_inputs["frozen_embeddings"] = frozen_embeddings.cpu()
        ds_inputs["hypothesis_embedding"] = hypothesis_embedding.cpu()
        # cut padding so we can batch by length later
        ds_inputs["hypothesis_input_ids"] = []
        ds_inputs["hypothesis_attention_mask"] = []
        for input_ids, attention_mask in zip(
            hypothesis_input_ids.cpu(), hypothesis_attention_mask.cpu()
        ):
            num_tokens = attention_mask.sum()
            ds_inputs["hypothesis_input_ids"].append(input_ids[:num_tokens])
            ds_inputs["hypothesis_attention_mask"].append(attention_mask[:num_tokens])
        print("input_ids[0]:", self.tokenizer.decode(ds_inputs["input_ids"][0]))
        print("hypothesis_input_ids[0]:", self.tokenizer.decode(ds_inputs["hypothesis_input_ids"][0]))
        return ds_inputs

    def _preprocess_dataset(
        self, 
        dataset: datasets.Dataset,
        cheat: bool=False
    ) -> datasets.Dataset:
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
        root_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
        )
        #### TEMP HACK UNTIL I FIGURE OUT WHY INVERSION TRAINER OUTPUT DIR CHANGES IN DDP
        # model_dir = os.path.join(root_dir, self.inversion_trainer.args.output_dir)
        model_dir = "/home/jxm3/research/retrieval/inversion/saves/f9abd65db4c4823264b133816d08612f/8d34a936d8e5905fe900d96ed65ec156/"
        assert os.path.exists(model_dir)
        if cheat:
            model_dir = os.path.join(model_dir, "improved_hypotheses")
        # model_dir = "/home/jxm3/research/retrieval/inversion/saves/f9abd65db4c4823264b133816d08612f/9d4a4d4b36da188a6e9dcb9736262823"
        # model_dir = "/home/jxm3/research/retrieval/inversion/saves/f9abd65db4c4823264b133816d08612f/8d34a936d8e5905fe900d96ed65ec156"
        ####
        cache_path = os.path.join(model_dir, f"{dataset._fingerprint}_hypotheses.cache")
        if not os.path.exists(cache_path):
            logging.info("Computing hypotheses to save to path %s", cache_path)
            print(f"Saving hypotheses to path {cache_path}")
            dataset = dataset.map(
                functools.partial(
                    self._precompute_hypothesis_and_embedding,
                    collator=self.data_collator,
                    cheat=cheat,
                ),
                batched=True,
                batch_size=(self.args.train_batch_size * 4),
                desc="Precomputing hypotheses for data",
            )
            dataset.save_to_disk(cache_path)
        else:
            logging.info("Loading hypotheses from path %s", cache_path)
            print(f"Loading hypotheses from path {cache_path}")
            dataset = datasets.load_from_disk(cache_path)
        dataset.set_format("pt")
        return dataset

    def precompute_hypotheses(self) -> None:
        # TODO: Compare doing this with and without training mode enabled.
        logger.info("Precomputing frozen embedding & hypotheses before training")
        self.train_dataset = self._preprocess_dataset(
            cheat=self.args.cheat_on_train_hypotheses,
            dataset=self.train_dataset    
        )
        for k, v in self.eval_dataset.items():
            self.eval_dataset[k] = self._preprocess_dataset(dataset=v)

    def _inner_training_loop(self, *args, **kwargs):
        # Don't let tokenizers run in parallel mode.
        os.environ["TOKENIZERS_PARALLELISM"] = "False"

        self.model.eval()
        self.precompute_hypotheses()
        self.model.train()

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
            ################################################################################
            # # print("temp: starting from best hypothesis")
            # hypothesis_input_ids = inputs["best_hypothesis_input_ids"]
            # hypothesis_embedding = inputs["best_hypothesis_embedding"]
            # hypothesis_attention_mask = (hypothesis_input_ids != self.model.encoder_decoder.config.pad_token_id).int()
            ################################################################################
            hypothesis_input_ids = inputs["hypothesis_input_ids"]
            hypothesis_attention_mask = inputs["hypothesis_attention_mask"]
            hypothesis_embedding = inputs["hypothesis_embedding"]
            ################################################################################
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
        return self._generate_with_beam(
            inputs=inputs,
            generation_kwargs=generation_kwargs,
            num_recursive_steps=(num_recursive_steps or self.num_gen_recursive_steps),
            num_recursive_steps_so_far=0,
            sequence_beam_width=(sequence_beam_width or self.sequence_beam_width),
        )
    
    def _generate_with_beam(
        self,
        inputs: Dict,
        generation_kwargs: Dict,
        num_recursive_steps: int,
        num_recursive_steps_so_far: int,
        sequence_beam_width: int,
    ) -> torch.Tensor:
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
        max_length = inputs.get("input_ids", inputs["embedder_input_ids"]).shape[1]

        num_return_sequences = max(sequence_beam_width, generation_kwargs.get("num_beams", 1))
        generation_kwargs["num_beams"] = num_return_sequences
        generation_kwargs["num_return_sequences"] = num_return_sequences

        if ((num_recursive_steps_so_far == 0) and (self.initial_hypothesis_str is not None)):
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
        elif self.is_corrector_encoder:
            outputs = self.model.generate(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                return_dict_in_generate=True,
            )
            transition_scores = self.model.encoder_decoder.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True,
            )
            gen_text_ids = outputs.sequences
            # get scores for sequences
            # https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075

            if 'beam_indices' in outputs:
                transition_scores = self.model.encoder_decoder.compute_transition_scores(
                    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
                )
            else:
                transition_scores = self.model.encoder_decoder.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=False
                )
            length_penalty = self.model.encoder_decoder.generation_config.length_penalty
            output_length = (transition_scores < 0).sum(1)
            gen_text_scores = transition_scores.sum(axis=1) / (output_length**length_penalty) # log probs
        else:
            gen_text_ids = self.model.generate(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                embed_generated_hypothesis_func=self.embed_generated_hypothesis,
                return_dict_in_generate=False,
            )
            # Don't return <hypothesis><text> upon generation, just return <text>
            gen_text_ids = gen_text_ids[:, max_length:]

        # Re-embed generated text so we can rerank, and track the best we've seen so far.
        hypothesis_embedding = self.embed_generated_hypothesis(input_ids=gen_text_ids)

        if num_recursive_steps_so_far == 0:
            batch_size = frozen_embeddings.shape[0]
        else:
            # after the first step, we've already copied frozen embeddings across the beam
            batch_size = int(frozen_embeddings.shape[0] / sequence_beam_width)
        
        # print(">batch_size:", batch_size)
        # print("frozen_embeddings.shape[0]", frozen_embeddings.shape[0], "gen_text_ids.shape[0]", gen_text_ids.shape[0])
        # print("num_recursive_steps:", num_recursive_steps)
        if gen_text_ids.shape[0] > batch_size:
            if (sequence_beam_width == 1):
                # This occurs when num_beams and num_return_sequences are both greater
                # than 1. We rerank examples in the beam according to their closeness
                # to the ground-truth.
                beam_width = int(gen_text_ids.shape[0] / batch_size)
                distances_per_beam = torch.nn.CosineSimilarity(dim=2)(
                    hypothesis_embedding.reshape((batch_size, beam_width, -1)), 
                    inputs["frozen_embeddings"][:, None, :]
                )
                if self.return_best_hypothesis:
                    best_idx_in_beam = distances_per_beam.argmax(1)
                else:
                    best_idx_in_beam = gen_text_scores.reshape((batch_size, beam_width)).argmax(1)
                hypothesis_embedding = hypothesis_embedding.reshape((batch_size, beam_width, -1))[
                    torch.arange(batch_size), best_idx_in_beam]
                gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))[
                    torch.arange(batch_size), best_idx_in_beam]
                # Flatten again so we can do normal operations.
                gen_text_ids = gen_text_ids.reshape((batch_size * sequence_beam_width, -1))
                hypothesis_embedding = hypothesis_embedding.reshape((batch_size * sequence_beam_width, -1))
            elif (num_recursive_steps == 1):
                # Base case for sequence-level beam search.
                beam_width = int(gen_text_ids.shape[0] / batch_size)
                frozen_embeddings_per_beam = (
                    inputs["frozen_embeddings"][:, None, :]
                        .repeat((1, num_return_sequences, 1))
                        .reshape((batch_size, beam_width, -1))
                )
                distances_per_beam = torch.nn.CosineSimilarity(dim=2)(
                    hypothesis_embedding.reshape((batch_size, beam_width, -1)), 
                    frozen_embeddings_per_beam
                )
                if self.return_best_hypothesis:
                    best_idx_in_beam = distances_per_beam.argmax(1)
                else:
                    best_idx_in_beam = gen_text_scores.reshape((batch_size, beam_width)).argmax(1)
                # print("best_idx_in_beam:", best_idx_in_beam)
                # print("avg_distances:", distances_per_beam.mean(1).tolist(), "max_distances:", distances_per_beam.max(1).values.tolist())
                hypothesis_embedding = hypothesis_embedding.reshape((batch_size, beam_width, -1))[
                    torch.arange(batch_size), best_idx_in_beam]
                gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))[
                    torch.arange(batch_size), best_idx_in_beam]
            else:
                # Now get top things in the beam like normal
                beam_width = int(gen_text_ids.shape[0] / batch_size)
                assert beam_width % sequence_beam_width == 0, "inner beam width must divide sequence beam width"
                
                if num_recursive_steps_so_far == 0:
                    # This is the first return for sequence-level beam search.
                    # First we have to copy the frozen embedding
                    frozen_embeddings_per_beam = (
                        inputs["frozen_embeddings"]
                        [:, None, :]
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
                            .reshape((batch_size, sequence_beam_width * num_return_sequences, -1))
                    )

                distances_per_beam = torch.nn.CosineSimilarity(dim=2)(
                    hypothesis_embedding.reshape((batch_size, beam_width, -1)), 
                    frozen_embeddings_per_beam,
                )

                if self.return_best_hypothesis:
                    scores = distances_per_beam
                else:
                    scores = gen_text_scores.reshape((batch_size, beam_width))

                if beam_width == sequence_beam_width:
                    # first time around for sequence-level beam search.
                    best_idx_in_beam = scores.topk(dim=1, k=sequence_beam_width).indices
                    hypothesis_embedding = hypothesis_embedding.reshape((batch_size, beam_width, -1))[torch.arange(batch_size)[:, None], best_idx_in_beam]
                    gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))[torch.arange(batch_size)[:, None], best_idx_in_beam]
                else:
                    # take top *unique* things in beam.
                    best_idx_in_beam_total = scores.topk(dim=1, k=sequence_beam_width*sequence_beam_width).indices
                    hypothesis_embedding = hypothesis_embedding.reshape((batch_size, beam_width, -1))
                    gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))
                    best_idx_in_beam = []
                    for batch_idx in range(len(best_idx_in_beam_total)):
                        gen_text_set = set() # track uniqueness
                        best_idx_in_beam.append([])
                        for j in best_idx_in_beam_total[batch_idx].tolist():
                            gen_text_i = tuple(gen_text_ids[batch_idx, j].tolist())
                            if gen_text_i not in gen_text_set:
                                gen_text_set.add(gen_text_i)
                                best_idx_in_beam[batch_idx].append(j)
                            if len(best_idx_in_beam[batch_idx]) == sequence_beam_width:
                                break
                    best_idx_in_beam = torch.tensor(best_idx_in_beam, device=best_idx_in_beam_total.device)
                    # now take top unique things
                    hypothesis_embedding = hypothesis_embedding.reshape((batch_size, beam_width, -1))[torch.arange(batch_size)[:, None], best_idx_in_beam]
                    gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))[torch.arange(batch_size)[:, None], best_idx_in_beam]
                
                # Flatten again so we can do normal operations.
                gen_text_ids = gen_text_ids.reshape((batch_size * sequence_beam_width, -1))
                hypothesis_embedding = hypothesis_embedding.reshape((batch_size * sequence_beam_width, -1))
                # print("len(gen_text_ids):", len(gen_text_ids), "len(set(gen_text_ids)):", len(set(self.tokenizer.batch_decode(gen_text_ids, skip_special_tokens=True))))
                # print("gen_text_ids:", self.tokenizer.batch_decode(gen_text_ids, skip_special_tokens=True))

        # make sure we reshape correctly
        assert hypothesis_embedding.shape[-1] == inputs["frozen_embeddings"].shape[-1]

        if num_recursive_steps == 1:
            return gen_text_ids
        else:
            inputs["hypothesis_input_ids"] = gen_text_ids
            inputs["hypothesis_attention_mask"] = (
                gen_text_ids != self.model.encoder_decoder.config.pad_token_id
            ).int()
            inputs["hypothesis_embedding"] = hypothesis_embedding
            return self._generate_with_beam(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                num_recursive_steps=(num_recursive_steps - 1),
                num_recursive_steps_so_far=num_recursive_steps_so_far + 1,
                sequence_beam_width=sequence_beam_width,
            )

    @property
    def is_corrector_encoder(self):
        return isinstance(self.model, CorrectorEncoderModel)

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
        return frozen_embeddings

    def embed_generated_hypothesis(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embeds a generated hypothesis. Has to remove EOS token and add BOS token
        at the beginning.
        """
        bos_token_id = self.model.encoder_decoder.config.decoder_start_token_id
        eos_token_id = self.model.encoder_decoder.config.eos_token_id
        assert (input_ids[:, 0] == bos_token_id).all()
        batch_size = len(input_ids)
        eos_tokens = (
            torch.ones((batch_size, 1), dtype=torch.long, device=self.args.device)
            * eos_token_id
        )

        input_ids = torch.cat((input_ids[:, 1:], eos_tokens), dim=1)
        attention_mask = input_ids != self.model.encoder_decoder.config.pad_token_id
        return self.get_frozen_embeddings(
            embedder_input_ids=input_ids,
            embedder_attention_mask=attention_mask,
        )

    def _get_hypothesis_uncached(self, inputs: Dict[str, torch.Tensor], cheat: bool = False) -> torch.Tensor:
        batch_size, seq_length = inputs["embedder_input_ids"].shape
        fake_embedder_input_ids = torch.ones(
            (batch_size, seq_length), device=self.args.device
        )
        fake_embedder_attention_mask = torch.ones(
            (batch_size, seq_length), device=self.args.device
        )
        if "frozen_embeddings" in inputs:
            frozen_embeddings = inputs["frozen_embeddings"]
        else:
            frozen_embeddings = self.get_frozen_embeddings(
                embedder_input_ids=inputs["embedder_input_ids"],
                embedder_attention_mask=inputs["embedder_attention_mask"],
            )
        
        generation_kwargs = {
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 0,
            "max_length": seq_length,
        }
        if cheat:
            assert "input_ids" in inputs.keys(), "cannot encourage true tokens for train hypotheses without them set"
            true_tokens_logits_processor = EncourageTrueTokensLogitsProcessor(
                true_input_ids=inputs["input_ids"],
            )
            generation_kwargs["logits_processor"] = transformers.LogitsProcessorList([true_tokens_logits_processor])
            # The following line tells HuggingFace to renormalize
            # generation_kwargs["renormalize_logits"] = True

        # TODO: support generated outputs of varying length.
        # TODO consider other (multiple?) hypothesis generation conditions.
        hypothesis_input_ids = self.inversion_trainer.model.generate(
            inputs={
                "embedder_input_ids": fake_embedder_input_ids,
                "embedder_attention_mask": fake_embedder_attention_mask,
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
        model: CorrectorModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """Computes contrastive loss using model generations and real text."""
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

        if self.is_corrector_encoder:
            labels = inputs["labels"]
            outputs = self.model(
                embedding=frozen_embeddings,
                hypothesis_embedding=hypothesis_embedding,
                hypothesis_input_ids=hypothesis_input_ids,
                hypothesis_attention_mask=hypothesis_attention_mask,
                labels=labels,
            )
        else:
            shift_right = self.model.encoder_decoder._shift_right
            # Special training scheme for the decoder-based model
            if self.model.training and random.random() < 0.5:
                # Half the time, we feed in a 'null' hypothesis embedding
                # and train the model to decode good hypotheses. The other
                # half of the time, we train it to correct its own hypotheses
                # using 'bad' hypotheses from the previous model.
                hypothesis_embedding = self.model.null_hypothesis_embedding(
                    hypothesis_embedding
                )
                # Will look like [...label_input_ids, 1] and get right-shifted to
                # [0, ..input_ids] by the model.
                decoder_input_ids = shift_right(inputs["labels"])
                labels = inputs["labels"]
            else:
                # Will look like [...hypothesis_input_ids, 1, ...label_ids, 1]
                # and get right-shifted to [0, ...hypothesis_input_ids, 1, ...label_ids]
                # by the model.
                # Do this always during evaluation, and 50% of the time during training.
                decoder_input_ids = shift_right(
                    torch.cat((hypothesis_input_ids, inputs["labels"]), dim=1)
                )
                empty_tokens = (
                    torch.ones_like(
                        hypothesis_input_ids, device=hypothesis_input_ids.device
                    )
                    * -100
                )
                labels = torch.cat((empty_tokens, inputs["labels"]), dim=1)
            outputs = self.model(
                embedding=frozen_embeddings,
                hypothesis_embedding=hypothesis_embedding,
                decoder_input_ids=decoder_input_ids,
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
        if {"embedding_transform.3.weight", "embedding_transform.3.bias"} <= state_dict.keys():
            print("Renaming keys", {"embedding_transform.2.weight", "embedding_transform.2.bias"}, "for backward compatibility.")
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
            state_dict["embedding_transform_2.0.weight"] = state_dict["embedding_transform_1.0.weight"]
            state_dict["embedding_transform_2.0.bias"] = state_dict["embedding_transform_1.0.bias"]
            state_dict["embedding_transform_2.3.weight"] = state_dict["embedding_transform_1.3.weight"]
            state_dict["embedding_transform_2.3.bias"] = state_dict["embedding_transform_1.3.bias"]
            #
            state_dict["embedding_transform_3.0.weight"] = state_dict["embedding_transform_1.0.weight"]
            state_dict["embedding_transform_3.0.bias"] = state_dict["embedding_transform_1.0.bias"]
            state_dict["embedding_transform_3.3.weight"] = state_dict["embedding_transform_1.3.weight"]
            state_dict["embedding_transform_3.3.bias"] = state_dict["embedding_transform_1.3.bias"]
        return state_dict

