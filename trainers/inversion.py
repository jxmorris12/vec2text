import math
import random
from typing import Dict

import torch
import torch.nn as nn
import transformers

from models.logits_processors import ContrastiveLogitsProcessor
from trainers.base import BaseTrainer


class InversionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ######################################################
        self.model.precompute_whitening_params(self.get_train_dataloader())
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model

        # todo move generation strategy into model?
        self.generation_strategy = "none"  # contrastive, none
        self.contrastive_generation_num_rounds = 1
        self.contrastive_generation_alpha = 1.0
        self.contrastive_generation_gamma = 0.1
        self.contrastive_generation_hypothesis_temperature = 0
        self.contrastive_generation_hypothesis_num_samples = 1

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        if self.generation_strategy == "contrastive":
            return self.generate_contrastive(
                inputs=inputs, generation_kwargs=generation_kwargs
            )
        else:
            return self.model.generate(
                inputs=inputs, generation_kwargs=generation_kwargs
            )

    def generate_contrastive(
        self, inputs: Dict, generation_kwargs: Dict
    ) -> torch.Tensor:
        # TODO consider moving this method into the InversionTrainer– better separation of concerns?
        #
        contrastive_logits_processor = ContrastiveLogitsProcessor(
            model=self.model,
            alpha=self.contrastive_generation_alpha,
            gamma=self.contrastive_generation_gamma,
            hypothesis_temperature=self.contrastive_generation_hypothesis_temperature,
            hypothesis_num_samples=self.contrastive_generation_hypothesis_num_samples,
            inputs=inputs,
        )
        generation_kwargs["logits_processor"] = transformers.LogitsProcessorList(
            [
                contrastive_logits_processor,
            ]
        )
        # The following line tells HuggingFace to renormalize, since we apply a mask
        # and mess with the softmax output
        generation_kwargs["renormalize_logits"] = True

        for round_ in range(self.contrastive_generation_num_rounds):
            generations = self.model.generate(
                inputs=inputs, generation_kwargs=generation_kwargs
            )
            if round_ + 1 < self.contrastive_generation_num_rounds:
                contrastive_logits_processor.update_hypotheses(
                    hypotheses=generations,
                )
        return generations

    def _randomly_truncate_inputs(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # randomly truncates inputs. assumes last input is a pad token.
        assert not self.model.use_frozen_embeddings_as_input  # need to re-embed
        seq_length = inputs["input_ids"].shape[1]
        new_length = random.randint(1, seq_length - 1)
        pos = random.randint(0, seq_length - new_length)
        truncated_inputs = {k: v[:, pos : pos + new_length] for k, v in inputs.items()}
        truncated_inputs_with_pad = {
            k: torch.cat((truncated_inputs[k], inputs[k][:, -1, None]), dim=1)
            for k, v in inputs.items()
        }
        return truncated_inputs_with_pad

    def training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Performs a training step. we override to compute data-specific metrics.
        """
        # TODO: Log training metrics from below... (How to do with huggingface?)
        self._compute_data_metrics(inputs=inputs)
        if self.args.randomly_truncate_train_inputs:
            inputs = self._randomly_truncate_inputs(inputs=inputs)
        # self.log({ f"train/{k}": v for k,v in metrics.items() })
        return super().training_step(model, inputs)

    def evaluation_loop(
        self, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics.

        Override to compute ppl from eval loss.
        """
        output = super().evaluation_loop(*args, **kwargs)

        metric_key_prefix = kwargs["metric_key_prefix"]
        try:
            perplexity = math.exp(output.metrics[f"{metric_key_prefix}_loss"])
        except OverflowError:
            perplexity = float("inf")
        output.metrics["eval_perplexity"] = perplexity

        return output
