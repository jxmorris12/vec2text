import math
import random
from typing import Dict

import torch
import torch.nn as nn
import transformers

from .base import BaseTrainer


class InversionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ######################################################
        self.model.precompute_whitening_params(self.get_train_dataloader())
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

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
