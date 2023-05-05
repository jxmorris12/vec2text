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

        # todo move generation strategy into model?
        self.generation_strategy = "contrastive"  # contrastive, none

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        if self.generation_strategy == "contrastive":
            return self.generate_contrastive(inputs=inputs)
        else:
            return self.model.generate(
                inputs=inputs, generation_kwargs=generation_kwargs
            )

    def generate_contrastive(self, inputs: Dict) -> torch.Tensor:
        # TODO consider moving this method into the InversionTrainer– better separation of concerns?
        #
        alpha = 0.0
        gamma = 0.01
        print("contrastive alpha =", alpha)
        batch_size, seq_length = inputs["input_ids"].shape
        with torch.no_grad():
            e = self.model.call_embedding_model(
                input_ids=inputs["embedder_input_ids"],
                attention_mask=inputs["embedder_attention_mask"],
            )
        #
        hypotheses = self.model.generate(
            inputs=inputs,
            generation_kwargs={
                "max_length": seq_length,
                "early_stopping": True,
                "num_beams": 1,
                "do_sample": False,
            },
        )
        eos_token_id = self.embedder_tokenizer.eos_token_id
        eos_tokens = (
            torch.ones((batch_size, 1), dtype=torch.long, device=self.args.device)
            * eos_token_id
        )
        hypotheses_with_eos = torch.cat((hypotheses[:, 1:], eos_tokens), dim=1)
        hypothesis_attention_mask = torch.ones_like(
            hypotheses_with_eos, device=self.args.device
        )
        with torch.no_grad():
            hypothesis_e = self.model.call_embedding_model(
                input_ids=hypotheses_with_eos,
                attention_mask=hypothesis_attention_mask,
            )
        #
        bos_id = self.embedder_tokenizer.pad_token_id
        gen_text_ids = (
            torch.ones((batch_size, 1), dtype=torch.long, device=self.args.device)
            * bos_id
        )
        while gen_text_ids.shape[1] < seq_length:
            with torch.no_grad():
                outputs = self.model(
                    embedder_input_ids=None,
                    embedder_attention_mask=None,
                    frozen_embeddings=e,
                    decoder_input_ids=gen_text_ids,
                )
            next_token_logits = outputs.logits[:, -1]
            #
            with torch.no_grad():
                bad_outputs = self.model(
                    embedder_input_ids=None,
                    embedder_attention_mask=None,
                    frozen_embeddings=hypothesis_e,
                    decoder_input_ids=gen_text_ids,
                )
            bad_token_logits = bad_outputs.logits[:, -1] * gamma
            #
            diff_logits = next_token_logits.log_softmax(
                1
            ) - bad_token_logits.log_softmax(1)
            #
            next_token_probs = next_token_logits.softmax(-1)
            V_mask = (
                next_token_probs
                >= alpha * next_token_probs.max(dim=1, keepdim=True).values
            )
            diff_logits = torch.where(V_mask, diff_logits, diff_logits - 10**10)
            #
            next_token_ids = diff_logits.argmax(-1)  # greedy sampling
            gen_text_ids = torch.cat((gen_text_ids, next_token_ids[:, None]), dim=-1)
        return gen_text_ids

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
