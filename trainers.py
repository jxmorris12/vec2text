import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import evaluate
import torch
import torch.nn as nn
import tqdm
import transformers

import aliases
from models import PrefixReranker
from run_args import TrainingArguments


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


class BaseTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.compute_metrics = self.compute_metrics_func
        self.metric_accuracy = evaluate.load("accuracy")
        self.metric_bleu = evaluate.load("sacrebleu")
        self.metric_bertscore = evaluate.load("bertscore")
        self.metric_rouge = evaluate.load("rouge")
        self.metric_meteor = evaluate.load("meteor")
        self.gen_kwargs = {
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 3,
        }

    def sanity_decode(self):
        """Encodes and decodes a string as a sanity check."""
        print("=" * 16, "Begin trainer sanity check", "=" * 16)
        input_string = "Twas brillig, and the slithy toves, Did gyre and gimble in the wabe, All mimsy were the borogoves, And the mome raths outgrabe."
        print("\tInput to encode ->", input_string)
        inputs = self.embedder_tokenizer(input_string, return_tensors="pt")
        inputs = inputs.to(self.args.device)
        regenerated = self.generate(
            inputs={
                "embedder_input_ids": inputs["input_ids"],
                "embedder_attention_mask": inputs["attention_mask"],
            },
            generation_kwargs=self.gen_kwargs,
        )
        output_string = self.embedder_tokenizer.decode(
            regenerated.flatten(), skip_special_tokens=True
        )
        print("\tDecoded output ->", output_string)
        print("=" * 16, "End trainer sanity check", "=" * 16)

    def _log_preds_table(
        self, table_key: str, decoded_preds: List[str], decoded_labels: List[str]
    ):
        if not self.args.use_wandb:
            return

        num_rows = 50
        idxs = random.choices(
            range(len(decoded_preds)), k=min(len(decoded_preds), num_rows)
        )

        data = []
        for idx in idxs:
            data.append([decoded_labels[idx], decoded_preds[idx]])

        import wandb

        table = wandb.Table(columns=["Original", "Decoded"], data=data)
        wandb.log({table_key: table})

    def _get_decoded_sequences(
        self, dataloader: torch.utils.data.DataLoader, n: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Iterates through eval dataset and does decoding.

        TODO: do this better. We shouldn't need to iterate through eval set twice
        but I don't want to copy 1000 lines of code to change their eval loop...

        Probably want custom eval eventually. Also this depends on eval data being
        in the same order which is annoying.
        """
        assert not self.model.training

        gen_kwargs = copy.copy(self.gen_kwargs)

        all_preds = []
        all_labels = []
        for step, inputs in enumerate(
            tqdm.tqdm(dataloader, desc="generating from val", leave=False)
        ):
            # https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
            inputs_cuda = {k: v.to(self.args.device) for k, v in inputs.items()}
            gen_kwargs["min_length"] = gen_kwargs["max_length"] = inputs[
                "input_ids"
            ].shape[1]
            with torch.no_grad():
                generated_text = self.generate(
                    inputs=inputs_cuda, generation_kwargs=gen_kwargs
                )
            all_preds.extend(generated_text.cpu().tolist())
            all_labels.extend(inputs["input_ids"].cpu().tolist())
            if len(all_preds) >= n:
                break

        return all_preds, all_labels

    def _compute_data_metrics(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        inputs_pad_tokens = (
            (inputs["input_ids"] == self.tokenizer.pad_token_id)
            .sum(dim=1)
            .float()
            .mean()
            .item()
        )
        embedder_inputs_pad_tokens = (
            (inputs["embedder_input_ids"] == self.embedder_tokenizer.pad_token_id)
            .sum(dim=1)
            .float()
            .mean()
            .item()
        )

        inputs_non_pad_tokens = inputs["input_ids"].shape[1] - inputs_pad_tokens
        embedder_inputs_non_pad_tokens = (
            inputs["input_ids"].shape[1] - embedder_inputs_pad_tokens
        )

        return {
            "encoder_decoder_inputs_pad_tokens": inputs_pad_tokens,
            "encoder_decoder_inputs_non_pad_tokens": inputs_non_pad_tokens,
            "embedder_inputs_pad_tokens": embedder_inputs_pad_tokens,
            "embedder_inputs_non_pad_tokens": embedder_inputs_non_pad_tokens,
        }

    def compute_metrics_func(self, eval_preds):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids

        assert len(labels), "got empty labels for eval"

        assert torch.tensor(preds).shape == torch.tensor(labels).shape
        # train_raw_bleu_result = self.metric_bleu.compute(
        #     predictions=decoded_train_preds, references=decoded_train_labels
        # )
        # train_bleu_result = { "bleu_score": train_raw_bleu_result["score"]}

        # preds have the same shape as the labels.
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        accuracy_result = self.metric_accuracy.compute(
            predictions=preds, references=labels
        )

        return {**accuracy_result}

    def _text_comparison_metrics(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        bleu_result = self.metric_bleu.compute(
            predictions=predictions, references=references
        )
        meteor_result = self.metric_meteor.compute(
            predictions=predictions, references=references
        )
        rouge_result = self.metric_rouge.compute(
            predictions=predictions, references=references
        )
        bertscore_result = self.metric_bleu.compute(
            predictions=predictions, references=references
        )

        return {
            "bleu_score": bleu_result["score"],
            "meteor_score": meteor_result["meteor"],
            "rouge_score": rouge_result[
                "rouge1"
            ],  # ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
            "bert_score": bertscore_result["score"],
        }

    def eval_generation_metrics(self, dataloader: torch.utils.data.DataLoader) -> Dict:
        # Get decoded text. Note that this is different than `preds`, which
        # is used to compute the loss.
        preds_sample, preds_sample_labels = self._get_decoded_sequences(
            dataloader=dataloader, n=1000
        )

        # Log BLEU, log table of text.
        decoded_preds = self.tokenizer.batch_decode(
            preds_sample, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            preds_sample_labels, skip_special_tokens=True
        )
        bleu_result = self._text_comparison_metrics(
            predictions=decoded_preds, references=decoded_labels
        )
        self._log_preds_table(
            table_key="val_text_preds",
            decoded_preds=decoded_preds,
            decoded_labels=decoded_labels,
        )

        print(decoded_preds[0])
        print(decoded_labels[0])
        print("\n\n")
        print(decoded_preds[1])
        print(decoded_labels[1])
        print("\n\n")
        print(decoded_preds[2])
        print(decoded_labels[2])

        # Compute sims of eval data using embedder.
        preds_sample = torch.tensor(preds_sample, device=self.args.device)[:128]
        preds_sample_labels = torch.tensor(
            preds_sample_labels, device=self.args.device
        )[:128]
        # Fix eos token on generated text.
        pad_token_id = self.embedder_tokenizer.pad_token_id
        eos_token_id = self.embedder_tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_tokens = (
                torch.ones(
                    (len(preds_sample), 1),
                    dtype=torch.long,
                    device=self.args.device,
                )
                * eos_token_id
            )
            preds_sample = torch.cat((preds_sample[:, 1:], eos_tokens), dim=1)
            assert preds_sample.shape == preds_sample_labels.shape
            # not true anymore, could be pad too.
            # assert (preds_sample_labels[:, -1] == eos_tokens).all()

        with torch.no_grad():
            preds_emb = self.call_embedding_model(
                input_ids=preds_sample,
                attention_mask=torch.ones_like(preds_sample, device=self.args.device),
            )
            labels_emb = self.call_embedding_model(
                input_ids=preds_sample_labels,
                attention_mask=(preds_sample_labels != pad_token_id),
            )
            emb_cos_sim = (
                torch.nn.CosineSimilarity(dim=1)(preds_emb, labels_emb).mean().item()
            )
            sim_result = {"emb_cos_sim": emb_cos_sim}

        # Log table for train data.
        # train_preds_sample, train_preds_sample_labels = self._get_decoded_sequences(
        #   dataloader=dataloader, n=100)
        # decoded_train_preds = self.tokenizer.batch_decode(
        #     train_preds_sample, skip_special_tokens=True
        # )
        # decoded_train_labels = self.tokenizer.batch_decode(
        #     train_preds_sample_labels, skip_special_tokens=True
        # )
        # self._log_preds_table(
        #     table_key="train_text_preds",
        #     decoded_preds=decoded_train_preds,
        #     decoded_labels=decoded_train_labels,
        # )

        metrics = {**bleu_result, **sim_result}
        return metrics

    def evaluation_loop(
        self, dataloader: torch.utils.data.DataLoader, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics.

        Override to compute ppl from eval loss.
        """
        output = super().evaluation_loop(dataloader=dataloader, *args, **kwargs)
        metric_key_prefix = kwargs["metric_key_prefix"]
        # TODO compute some data metrics here too.
        generation_metrics = self.eval_generation_metrics(dataloader=dataloader)
        generation_metrics = {
            f"{metric_key_prefix}_{k}": v for k, v in generation_metrics.items()
        }
        output.metrics.update(generation_metrics)
        return output


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
