import copy
import math
import random
import statistics
from typing import Dict, List, Tuple

import evaluate
import torch
import tqdm
import transformers


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
        
        # preds have the same shape as the labels.
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        accuracy_result = self.metric_accuracy.compute(
            predictions=preds, references=labels
        )

        return {**accuracy_result}

    def _text_comparison_metrics(
        self,
        predictions_ids: List[List[int]],
        predictions_str: List[str],
        references_ids: List[List[int]],
        references_str: List[str],
    ) -> Dict[str, float]:
        assert len(predictions_ids) == len(references_ids)
        assert len(predictions_ids) == len(predictions_str)
        assert len(predictions_str) == len(references_str)
        num_preds = len(predictions_ids)
        if not num_preds:
            return {}
        
        ###########################################################
        # TODO: Optimize this code
        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0
        for i in range(num_preds):
            true_words = set(references_ids[i]) - {0, 1, -100}
            pred_words = set(predictions_ids[i]) - {0, 1, -100}

            TP = len(true_words & pred_words)
            FP = len(true_words) - len(true_words & pred_words)
            FN = len(pred_words) - len(true_words & pred_words)

            precision = (TP) / (TP + FP + 1e-20)
            recall    = (TP) / (TP + FN + 1e-20)
            
            try:
                f1 = (2 * precision * recall) / (precision + recall + 1e-20)
            except ZeroDivisionError:
                f1 = 0.0

            precision_sum += precision
            recall_sum += recall
            f1_sum += f1

        set_token_metrics = {
            "token_set_precision": (precision_sum / num_preds),
            "token_set_recall":    (recall_sum / num_preds),
            "token_set_f1":        (f1_sum / num_preds),
        }
        ############################################################
        bleu_result = self.metric_bleu.compute(
            predictions=predictions_str, references=references_str
        )
        meteor_result = self.metric_meteor.compute(
            predictions=predictions_str, references=references_str
        )
        rouge_result = self.metric_rouge.compute(
            predictions=predictions_str, references=references_str
        )
        bertscore_result = self.metric_bertscore.compute(
            predictions=predictions_str, references=references_str, lang="en"
        )
        gen_metrics = {
            "bleu_score": bleu_result["score"],
            "meteor_score": meteor_result["meteor"],
            "rouge_score": rouge_result[
                "rouge1"
            ],  # ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
            "bert_score": statistics.fmean(bertscore_result["f1"]),
        }
        return { **set_token_metrics, **gen_metrics }

    def eval_generation_metrics(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        # Get decoded text. Note that this is different than `preds`, which
        # is used to compute the loss.
        preds_sample_list, preds_sample_labels_list = self._get_decoded_sequences(
            dataloader=dataloader, n=1000
        )

        # Log BLEU, log table of text.
        decoded_preds = self.tokenizer.batch_decode(
            preds_sample_list, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            preds_sample_labels_list, skip_special_tokens=True
        )
        bleu_result = self._text_comparison_metrics(
            predictions_ids=preds_sample_list,
            predictions_str=decoded_preds, 
            references_ids=preds_sample_labels_list,
            references_str=decoded_labels,
        )
        self._log_preds_table(
            table_key="val_text_preds",
            decoded_preds=decoded_preds,
            decoded_labels=decoded_labels,
        )

        if not len(decoded_preds):
            return {}
        print('[pred]', decoded_preds[0])
        print('[true]', decoded_labels[0])
        print("\n\n")
        print('[pred]', decoded_preds[1])
        print('[true]', decoded_labels[1])
        print("\n\n")
        print('[pred]', decoded_preds[2])
        print('[true]', decoded_labels[2])

        # Compute sims of eval data using embedder.
        preds_sample = torch.tensor(preds_sample_list, device=self.args.device)[:128]
        preds_sample_labels = torch.tensor(
            preds_sample_labels_list, device=self.args.device
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
                attention_mask=torch.ones_like(preds_sample_labels),
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
